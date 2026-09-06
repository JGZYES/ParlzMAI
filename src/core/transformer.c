#include "core/transformer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- 微型算子 ---- */

/* y = x / sqrt(mean(x^2)+eps) * w */
static void rmsnorm(const float *x, const float *w, float eps, int d, float *y) {
    float sq = 0.0f;
    for (int i = 0; i < d; i++) sq += x[i] * x[i];
    float inv = 1.0f / sqrtf(sq / (float)d + eps);
    for (int i = 0; i < d; i++) y[i] = x[i] * inv * w[i];
}

/* y[cols] = x[k] @ W[k,cols]（行主序 W） */
static void matvec(const float *x, const float *W, int k, int cols, float *y) {
    for (int j = 0; j < cols; j++) {
        float s = 0.0f;
        for (int i = 0; i < k; i++) s += x[i] * W[i * cols + j];
        y[j] = s;
    }
}

/* GELU (tanh 近似)，与 Python 一致 */
static void gelu(float *x, int n) {
    const float c = 0.7978845608028654f;
    const float a = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(c * (v + a * v * v * v)));
    }
}

/* ---- KV cache ---- */

void kv_init(KVState *kv, const Model *m) {
    memset(kv, 0, sizeof(*kv));
    kv->max_seq = m->config.max_seq_len;
    kv->n_layer = m->config.n_layer;
    size_t per = (size_t)kv->max_seq * m->config.n_head * m->config.head_dim;
    kv->k_cache = calloc((size_t)kv->n_layer, sizeof(float *));
    kv->v_cache = calloc((size_t)kv->n_layer, sizeof(float *));
    for (int l = 0; l < kv->n_layer; l++) {
        kv->k_cache[l] = malloc(per * sizeof(float));
        kv->v_cache[l] = malloc(per * sizeof(float));
    }
    kv->seq_len = 0;
}

void kv_free(KVState *kv) {
    if (!kv) return;
    for (int l = 0; l < kv->n_layer; l++) {
        free(kv->k_cache[l]);
        free(kv->v_cache[l]);
    }
    free(kv->k_cache);
    free(kv->v_cache);
    memset(kv, 0, sizeof(*kv));
}

void kv_reset(KVState *kv) { kv->seq_len = 0; }

void moe_embed(const Model *m, int token, int pos, float *x) {
    int d = m->config.d_model;
    const float *te = m->wte + (size_t)token * d;
    const float *pe = m->wpe + (size_t)pos * d;
    for (int i = 0; i < d; i++) x[i] = te[i] + pe[i];
}

/* ---- 稠密 FFN：x -> gelu(x@w1) @ w2 ---- */
static void dense_ffn(const Model *m, int l, const float *x, float *out) {
    int d = m->config.d_model;
    int f = m->config.d_ff;
    float *mid = malloc((size_t)f * sizeof(float));
    matvec(x, m->ffn_w1[l], d, f, mid);
    gelu(mid, f);
    matvec(mid, m->ffn_w2[l], f, d, out);
    free(mid);
}

/* ---- MoE：router softmax + top-k 专家加权和 ---- */
static void moe_ffn(const Model *m, int l, const float *x, float *out) {
    int d = m->config.d_model;
    int ne = m->config.moe_n_experts;
    int topk = m->config.moe_top_k;
    int de = m->config.d_expert;

    /* router logits & softmax */
    float *probs = malloc((size_t)ne * sizeof(float));
    matvec(x, m->router[l], d, ne, probs);

    float mx = probs[0];
    for (int e = 1; e < ne; e++) if (probs[e] > mx) mx = probs[e];
    float sum = 0.0f;
    for (int e = 0; e < ne; e++) { probs[e] = expf(probs[e] - mx); sum += probs[e]; }
    float inv = 1.0f / sum;
    for (int e = 0; e < ne; e++) probs[e] *= inv;

    /* top-k 选择（在概率上做简单选择）*/
    int *idx = malloc((size_t)topk * sizeof(int));
    for (int t = 0; t < topk; t++) {
        int best = -1;
        float bestp = -1.0f;
        for (int e = 0; e < ne; e++) {
            int taken = 0;
            for (int q = 0; q < t; q++) if (idx[q] == e) { taken = 1; break; }
            if (!taken && probs[e] > bestp) { bestp = probs[e]; best = e; }
        }
        idx[t] = best;
    }

    float *mid = malloc((size_t)de * sizeof(float));
    float *eo = malloc((size_t)d * sizeof(float));
    for (int i = 0; i < d; i++) out[i] = 0.0f;
    for (int t = 0; t < topk; t++) {
        int e = idx[t];
        matvec(x, m->exp_w1[l][e], d, de, mid);
        gelu(mid, de);
        matvec(mid, m->exp_w2[l][e], de, d, eo);
        float w = probs[e];
        for (int i = 0; i < d; i++) out[i] += w * eo[i];
    }
    free(probs);
    free(idx);
    free(mid);
    free(eo);
}

/* ---- 单位置前向 ---- */
void moe_forward(const Model *m, KVState *kv, int pos, const float *x, float *logits) {
    const ModelConfig *c = &m->config;
    int d = c->d_model, nh = c->n_head, hd = c->head_dim;
    int L = c->n_layer, V = c->vocab_size;

    float *a = malloc((size_t)d * sizeof(float));     /* norm out */
    float *q = malloc((size_t)(nh * hd) * sizeof(float));
    float *k = malloc((size_t)(nh * hd) * sizeof(float));
    float *v = malloc((size_t)(nh * hd) * sizeof(float));
    float *att = malloc((size_t)d * sizeof(float));   /* 注意力输出（暂存 o 前） */
    float *o = malloc((size_t)d * sizeof(float));
    float *ffn = malloc((size_t)d * sizeof(float));
    float *scores = malloc((size_t)(kv->max_seq) * sizeof(float));

    float *h = malloc((size_t)d * sizeof(float));
    memcpy(h, x, (size_t)d * sizeof(float));

    float inv_sqrt = 1.0f / sqrtf((float)hd);

    for (int l = 0; l < L; l++) {
        /* 1) norm1 */
        rmsnorm(h, m->l_norm1[l], c->rmsnorm_eps, d, a);

        /* 2) qkv */
        matvec(a, m->wq[l], d, d, q);
        matvec(a, m->wk[l], d, d, k);
        matvec(a, m->wv[l], d, d, v);

        /* 存 cache：index = pos*(nh*hd) + h*hd + t */
        {
            float *kc = kv->k_cache[l] + (size_t)pos * nh * hd;
            float *vc = kv->v_cache[l] + (size_t)pos * nh * hd;
            memcpy(kc, k, (size_t)(nh * hd) * sizeof(float));
            memcpy(vc, v, (size_t)(nh * hd) * sizeof(float));
        }

        /* 3) 自注意力（因果），每个 head 独立 */
        for (int hh = 0; hh < nh; hh++) {
            const float *qh = q + (size_t)hh * hd;
            /* scores[j] = (qh . k_j)/sqrt(hd) for j in 0..pos */
            for (int j = 0; j <= pos; j++) {
                const float *kj = kv->k_cache[l] + (size_t)j * nh * hd + (size_t)hh * hd;
                float s = 0.0f;
                for (int t = 0; t < hd; t++) s += qh[t] * kj[t];
                scores[j] = s * inv_sqrt;
            }
            float mxs = scores[0];
            for (int j = 1; j <= pos; j++) if (scores[j] > mxs) mxs = scores[j];
            float ss = 0.0f;
            for (int j = 0; j <= pos; j++) { scores[j] = expf(scores[j] - mxs); ss += scores[j]; }
            float is = 1.0f / ss;
            for (int j = 0; j <= pos; j++) scores[j] *= is;

            for (int t = 0; t < hd; t++) {
                float acc = 0.0f;
                for (int j = 0; j <= pos; j++) {
                    const float *vj = kv->v_cache[l] + (size_t)j * nh * hd + (size_t)hh * hd;
                    acc += scores[j] * vj[t];
                }
                att[(size_t)hh * hd + t] = acc;
            }
        }

        /* 4) o = att @ wo */
        matvec(att, m->wo[l], d, d, o);
        /* 5) residual */
        for (int i = 0; i < d; i++) h[i] += o[i];

        /* 6) norm2 */
        rmsnorm(h, m->l_norm2[l], c->rmsnorm_eps, d, a);

        /* 7) ffn / moe */
        if (c->moe_mask[l]) {
            moe_ffn(m, l, a, ffn);
        } else {
            dense_ffn(m, l, a, ffn);
        }
        for (int i = 0; i < d; i++) h[i] += ffn[i];
    }

    /* 最终 norm + lm_head */
    rmsnorm(h, m->norm_final, c->rmsnorm_eps, d, a);
    matvec(a, m->lm_head, d, V, logits);   /* lm_head 为 (d, vocab) 行主序 */

    free(a); free(q); free(k); free(v); free(att); free(o);
    free(ffn); free(scores); free(h);
}
