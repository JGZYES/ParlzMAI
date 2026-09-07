#include "model/llama.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "infer/sample.h"
#include "model/gguf.h"
#include "utils/logger.h"

/* ---- 微型算子 ---- */
static float siluf(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm(const float *x, const float *w, float eps, int d, float *y) {
    float sq = 0.0f;
    for (int i = 0; i < d; i++) sq += x[i] * x[i];
    float inv = 1.0f / sqrtf(sq / (float)d + eps);
    for (int i = 0; i < d; i++) y[i] = x[i] * inv * w[i];
}

/* y[cols] = x[k] @ W[k,cols] （行主序 W=(k,cols)） */
static void matvec(const float *x, const float *W, int k, int cols, float *y) {
    for (int j = 0; j < cols; j++) {
        float s = 0.0f;
        for (int i = 0; i < k; i++) s += x[i] * W[i * cols + j];
        y[j] = s;
    }
}

static void silu_inplace(float *x, int n) { for (int i = 0; i < n; i++) x[i] = siluf(x[i]); }

/* RoPE (NeoX half-rotate) 作用于一个 head 向量（长度 dim），位置 pos */
static void rope_inplace(const LlamaModel *m, float *v, int pos) {
    int half = m->c.head_dim / 2;
    const float *c = m->cos_cache[pos];
    const float *s = m->sin_cache[pos];
    for (int i = 0; i < half; i++) {
        float a = v[i], b = v[i + half];
        v[i]         = a * c[i] - b * s[i];
        v[i + half]  = b * c[i] + a * s[i];
    }
}

void llama_embed(const LlamaModel *m, int token, float *x) {
    const float *te = m->token_embd + (size_t)token * m->c.n_embd;
    memcpy(x, te, (size_t)m->c.n_embd * sizeof(float));
}

/* ---- MoE：router softmax → top-k → 重归一化 → SwiGLU 专家加权和 ---- */
static void moe_ffn(const LlamaModel *m, int l, const float *x, float *out) {
    int d = m->c.n_embd, ne = m->c.n_expert, used = m->c.n_expert_used, de = m->c.ffn_dim;
    float *probs = malloc((size_t)ne * sizeof(float));
    matvec(x, m->router[l], d, ne, probs);
    float mx = probs[0];
    for (int e = 1; e < ne; e++) if (probs[e] > mx) mx = probs[e];
    float sum = 0.0f;
    for (int e = 0; e < ne; e++) { probs[e] = expf(probs[e] - mx); sum += probs[e]; }
    for (int e = 0; e < ne; e++) probs[e] /= sum;

    int *idx = malloc((size_t)used * sizeof(int));
    for (int t = 0; t < used; t++) {
        int best = -1; float bp = -1.0f;
        for (int e = 0; e < ne; e++) {
            int taken = 0;
            for (int q = 0; q < t; q++) if (idx[q] == e) { taken = 1; break; }
            if (!taken && probs[e] > bp) { bp = probs[e]; best = e; }
        }
        idx[t] = best;
    }
    float wsum = 0.0f;
    for (int t = 0; t < used; t++) wsum += probs[idx[t]];

    float *g = malloc((size_t)de * sizeof(float));
    float *up = malloc((size_t)de * sizeof(float));
    float *eo = malloc((size_t)d * sizeof(float));
    for (int i = 0; i < d; i++) out[i] = 0.0f;
    for (int t = 0; t < used; t++) {
        int e = idx[t];
        float w = probs[e] / (wsum + 1e-9f);
        matvec(x, m->ffn_gate[l][e], d, de, g);      silu_inplace(g, de);
        matvec(x, m->ffn_up[l][e], d, de, up);
        for (int i = 0; i < de; i++) g[i] *= up[i];
        matvec(g, m->ffn_down[l][e], de, d, eo);
        for (int i = 0; i < d; i++) out[i] += w * eo[i];
    }
    free(probs); free(idx); free(g); free(up); free(eo);
}

/* ---- 稠密 SwiGLU：down(silu(gate(x)) * up(x)) ---- */
static void dense_ffn(const LlamaModel *m, int l, const float *x, float *out) {
    int d = m->c.n_embd, de = m->c.ffn_dim;
    float *g = malloc((size_t)de * sizeof(float));
    float *up = malloc((size_t)de * sizeof(float));
    matvec(x, m->df_gate[l], d, de, g);  silu_inplace(g, de);
    matvec(x, m->df_up[l], d, de, up);
    for (int i = 0; i < de; i++) g[i] *= up[i];
    matvec(g, m->df_down[l], de, d, out);
    free(g); free(up);
}

/* ---- 单位置前向 ---- */
void llama_forward(LlamaModel *m, int pos, const float *x, float *logits) {
    const LlamaConfig *c = &m->c;
    int d = c->n_embd, H = c->n_head, HK = c->n_head_kv, hd = c->head_dim;
    int hpk = H / HK;   /* 每组 kv 对应查询头数 */
    int L = c->n_layer, V = c->vocab;

    float *a  = malloc((size_t)d * sizeof(float));
    float *q  = malloc((size_t)(H * hd) * sizeof(float));
    float *k  = malloc((size_t)(HK * hd) * sizeof(float));
    float *v  = malloc((size_t)(HK * hd) * sizeof(float));
    float *att= malloc((size_t)(H * hd) * sizeof(float));
    float *o  = malloc((size_t)d * sizeof(float));
    float *ffn= malloc((size_t)d * sizeof(float));
    float *scores = malloc((size_t)(m->c.max_seq) * sizeof(float));

    float *h = malloc((size_t)d * sizeof(float));
    memcpy(h, x, (size_t)d * sizeof(float));
    float isq = 1.0f / sqrtf((float)hd);

    for (int l = 0; l < L; l++) {
        rmsnorm(h, m->attn_norm[l], c->rmsnorm_eps, d, a);
        matvec(a, m->attn_q[l], d, H * hd, q);
        matvec(a, m->attn_k[l], d, HK * hd, k);
        matvec(a, m->attn_v[l], d, HK * hd, v);

        for (int hh = 0; hh < H; hh++) rope_inplace(m, q + (size_t)hh * hd, pos);
        for (int hh = 0; hh < HK; hh++) rope_inplace(m, k + (size_t)hh * hd, pos);

        /* 存 KV cache */
        memcpy(m->k_cache[l] + (size_t)pos * HK * hd, k, (size_t)HK * hd * sizeof(float));
        memcpy(m->v_cache[l] + (size_t)pos * HK * hd, v, (size_t)HK * hd * sizeof(float));

        for (int hh = 0; hh < H; hh++) {
            int kvh = hh / hpk;
            const float *qh = q + (size_t)hh * hd;
            for (int j = 0; j <= pos; j++) {
                const float *kj = m->k_cache[l] + ((size_t)j * HK + kvh) * hd;
                float s = 0.0f;
                for (int t = 0; t < hd; t++) s += qh[t] * kj[t];
                scores[j] = s * isq;
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
                    const float *vj = m->v_cache[l] + ((size_t)j * HK + kvh) * hd;
                    acc += scores[j] * vj[t];
                }
                att[(size_t)hh * hd + t] = acc;
            }
        }
        matvec(att, m->attn_o[l], d, d, o);
        for (int i = 0; i < d; i++) h[i] += o[i];

        rmsnorm(h, m->ffn_norm[l], c->rmsnorm_eps, d, a);
        if (c->is_moe) moe_ffn(m, l, a, ffn); else dense_ffn(m, l, a, ffn);
        for (int i = 0; i < d; i++) h[i] += ffn[i];
    }

    rmsnorm(h, m->output_norm, c->rmsnorm_eps, d, a);
    matvec(a, m->output, d, V, logits);

    free(a); free(q); free(k); free(v); free(att); free(o); free(ffn); free(scores); free(h);
}

/* ---- 生成：自回归，输出 token id 到 out_ids（返回产出 token 数） ---- */
int llama_generate(LlamaModel *m, const int *input_ids, int n_input,
                   int max_new, float temperature, int top_k, int *out_ids) {
    const int V = m->c.vocab, maxseq = m->c.max_seq;
    float *logits = malloc((size_t)V * sizeof(float));
    float *x = malloc((size_t)m->c.n_embd * sizeof(float));
    int *ids = malloc((size_t)(maxseq + 8) * sizeof(int));

    int n = n_input < maxseq ? n_input : maxseq;
    for (int i = 0; i < n; i++) ids[i] = input_ids[i];
    for (int i = 0; i < n; i++) { llama_embed(m, ids[i], x); llama_forward(m, i, x, logits); }

    int pos = n, produced = 0;
    for (int t = 0; t < max_new; t++) {
        if (pos >= maxseq) break;
        int next = sample_top_k(logits, V, temperature, top_k);
        if (out_ids) out_ids[produced] = next;
        llama_embed(m, next, x);
        llama_forward(m, pos, x, logits);
        pos++;
        produced++;
    }
    free(logits); free(x); free(ids);
    return produced;
}

/* ---- 名字精确匹配 ---- */
static int parse_blk(const char *name, int *i, const char **suffix) {
    if (strncmp(name, "blk.", 4) != 0) return 0;
    const char *p = name + 4;
    if (*p < '0' || *p > '9') return 0;
    *i = 0;
    while (*p >= '0' && *p <= '9') { *i = *i * 10 + (*p - '0'); p++; }
    if (*p != '.') return 0;
    *suffix = p;
    return 1;
}

static int parse_expert(const char *name, int *i, int *e, const char **suffix) {
    static const char *pfx = ".ffn_experts.";
    const char *s;
    if (!parse_blk(name, i, &s)) return 0;
    if (strncmp(s, pfx, strlen(pfx)) != 0) return 0;
    const char *q = s + strlen(pfx);
    if (*q < '0' || *q > '9') return 0;
    *e = 0;
    while (*q >= '0' && *q <= '9') { *e = *e * 10 + (*q - '0'); q++; }
    if (*q != '.') return 0;
    *suffix = q;
    return 1;
}

/* 装载一个名为 name 的张量 buf（长度 n，浮点已反量化） */
static void load_w(LlamaModel *m, const char *name, const float *buf, uint64_t n) {
    LlamaConfig *c = &m->c;
    (void)n;
    if (strcmp(name, "token_embd.weight") == 0) { memcpy(m->token_embd, buf, (size_t)c->vocab * c->n_embd * 4); return; }
    if (strcmp(name, "output_norm.weight") == 0) { memcpy(m->output_norm, buf, (size_t)c->n_embd * 4); return; }
    if (strcmp(name, "output.weight") == 0) { memcpy(m->output, buf, (size_t)c->n_embd * c->vocab * 4); return; }

    int i, e;
    const char *s;
    if (parse_expert(name, &i, &e, &s)) {
        if (strcmp(s, ".ffn_gate.weight") == 0) memcpy(m->ffn_gate[i][e], buf, (size_t)c->n_embd * c->ffn_dim * 4);
        else if (strcmp(s, ".ffn_up.weight") == 0) memcpy(m->ffn_up[i][e], buf, (size_t)c->n_embd * c->ffn_dim * 4);
        else if (strcmp(s, ".ffn_down.weight") == 0) memcpy(m->ffn_down[i][e], buf, (size_t)c->ffn_dim * c->n_embd * 4);
        return;
    }
    if (parse_blk(name, &i, &s)) {
        if (strcmp(s, ".attn_norm.weight") == 0) memcpy(m->attn_norm[i], buf, (size_t)c->n_embd * 4);
        else if (strcmp(s, ".attn_q.weight") == 0) memcpy(m->attn_q[i], buf, (size_t)c->n_embd * c->n_head * c->head_dim * 4);
        else if (strcmp(s, ".attn_k.weight") == 0) memcpy(m->attn_k[i], buf, (size_t)c->n_embd * c->n_head_kv * c->head_dim * 4);
        else if (strcmp(s, ".attn_v.weight") == 0) memcpy(m->attn_v[i], buf, (size_t)c->n_embd * c->n_head_kv * c->head_dim * 4);
        else if (strcmp(s, ".attn_o.weight") == 0) memcpy(m->attn_o[i], buf, (size_t)c->n_embd * c->n_embd * 4);
        else if (strcmp(s, ".ffn_norm.weight") == 0) memcpy(m->ffn_norm[i], buf, (size_t)c->n_embd * 4);
        else if (strcmp(s, ".ffn_gate_inp.weight") == 0) memcpy(m->router[i], buf, (size_t)c->n_embd * c->n_expert * 4);
        else if (strcmp(s, ".ffn_gate.weight") == 0) memcpy(m->df_gate[i], buf, (size_t)c->n_embd * c->ffn_dim * 4);
        else if (strcmp(s, ".ffn_up.weight") == 0) memcpy(m->df_up[i], buf, (size_t)c->n_embd * c->ffn_dim * 4);
        else if (strcmp(s, ".ffn_down.weight") == 0) memcpy(m->df_down[i], buf, (size_t)c->ffn_dim * c->n_embd * 4);
        return;
    }
    fprintf(stderr, "[warn] 忽略未知权重: %s\n", name);
}

/* ---- 加载 GGUF ---- */
static int get_u32(const Gguf *g, const char *k, int def, int *out) {
    uint32_t v; if (gguf_meta_u32(g, k, &v)) { *out = (int)v; return 1; } *out = def; return 0;
}
static int get_f32(const Gguf *g, const char *k, float def, float *out) {
    if (gguf_meta_f32(g, k, out)) { return 1; }
    *out = def;
    return 0;
}

int llama_load(LlamaModel *m, const char *path) {
    memset(m, 0, sizeof(*m));
    Gguf g;
    if (gguf_open(&g, path) != 0) { fprintf(stderr, "[err] GGUF 打开失败: %s\n", path); return -1; }
    LlamaConfig *c = &m->c;
    get_u32(&g, "llama.block_count", 4, &c->n_layer);
    get_u32(&g, "llama.embedding_length", 512, &c->n_embd);
    get_u32(&g, "llama.attention.head_count", 8, &c->n_head);
    get_u32(&g, "llama.attention.head_count_kv", c->n_head, &c->n_head_kv);
    get_f32(&g, "llama.attention.layer_norm_rms_epsilon", 1e-5f, &c->rmsnorm_eps);
    get_u32(&g, "llama.feed_forward_length", 1024, &c->ffn_dim);
    get_u32(&g, "llama.expert_count", 0, &c->n_expert);
    get_u32(&g, "llama.expert_used_count", 2, &c->n_expert_used);
    get_u32(&g, "llama.context_length", 256, &c->max_seq);
    get_u32(&g, "llama.rope.dimension_count", 0, &c->head_dim);
    get_f32(&g, "llama.rope.freq_base", 10000.0f, &c->rope_theta);
    get_u32(&g, "llama.vocab_size", 0, &c->vocab);
    get_u32(&g, "tokenizer.ggml.bos_token_id", 1, &c->bos_id);
    get_u32(&g, "tokenizer.ggml.eos_token_id", 2, &c->eos_id);
    get_u32(&g, "tokenizer.ggml.unknown_token_id", 0, &c->unk_id);
    c->is_moe = c->n_expert > 0;
    if (c->head_dim == 0) c->head_dim = c->n_embd / c->n_head;

    /* vocab 从 token_embd 张量推断 */
    if (c->vocab == 0 && gguf_has_tensor(&g, "token_embd.weight")) {
        /* 读 dims: ne0 = n_embd, ne1 = vocab */
        c->vocab = 0;
        for (uint64_t i = 0; i < g.tensor_count; i++)
            if (strcmp(g.tensor_name[i], "token_embd.weight") == 0) { c->vocab = (int)g.dims[i][1]; break; }
    }

    /* 分配权重 */
    int L = c->n_layer, V = c->vocab, d = c->n_embd, E = c->n_expert, de = c->ffn_dim;
    m->token_embd = malloc((size_t)V * d * 4);
    m->output = malloc((size_t)d * V * 4);
    m->output_norm = malloc((size_t)d * 4);
    m->attn_norm = calloc(L, sizeof(float *));
    m->attn_q = calloc(L, sizeof(float *));
    m->attn_k = calloc(L, sizeof(float *));
    m->attn_v = calloc(L, sizeof(float *));
    m->attn_o = calloc(L, sizeof(float *));
    m->ffn_norm = calloc(L, sizeof(float *));
    m->router = calloc(L, sizeof(float *));
    m->ffn_gate = calloc(L, sizeof(float **));
    m->ffn_up = calloc(L, sizeof(float **));
    m->ffn_down = calloc(L, sizeof(float **));
    m->df_gate = calloc(L, sizeof(float *));
    m->df_up = calloc(L, sizeof(float *));
    m->df_down = calloc(L, sizeof(float *));
    m->k_cache = calloc(L, sizeof(float *));
    m->v_cache = calloc(L, sizeof(float *));
    size_t kv_per = (size_t)c->max_seq * c->n_head_kv * c->head_dim;

    for (int i = 0; i < L; i++) {
        m->attn_norm[i] = malloc((size_t)d * 4);
        m->attn_q[i] = malloc((size_t)d * c->n_head * c->head_dim * 4);
        m->attn_k[i] = malloc((size_t)d * c->n_head_kv * c->head_dim * 4);
        m->attn_v[i] = malloc((size_t)d * c->n_head_kv * c->head_dim * 4);
        m->attn_o[i] = malloc((size_t)d * d * 4);
        m->ffn_norm[i] = malloc((size_t)d * 4);
        m->k_cache[i] = malloc(kv_per * 4);
        m->v_cache[i] = malloc(kv_per * 4);
        if (c->is_moe) {
            m->router[i] = malloc((size_t)d * E * 4);
            m->ffn_gate[i] = calloc((size_t)E, sizeof(float *));
            m->ffn_up[i] = calloc((size_t)E, sizeof(float *));
            m->ffn_down[i] = calloc((size_t)E, sizeof(float *));
            for (int e = 0; e < E; e++) {
                m->ffn_gate[i][e] = malloc((size_t)d * de * 4);
                m->ffn_up[i][e] = malloc((size_t)d * de * 4);
                m->ffn_down[i][e] = malloc((size_t)de * d * 4);
            }
        } else {
            m->df_gate[i] = malloc((size_t)d * de * 4);
            m->df_up[i] = malloc((size_t)d * de * 4);
            m->df_down[i] = malloc((size_t)de * d * 4);
        }
    }

    /* 位置编码缓存 cos/sin[max_seq][head_dim/2] */
    int half = c->head_dim / 2;
    m->cos_cache = malloc((size_t)c->max_seq * sizeof(float *));
    m->sin_cache = malloc((size_t)c->max_seq * sizeof(float *));
    for (int p = 0; p < c->max_seq; p++) {
        m->cos_cache[p] = malloc((size_t)half * 4);
        m->sin_cache[p] = malloc((size_t)half * 4);
        for (int i = 0; i < half; i++) {
            float theta = powf(c->rope_theta, -2.0f * i / (float)c->head_dim);
            float ang = (float)p * theta;
            m->cos_cache[p][i] = cosf(ang);
            m->sin_cache[p][i] = sinf(ang);
        }
    }

    /* 逐张量装载 */
    for (uint64_t i = 0; i < g.tensor_count; i++) {
        uint64_t n = 1;
        for (uint32_t dd = 0; dd < g.n_dims[i]; dd++) n *= g.dims[i][dd];
        float *tmp = malloc((size_t)n * 4);
        if (gguf_tensor_to_f32(&g, g.tensor_name[i], tmp) > 0)
            load_w(m, g.tensor_name[i], tmp, n);
        free(tmp);
    }

    /* 分词器（按 tokenizer.ggml.model 选 SPM(llama) 或 gpt2 字节级） */
    {
        char *model_type = NULL; char tmp[64];
        if (gguf_meta_string(&g, "tokenizer.ggml.model", tmp, sizeof(tmp))) model_type = tmp;
        char **toks = NULL; int ntok = 0;
        if (gguf_meta_string_array(&g, "tokenizer.ggml.tokens", &toks, &ntok) && ntok > 0) {
            int spm = model_type && (strstr(model_type, "llama") || strstr(model_type, "spm"));
            if (spm) {
                float *scores = calloc((size_t)ntok, sizeof(float));
                gguf_meta_f32_array(&g, "tokenizer.ggml.scores", scores, (uint64_t)ntok);
                llmtok_spm_init(&m->spm_tok, toks, ntok, scores);
                free(scores);
                m->is_spm = 1;
            } else {
                char **mrg = NULL; int nmrg = 0;
                gguf_meta_string_array(&g, "tokenizer.ggml.merges", &mrg, &nmrg);
                llmtok_init(&m->tok, toks, ntok, mrg, nmrg);
                gguf_free_string_array(mrg, nmrg);
                m->is_spm = 0;
            }
            gguf_free_string_array(toks, ntok);
        }
    }

    gguf_close(&g);
    m->is_loaded = 1;
    return 0;
}

int llama_tokenize(const LlamaModel *m, const char *text, int *ids, int max) {
    if (!m->is_loaded) return -1;
    if (m->is_spm) return (m->spm_tok.vocab_size > 0) ? llmtok_spm_encode(&m->spm_tok, text, ids, max) : -1;
    return (m->tok.vocab_size > 0) ? llmtok_encode(&m->tok, text, ids, max) : -1;
}

int llama_detokenize(const LlamaModel *m, const int *ids, int n, char *out, int max) {
    if (!m->is_loaded) return -1;
    if (m->is_spm) return (m->spm_tok.vocab_size > 0) ? llmtok_spm_decode(&m->spm_tok, ids, n, out, max) : -1;
    return (m->tok.vocab_size > 0) ? llmtok_decode(&m->tok, ids, n, out, max) : -1;
}

void llama_free(LlamaModel *m) {
    if (!m) return;
    int L = m->c.n_layer, E = m->c.n_expert;
    llmtok_free(&m->tok);
    llmtok_spm_free(&m->spm_tok);
    free(m->token_embd); free(m->output); free(m->output_norm);
    for (int i = 0; i < L; i++) {
        free(m->attn_norm[i]); free(m->attn_q[i]); free(m->attn_k[i]);
        free(m->attn_v[i]); free(m->attn_o[i]); free(m->ffn_norm[i]);
        free(m->k_cache[i]); free(m->v_cache[i]);
        free(m->router[i]);
        if (m->ffn_gate[i]) {
            for (int e = 0; e < E; e++) { free(m->ffn_gate[i][e]); free(m->ffn_up[i][e]); free(m->ffn_down[i][e]); }
        }
        free(m->ffn_gate[i]); free(m->ffn_up[i]); free(m->ffn_down[i]);
        free(m->df_gate[i]); free(m->df_up[i]); free(m->df_down[i]);
    }
    free(m->attn_norm); free(m->attn_q); free(m->attn_k); free(m->attn_v);
    free(m->attn_o); free(m->ffn_norm); free(m->router);
    free(m->ffn_gate); free(m->ffn_up); free(m->ffn_down);
    free(m->df_gate); free(m->df_up); free(m->df_down);
    free(m->k_cache); free(m->v_cache);
    for (int p = 0; p < m->c.max_seq; p++) { free(m->cos_cache[p]); free(m->sin_cache[p]); }
    free(m->cos_cache); free(m->sin_cache);
    memset(m, 0, sizeof(*m));
}

int64_t llama_param_count(const LlamaModel *m) {
    const LlamaConfig *c = &m->c;
    int64_t n = 0;
    n += (int64_t)c->vocab * c->n_embd;                     /* token_embd */
    n += (int64_t)c->n_embd * c->vocab;                     /* output */
    n += c->n_embd;                                          /* output_norm */
    for (int i = 0; i < c->n_layer; i++) {
        n += c->n_embd;                                      /* attn_norm */
        n += (int64_t)c->n_embd * (c->n_head * c->head_dim); /* q */
        n += (int64_t)c->n_embd * (c->n_head_kv * c->head_dim); /* k,v */
        n += (int64_t)c->n_embd * (c->n_head_kv * c->head_dim);
        n += (int64_t)c->n_embd * c->n_embd;                 /* o */
        n += c->n_embd;                                      /* ffn_norm */
        if (c->is_moe) {
            n += (int64_t)c->n_embd * c->n_expert;           /* router */
            n += 3LL * c->n_expert * c->n_embd * c->ffn_dim;
        } else {
            n += 3LL * c->n_embd * c->ffn_dim;
        }
    }
    return n;
}
