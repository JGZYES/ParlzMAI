#include "infer/generate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "infer/sample.h"
#include "model/tokenizer.h"
#include "utils/logger.h"

/* ---- 流式生成 ---- */

int gen_stream_begin(GenStream *gs, const Model *m, const char *prompt,
                     int max_new, float temperature, int top_k) {
    memset(gs, 0, sizeof(*gs));
    gs->m = m;
    gs->max_new = max_new;
    gs->temperature = temperature;
    gs->top_k = top_k;

    const int max_seq = m->config.max_seq_len;
    gs->ids = malloc((size_t)(max_seq + 16) * sizeof(int));
    if (!gs->ids) return -1;
    gs->ptoks = tokenizer_encode(&m->tok, prompt, gs->ids, max_seq + 16);
    if (gs->ptoks < 0) { free(gs->ids); gs->ids = NULL; return -1; }
    if (gs->ptoks > max_seq) gs->ptoks = max_seq;

    gs->logits = malloc((size_t)m->config.vocab_size * sizeof(float));
    gs->x = malloc((size_t)m->config.d_model * sizeof(float));
    if (!gs->logits || !gs->x) return -1;
    kv_init(&gs->kv, m);

    /* 预填充 prompt，建立 KV cache */
    for (int i = 0; i < gs->ptoks; i++) {
        moe_embed(m, gs->ids[i], i, gs->x);
        moe_forward(m, &gs->kv, i, gs->x, gs->logits);
    }
    gs->pos = gs->ptoks;
    return 0;
}

int gen_stream_next(GenStream *gs, char *out_buf, size_t out_sz) {
    const Model *m = gs->m;
    if (gs->done) return 0;
    if (gs->emitted >= gs->max_new || gs->pos >= m->config.max_seq_len) {
        gs->done = 1;
        return 0;
    }
    int next = sample_top_k(gs->logits, m->config.vocab_size, gs->temperature, gs->top_k);
    tokenizer_decode(&m->tok, &next, 1, out_buf, (int)out_sz);

    moe_embed(m, next, gs->pos, gs->x);
    moe_forward(m, &gs->kv, gs->pos, gs->x, gs->logits);
    gs->pos++;
    gs->emitted++;
    return 1;
}

void gen_stream_end(GenStream *gs) {
    if (!gs) return;
    kv_free(&gs->kv);
    free(gs->logits);
    free(gs->x);
    free(gs->ids);
    memset(gs, 0, sizeof(*gs));
}

/* ---- 一次性生成（CLI） ---- */

int moe_generate(const Model *m, const char *prompt, int max_new_tokens,
                 float temperature, int top_k) {
    GenStream gs;
    if (gen_stream_begin(&gs, m, prompt, max_new_tokens, temperature, top_k) < 0) {
        MO_LOGE("prompt 编码失败");
        return -1;
    }
    printf("%s", prompt);
    fflush(stdout);
    char buf[64];
    while (gen_stream_next(&gs, buf, sizeof(buf)) == 1) {
        printf("%s", buf);
        fflush(stdout);
    }
    printf("\n");
    gen_stream_end(&gs);
    return 0;
}

char *moe_generate_text(const Model *m, const char *prompt, int max_new_tokens,
                        float temperature, int top_k) {
    GenStream gs;
    size_t cap = strlen(prompt) + (size_t)max_new_tokens * 16 + 128;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    size_t w = (size_t)snprintf(buf, cap, "%s", prompt);

    if (gen_stream_begin(&gs, m, prompt, max_new_tokens, temperature, top_k) < 0) {
        MO_LOGE("prompt 编码失败");
        gen_stream_end(&gs);
        return buf;
    }
    char tok[64];
    while (gen_stream_next(&gs, tok, sizeof(tok)) == 1) {
        size_t tl = strlen(tok);
        if (w + tl + 1 > cap) {
            cap = cap * 2 + tl + 64;
            buf = realloc(buf, cap);
        }
        memcpy(buf + w, tok, tl);
        w += tl;
        buf[w] = '\0';
    }
    gen_stream_end(&gs);
    return buf;
}
