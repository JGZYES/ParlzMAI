#ifndef MO_GENERATE_H
#define MO_GENERATE_H

#include <stddef.h>

#include "core/transformer.h"
#include "model/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- 一次性生成（CLI 用，自回归打印到 stdout） ---------- */
int moe_generate(const Model *m, const char *prompt, int max_new_tokens,
                 float temperature, int top_k);

/* 生成并返回完整文本（prompt + 生成）。调用方 free()。失败返回 strdup(prompt)。 */
char *moe_generate_text(const Model *m, const char *prompt, int max_new_tokens,
                        float temperature, int top_k);

/* ---------- 流式生成（逐个 token 拉取，供 HTTP/SSE 用） ---------- */
typedef struct {
    const Model *m;
    KVState kv;          /* 内部 KV cache */
    float *logits;
    float *x;
    int   *ids;          /* prompt token 缓存 */
    int    ptoks;
    int    pos;
    int    emitted;      /* 已产出 token 数 */
    int    max_new;
    float  temperature;
    int    top_k;
    int    done;
    char   last_buf[64]; /* 最后一个 token 的utf8（避免重复解码） */
} GenStream;

/* 初始化 + 预填充 prompt。成功返回 0 */
int gen_stream_begin(GenStream *gs, const Model *m, const char *prompt,
                     int max_new, float temperature, int top_k);
/* 拉取下一个 token 文本到 out_buf；返回 1=产出tokens，0=结束，负=出错 */
int gen_stream_next(GenStream *gs, char *out_buf, size_t out_sz);
void gen_stream_end(GenStream *gs);

#ifdef __cplusplus
}
#endif

#endif /* MO_GENERATE_H */
