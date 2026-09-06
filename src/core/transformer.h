#ifndef MO_TRANSFORMER_H
#define MO_TRANSFORMER_H

#include <stdint.h>

#include "model/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* KV 缓存：每层一份，容量 = max_seq * n_head * head_dim */
typedef struct {
    int seq_len;      /* 已处理位置数 */
    int max_seq;
    int n_layer;
    float **k_cache;  /* [layer] */
    float **v_cache;  /* [layer] */
} KVState;

void kv_init(KVState *kv, const Model *m);
void kv_free(KVState *kv);
void kv_reset(KVState *kv);

/* 计算单个位置 token 的嵌入 x = wte[token] + wpe[pos]，写入 x(d_model) */
void moe_embed(const Model *m, int token, int pos, float *x);

/* 对位置 pos 执行一次前向，返回该位置的 logits(vocab)。
   会写入 KV cache（pos 处）。调用方负责已就绪的嵌入与 cache 状态。 */
void moe_forward(const Model *m, KVState *kv, int pos, const float *x, float *logits);

#ifdef __cplusplus
}
#endif

#endif /* MO_TRANSFORMER_H */
