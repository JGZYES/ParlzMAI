#ifndef MO_LLAMA_H
#define MO_LLAMA_H

#include <stddef.h>
#include <stdint.h>

#include "model/llama_tokenizer.h"
#include "model/llama_tokenizer_spm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* llama / Mixtral 架构。所有权重以 (in, out) 行主序存储，C 端直接 x @ W。 */
typedef struct {
    int vocab, n_layer, n_embd, n_head, n_head_kv, head_dim, ffn_dim;
    int n_expert, n_expert_used, max_seq;
    float rmsnorm_eps, rope_theta;
    int is_moe;   /* 1=Mixrtal MoE, 0=dense SwiGLU */
    int bos_id, eos_id, unk_id;   /* 特殊 token id */
} LlamaConfig;

typedef struct {
    LlamaConfig c;

    float *token_embd;    /* vocab * n_embd */
    float *output_norm;   /* n_embd */
    float *output;        /* n_embd * vocab */

    float **attn_norm, **attn_q, **attn_k, **attn_v, **attn_o, **ffn_norm; /* [L] */
    float **router;                 /* [L] n_embd * n_expert */
    float ***ffn_gate, ***ffn_up, ***ffn_down;  /* [L][E] */
    float **df_gate, **df_up, **df_down;        /* [L] 稠密路径 */

    float **cos_cache, **sin_cache;  /* [max_seq][head_dim/2] */

    /* KV cache（内部使用） */
    float **k_cache, **v_cache;      /* [L] max_seq * n_head_kv * head_dim */

    LlamaTokenizer tok;             /* gpt2 分词器 */
    LlamaTokenizerSPM spm_tok;      /* llama/spm 分词器 */
    int  is_spm;                    /* 1=SentencePiece(llama), 0=gpt2 */

    int is_loaded;
} LlamaModel;

int llama_load(LlamaModel *m, const char *path);
void llama_free(LlamaModel *m);
int64_t llama_param_count(const LlamaModel *m);

/* 文本 <-> token id（若模型带分词器）。返回 token 数/字节数，负为错 */
int llama_tokenize(const LlamaModel *m, const char *text, int *ids, int max);
int llama_detokenize(const LlamaModel *m, const int *ids, int n, char *out, int max);

/* 输入嵌入：x = token_embd[token]（RoPE 在 forward 内处理位置） */
void llama_embed(const LlamaModel *m, int token, float *x);

/* 对位置 pos 前向（会写入内部 KV cache），输出 logits(vocab) */
void llama_forward(LlamaModel *m, int pos, const float *x, float *logits);

/* 自回归生成：从 input_ids 开始，产出 token id 写入 out_ids（长度 max_new），返回产出数。 */
int llama_generate(LlamaModel *m, const int *input_ids, int n_input,
                   int max_new, float temperature, int top_k, int *out_ids);

#ifdef __cplusplus
}
#endif

#endif /* MO_LLAMA_H */
