#ifndef MO_MODEL_H
#define MO_MODEL_H

#include <stdint.h>

#include "model/config.h"
#include "model/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 已加载的模型：config + tokenizer + 全部权重（float32，行主序，列为 (in, out)）。 */
typedef struct {
    ModelConfig config;
    Tokenizer   tok;

    /* 嵌入 */
    float *wte;           /* vocab * d_model */
    float *wpe;           /* max_seq * d_model */

    /* 每层 */
    float **l_norm1;      /* [layer] d_model */
    float **wq, **wk, **wv, **wo;      /* [layer] d_model*d_model */
    float **l_norm2;      /* [layer] d_model */
    float **router;       /* [layer] d_model*n_experts；稠密层为 NULL */
    float ***exp_w1;      /* [layer][expert] d_model*d_expert */
    float ***exp_w2;      /* [layer][expert] d_expert*d_model */
    float **ffn_w1;       /* [layer] d_model*d_ff；MoE 层为 NULL */
    float **ffn_w2;       /* [layer] d_ff*d_model */

    /* 最终 */
    float *norm_final;    /* d_model */
    float *lm_head;       /* vocab * d_model */

    int is_loaded;
} Model;

/* 从 .pap 文件加载模型。成功返回 0 */
int model_load(Model *m, const char *path);
/* 从 GGUF 文件加载模型（使用我们定义的元数据键）。成功返回 0 */
int model_load_gguf(Model *m, const char *path);
void model_free(Model *m);
int64_t model_param_count(const Model *m);

#ifdef __cplusplus
}
#endif

#endif /* MO_MODEL_H */
