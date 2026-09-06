#ifndef MO_CONFIG_H
#define MO_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    int n_layer;
    int d_model;
    int n_head;
    int head_dim;     /* d_model / n_head */
    int d_ff;         /* 稠密 FFN 中间层 */
    int moe_n_experts;
    int moe_top_k;
    int d_expert;     /* MoE 专家中间层 */
    int max_seq_len;
    int num_merges;
    int n_special;
    int tie_weights;
    float rmsnorm_eps;
    unsigned char *moe_mask; /* n_layer: 1=MoE, 0=dense */
} ModelConfig;

void config_init(ModelConfig *c);
void config_free(ModelConfig *c);

#ifdef __cplusplus
}
#endif

#endif /* MO_CONFIG_H */
