#ifndef MO_SAMPLE_H
#define MO_SAMPLE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void mo_rng_seed(unsigned seed);

/* 依据 logits 采样一个 token id。
   temperature 缩放 logits；top_k 限制候选；用可复现的 xorshift。 */
int sample_top_k(const float *logits, int vocab, float temperature, int top_k);

#ifdef __cplusplus
}
#endif

#endif /* MO_SAMPLE_H */
