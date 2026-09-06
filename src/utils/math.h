#ifndef MO_MATH_H
#define MO_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 神经网络常用激活函数（标准库实现，精度优先） ---- */

/* tanh */
float mo_tanhf(float x);

/* sigmoid = 1 / (1 + exp(-x)) */
float mo_sigmoidf(float x);

/* GELU（GPT-2 的 tanh 近似） */
float mo_gelu(float x);

/* ---- 快速近似（速度优先，精度有损） ---- */

/* 快速指数近似（Schraudolph 位技巧），用于需要高吞吐的路径 */
float mo_fast_exp(float x);

/* 快速 sigmoid，基于 mo_fast_exp */
float mo_fast_sigmoid(float x);

#ifdef __cplusplus
}
#endif

#endif /* MO_MATH_H */
