#include "utils/math.h"

#include <math.h>
#include <stdint.h>

float mo_tanhf(float x) { return tanhf(x); }

float mo_sigmoidf(float x) {
    /* 数值稳定：避免 expf 溢出 */
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

float mo_gelu(float x) {
    /* GELU 的 tanh 近似：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))) */
    const float c = 0.7978845608028654f;  /* sqrt(2/pi) */
    const float a = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c * (x + a * x * x * x)));
}

float mo_fast_exp(float x) {
    /* Schraudolph 近似；Range 约为 (-88, 88)，略偏大，适合做 softmax 相对比较 */
    union { float f; int32_t i; } v;
    v.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return v.f;
}

float mo_fast_sigmoid(float x) {
    float e = mo_fast_exp(-x);
    return 1.0f / (1.0f + e);
}
