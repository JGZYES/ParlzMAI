#include "core/matrix.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 简单可复现的 xorshift RNG（避免 srand 的跨平台差异） */
static uint32_t g_rng = 2463534242u;

static uint32_t next_rng(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

static float uniform01(void) {
    return (float)(next_rng() & 0xFFFFFFu) / (float)(1u << 24);
}

MoMat *mat_alloc(int rows, int cols) {
    if (rows < 0 || cols < 0) return NULL;
    MoMat *m = calloc(1, sizeof(MoMat));
    if (!m) return NULL;
    m->data = calloc((size_t)rows * (size_t)cols, sizeof(float));
    if (!m->data) {
        free(m);
        return NULL;
    }
    m->rows = rows;
    m->cols = cols;
    m->stride = cols;
    m->owns = 1;
    return m;
}

void mat_free(MoMat *m) {
    if (!m) return;
    if (m->owns && m->data) free(m->data);
    free(m);
}

MoMat *mat_view(MoMat *m, int rows, int cols, int64_t stride, float *data) {
    if (!m) return NULL;
    m->rows = rows;
    m->cols = cols;
    m->stride = stride;
    m->data = data;
    m->owns = 0;
    return m;
}

void mat_fill(MoMat *m, float value) {
    for (int i = 0; i < m->rows; i++) {
        float *row = m->data + (int64_t)i * m->stride;
        for (int j = 0; j < m->cols; j++) row[j] = value;
    }
}

void mat_randn(MoMat *m, unsigned seed, float stddev) {
    if (seed) g_rng = seed;
    for (int i = 0; i < m->rows; i++) {
        float *row = m->data + (int64_t)i * m->stride;
        for (int j = 0; j < m->cols; j++) {
            /* Box-Muller：两个均匀 -> 两个高斯，用其中一个 */
            float u1 = uniform01();
            if (u1 <= 0.0f) u1 = 1e-9f;
            float u2 = uniform01();
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
            row[j] = z * stddev;
        }
    }
}

int mat_mul(const MoMat *a, const MoMat *b, MoMat *out) {
    if (!a || !b || !out) return -1;
    int M = a->rows;
    int K = a->cols;
    int N = b->cols;
    if (b->rows != K) return -1;
    if (out->rows != M || out->cols != N) return -1;

    for (int i = 0; i < M; i++) {
        const float *pa = a->data + (int64_t)i * a->stride;
        float *po = out->data + (int64_t)i * out->stride;
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) {
                s += pa[k] * b->data[(int64_t)k * b->stride + j];
            }
            po[j] = s;
        }
    }
    return 0;
}

void mat_add(MoMat *a, const MoMat *b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return;
    for (int i = 0; i < a->rows; i++) {
        float *pa = a->data + (int64_t)i * a->stride;
        const float *pb = b->data + (int64_t)i * b->stride;
        for (int j = 0; j < a->cols; j++) pa[j] += pb[j];
    }
}

void mat_scale(MoMat *m, float factor) {
    for (int i = 0; i < m->rows; i++) {
        float *row = m->data + (int64_t)i * m->stride;
        for (int j = 0; j < m->cols; j++) row[j] *= factor;
    }
}

MoMat *mat_transpose(const MoMat *a, MoMat *dst) {
    if (!a || !dst) return NULL;
    if (dst->rows != a->cols || dst->cols != a->rows) return NULL;
    for (int i = 0; i < a->rows; i++) {
        const float *pa = a->data + (int64_t)i * a->stride;
        for (int j = 0; j < a->cols; j++) {
            dst->data[(int64_t)j * dst->stride + i] = pa[j];
        }
    }
    return dst;
}

void mat_softmax_rows(MoMat *m) {
    for (int i = 0; i < m->rows; i++) {
        float *row = m->data + (int64_t)i * m->stride;
        /* 数值稳定：减去行内最大值 */
        float mx = row[0];
        for (int j = 1; j < m->cols; j++) if (row[j] > mx) mx = row[j];

        float sum = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            row[j] = expf(row[j] - mx);
            sum += row[j];
        }
        float inv = 1.0f / sum;
        for (int j = 0; j < m->cols; j++) row[j] *= inv;
    }
}

void mat_relu(MoMat *m) {
    for (int i = 0; i < m->rows; i++) {
        float *row = m->data + (int64_t)i * m->stride;
        for (int j = 0; j < m->cols; j++) if (row[j] < 0.0f) row[j] = 0.0f;
    }
}

void mat_sigmoid(MoMat *m) {
    for (int i = 0; i < m->rows; i++) {
        float *row = m->data + (int64_t)i * m->stride;
        for (int j = 0; j < m->cols; j++) {
            float x = row[j];
            row[j] = x >= 0.0f ? (1.0f / (1.0f + expf(-x)))
                               : (expf(x) / (1.0f + expf(x)));
        }
    }
}
