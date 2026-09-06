#ifndef MO_MATRIX_H
#define MO_MATRIX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 浮点矩阵（行主序）。支持 stride，方便对张量切片做视图而不复制。
 * Phase 1 提供基础算子（串行）；后续用 OpenMP/BLAS 加速。
 */
typedef struct {
    int      rows;
    int      cols;
    int64_t  stride; /* 相邻两行之间的浮点偏移 */
    float   *data;
    uint8_t  owns;   /* 是否负责释放 data */
} MoMat;

/* 分配 rows x cols 全 0 矩阵。失败返回 NULL */
MoMat *mat_alloc(int rows, int cols);

/* 释放矩阵（data 仅当 owns 为真时释放） */
void mat_free(MoMat *m);

/* 用外部缓冲构造一个视图（不复制、不释放） */
MoMat *mat_view(MoMat *m, int rows, int cols, int64_t stride, float *data);

/* 填充常量 */
void mat_fill(MoMat *m, float value);

/* 高斯随机初始化（Box-Muller；seed=0 恢复默认序列） */
void mat_randn(MoMat *m, unsigned seed, float stddev);

/* out = a * b。维度须满足 (M,K)x(K,N)->(M,N)。成功返回 0 */
int mat_mul(const MoMat *a, const MoMat *b, MoMat *out);

/* a += b（同尺寸） */
void mat_add(MoMat *a, const MoMat *b);

/* m *= factor */
void mat_scale(MoMat *m, float factor);

/* dst = transpose(a)。dst 尺寸应为 (cols, rows)；返回 dst */
MoMat *mat_transpose(const MoMat *a, MoMat *dst);

/* 逐行 softmax（原地） */
void mat_softmax_rows(MoMat *m);

/* ReLU 原地 */
void mat_relu(MoMat *m);

/* Sigmoid 原地 */
void mat_sigmoid(MoMat *m);

#ifdef __cplusplus
}
#endif

#endif /* MO_MATRIX_H */
