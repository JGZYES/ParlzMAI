#ifndef MO_TENSOR_H
#define MO_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 多维张量。行主序（row-major）连续存储，浮点单精度。
 * 参考 llama2.c / Qwen 的分层思路，Phase 1 先做基础结构与访问。
 */
typedef struct {
    int      ndim;    /* 维度数量 */
    int64_t *shape;   /* 每个维度大小，长度 ndim */
    int64_t *strides; /* 每个维度的元素步长（行主序），长度 ndim */
    int64_t  numel;   /* 元素总数 = prod(shape) */
    float   *data;    /* 连续元素缓冲（行主序） */
    uint8_t  owns;    /* data 是否为本对象所有（由本对象负责释放） */
} MoTensor;

/* 创建全 0 张量（赋值阻塞维度语义）。失败返回 NULL */
MoTensor *tensor_create(int ndim, const int64_t *shape);
MoTensor *tensor_create_zeros(int ndim, const int64_t *shape);

/* 释放张量（data 仅当 owns 为真时释放） */
void tensor_free(MoTensor *t);

/* 深拷贝（新的 data 缓冲区） */
MoTensor *tensor_clone(const MoTensor *t);

/* 用标量填充 */
void tensor_fill(MoTensor *t, float value);

/* 元素总数 */
int64_t tensor_numel(const MoTensor *t);
/* 某维大小 */
int64_t tensor_size(const MoTensor *t, int dim);

/* 计算给定下标（长度 ndim）的线性偏移 */
int64_t tensor_offset(const MoTensor *t, const int64_t *indices);

/* 就地 reshape 视图（不复制数据）。元素总数必须一致，否则返回 NULL */
MoTensor *tensor_reshape(MoTensor *t, int ndim, const int64_t *shape);

#ifdef __cplusplus
}
#endif

#endif /* MO_TENSOR_H */
