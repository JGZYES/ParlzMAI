#ifndef MO_MEMPOOL_H
#define MO_MEMPOOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 内存池（Arena）。
 *
 * 目的：把大量短生命周期的小分配聚合成少数大块分配，
 * 减少 malloc/free 调用次数与内存碎片化。一次性全部释放（mempool_clear/destroy）。
 *
 * 注意：不是通用 malloc 替代品 —— 不支持单个对象 free，仅供一次性、同生命周期数据使用。
 */

typedef struct MoMemPoolBlock {
    uint8_t *base;     /* 块起始 */
    size_t   capacity; /* 块容量（字节） */
    size_t   used;     /* 已用（字节，对齐后） */
    struct MoMemPoolBlock *next;
} MoMemPoolBlock;

typedef struct {
    size_t         block_size; /* 每个新块的默认容量 */
    MoMemPoolBlock *blocks;
} MoMemPool;

/* 创建内存池；block_size=0 时用默认 64KB。返回池指针，失败返回 NULL */
MoMemPool *mempool_create(size_t block_size);

/* 分配 size 字节（至少 64 字节对齐）。失败返回 NULL */
void *mempool_alloc(MoMemPool *pool, size_t size);

/* 清空所有已用空间（复用底层块，不释放） */
void mempool_clear(MoMemPool *pool);

/* 释放整个池及其所有块 */
void mempool_destroy(MoMemPool *pool);

/* 当前已用字节数 */
size_t mempool_used_bytes(const MoMemPool *pool);

#ifdef __cplusplus
}
#endif

#endif /* MO_MEMPOOL_H */
