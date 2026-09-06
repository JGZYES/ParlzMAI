#include "utils/mempool.h"

#include <stdlib.h>

#define MO_DEFAULT_BLOCK_SIZE (64u * 1024u)
#define MO_ALIGN 64u

static size_t align_up(size_t n, size_t a) {
    return (n + a - 1u) & ~(a - 1u);
}

MoMemPool *mempool_create(size_t block_size) {
    MoMemPool *p = calloc(1, sizeof(MoMemPool));
    if (!p) return NULL;
    p->block_size = block_size ? block_size : MO_DEFAULT_BLOCK_SIZE;
    return p;
}

static MoMemPoolBlock *new_block(size_t min_cap, size_t block_size) {
    /* 块容量向上取整到对齐边界；基址用 posix_memalign 保证强对齐，
       这样返回给调用方的指针也能保持 MO_ALIGN 对齐（为 SIMD/AVX 准备）。 */
    size_t cap = align_up(min_cap > block_size ? min_cap : block_size, MO_ALIGN);
    MoMemPoolBlock *b = calloc(1, sizeof(MoMemPoolBlock));
    if (!b) return NULL;
    int rc = posix_memalign((void **)&b->base, MO_ALIGN, cap);
    if (rc != 0) {
        free(b);
        return NULL;
    }
    b->capacity = cap;
    b->used = 0;
    return b;
}

void *mempool_alloc(MoMemPool *pool, size_t size) {
    if (!pool || size == 0) return NULL;
    size = align_up(size, MO_ALIGN); /* 至少 64 对齐 */

    /* 优先复用已有块 */
    for (MoMemPoolBlock *b = pool->blocks; b; b = b->next) {
        if (b->capacity - b->used >= size) {
            void *p = b->base + b->used;
            b->used += size;
            return p;
        }
    }

    /* 没有合适块，新建一个并头插 */
    MoMemPoolBlock *b = new_block(size, pool->block_size);
    if (!b) return NULL;
    b->next = pool->blocks;
    pool->blocks = b;

    void *p = b->base + b->used;
    b->used += size;
    return p;
}

void mempool_clear(MoMemPool *pool) {
    if (!pool) return;
    for (MoMemPoolBlock *b = pool->blocks; b; b = b->next) {
        b->used = 0;
    }
}

void mempool_destroy(MoMemPool *pool) {
    if (!pool) return;
    MoMemPoolBlock *b = pool->blocks;
    while (b) {
        MoMemPoolBlock *next = b->next;
        free(b->base);
        free(b);
        b = next;
    }
    free(pool);
}

size_t mempool_used_bytes(const MoMemPool *pool) {
    size_t total = 0;
    for (const MoMemPoolBlock *b = pool->blocks; b; b = b->next) {
        total += b->used;
    }
    return total;
}
