#include "core/tensor.h"

#include <stdlib.h>
#include <string.h>

static int64_t prod_dims(int ndim, const int64_t *shape) {
    int64_t p = 1;
    for (int i = 0; i < ndim; i++) p *= shape[i];
    return p;
}

static void compute_strides(int ndim, const int64_t *shape, int64_t *strides) {
    int64_t s = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = s;
        s *= shape[i];
    }
}

MoTensor *tensor_create(int ndim, const int64_t *shape) {
    if (ndim <= 0 || !shape) return NULL;

    MoTensor *t = calloc(1, sizeof(MoTensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->shape = malloc((size_t)ndim * sizeof(int64_t));
    t->strides = malloc((size_t)ndim * sizeof(int64_t));
    if (!t->shape || !t->strides) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    memcpy(t->shape, shape, (size_t)ndim * sizeof(int64_t));
    compute_strides(ndim, shape, t->strides);

    t->numel = prod_dims(ndim, shape);
    t->data = calloc((size_t)t->numel, sizeof(float));
    t->owns = 1;
    if (!t->data) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }
    return t;
}

MoTensor *tensor_create_zeros(int ndim, const int64_t *shape) {
    return tensor_create(ndim, shape); /* calloc 已将 data 清零 */
}

void tensor_free(MoTensor *t) {
    if (!t) return;
    if (t->owns && t->data) free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}

MoTensor *tensor_clone(const MoTensor *t) {
    if (!t) return NULL;
    MoTensor *c = tensor_create(t->ndim, t->shape);
    if (!c) return NULL;
    memcpy(c->data, t->data, (size_t)t->numel * sizeof(float));
    return c;
}

void tensor_fill(MoTensor *t, float value) {
    if (!t) return;
    for (int64_t i = 0; i < t->numel; i++) t->data[i] = value;
}

int64_t tensor_numel(const MoTensor *t) { return t ? t->numel : 0; }
int64_t tensor_size(const MoTensor *t, int dim) { return t ? t->shape[dim] : 0; }

int64_t tensor_offset(const MoTensor *t, const int64_t *indices) {
    int64_t off = 0;
    for (int i = 0; i < t->ndim; i++) {
        off += indices[i] * t->strides[i];
    }
    return off;
}

MoTensor *tensor_reshape(MoTensor *t, int ndim, const int64_t *shape) {
    if (!t || ndim <= 0 || !shape) return NULL;
    if (prod_dims(ndim, shape) != t->numel) return NULL; /* 元素总数不一致 */

    memcpy(t->shape, shape, (size_t)ndim * sizeof(int64_t));
    compute_strides(ndim, shape, t->strides);
    t->ndim = ndim;
    return t; /* 就地改写，返回同指针 */
}
