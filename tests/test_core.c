#include "core/matrix.h"
#include "core/tensor.h"
#include "utils/math.h"
#include "utils/mempool.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  ok: %s\n", msg); } \
    else      { g_fail++; printf("  FAIL: %s\n", msg); } \
} while (0)

#define CHECK_NEAR(got, want, eps, msg) do { \
    float _g = (float)(got), _w = (float)(want); \
    if (fabsf(_g - _w) <= (eps)) { g_pass++; printf("  ok: %s (%.6f)\n", msg, _g); } \
    else { g_fail++; printf("  FAIL: %s  got %.6f want %.6f\n", msg, _g, _w); } \
} while (0)

static void test_tensor(void) {
    printf("[tensor]\n");
    int64_t shape[2] = {3, 4};
    MoTensor *t = tensor_create(2, shape);
    CHECK(t != NULL, "create 3x4");
    CHECK(tensor_numel(t) == 12, "numel==12");
    CHECK(t->strides[0] == 4 && t->strides[1] == 1, "row-major strides");

    tensor_fill(t, 5.0f);
    CHECK(t->data[0] == 5.0f && t->data[11] == 5.0f, "filled all");

    int64_t idx[2] = {2, 3};
    CHECK(tensor_offset(t, idx) == 11, "offset (2,3)==11");

    int64_t flat[1] = {12};
    MoTensor *r = tensor_reshape(t, 1, flat);
    CHECK(r != NULL && r->ndim == 1, "reshape to 1D ok");
    int64_t bad[1] = {13};
    CHECK(tensor_reshape(t, 1, bad) == NULL, "reshape wrong size -> NULL");

    MoTensor *clone = tensor_clone(t);
    CHECK(clone != NULL && clone->data[5] == 5.0f, "clone preserves data");
    tensor_free(clone);
    tensor_free(t);
}

static void test_matmul(void) {
    printf("[matmul]\n");
    MoMat *a = mat_alloc(2, 3);
    MoMat *b = mat_alloc(3, 2);
    MoMat *c = mat_alloc(2, 2);
    float ad[6] = {1, 2, 3, 4, 5, 6};
    float bd[6] = {7, 8, 9, 10, 11, 12};
    memcpy(a->data, ad, sizeof(ad));
    memcpy(b->data, bd, sizeof(bd));

    int rc = mat_mul(a, b, c);
    CHECK(rc == 0, "mul success");
    CHECK_NEAR(c->data[0], 58.0f, 1e-5f, "c[0][0]==58");
    CHECK_NEAR(c->data[1], 64.0f, 1e-5f, "c[0][1]==64");
    CHECK_NEAR(c->data[c->stride + 0], 139.0f, 1e-5f, "c[1][0]==139");
    CHECK_NEAR(c->data[c->stride + 1], 154.0f, 1e-5f, "c[1][1]==154");

    MoMat *w = mat_alloc(2, 2); /* wrong N */
    CHECK(mat_mul(a, w, w) != 0, "mismatched dims -> error");
    mat_free(w);
    mat_free(a);
    mat_free(b);
    mat_free(c);
}

static void test_softmax(void) {
    printf("[softmax]\n");
    MoMat *m = mat_alloc(1, 5);
    float d[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    memcpy(m->data, d, sizeof(d));
    mat_softmax_rows(m);
    float sum = 0.0f;
    for (int j = 0; j < 5; j++) sum += m->data[j];
    CHECK_NEAR(sum, 1.0f, 1e-5f, "row sums to 1");
    CHECK(m->data[4] > m->data[0], "monotonic increasing");
    mat_free(m);
}

static void test_math(void) {
    printf("[math]\n");
    CHECK_NEAR(mo_sigmoidf(0.0f), 0.5f, 1e-4f, "sigmoid(0)==0.5");
    CHECK_NEAR(mo_tanhf(0.0f), 0.0f, 1e-5f, "tanh(0)==0");
    CHECK(mo_gelu(0.0f) == 0.0f, "gelu(0)==0");
    CHECK_NEAR(mo_gelu(1.0f), 0.841192f, 1e-3f, "gelu(1) approx");
    CHECK(mo_fast_exp(1.0f) > 2.0f && mo_fast_exp(1.0f) < 3.0f, "fast_exp(1) in (2,3)");
}

static void test_mempool(void) {
    printf("[mempool]\n");
    MoMemPool *pool = mempool_create(0);
    CHECK(pool != NULL, "create");

    void *p1 = mempool_alloc(pool, 100);
    void *p2 = mempool_alloc(pool, 5000);
    CHECK(p1 != NULL && p2 != NULL, "two allocs");
    CHECK(((uintptr_t)p1 & 63u) == 0, "64-byte aligned");
    CHECK(p2 > p1, "distinct allocation");

    /* 数据可写读回 */
    ((uint8_t *)p1)[0] = 0xAB;
    ((uint8_t *)p1)[99] = 0xCD;
    CHECK(((uint8_t *)p1)[0] == 0xAB && ((uint8_t *)p1)[99] == 0xCD, "read/write intact");

    mempool_clear(pool);
    void *p3 = mempool_alloc(pool, 100);
    CHECK(p3 != NULL, "alloc after clear");

    mempool_destroy(pool);
    CHECK(1, "destroy");
}

int main(void) {
    printf("=== moe-serve core tests ===\n");
    test_tensor();
    test_matmul();
    test_softmax();
    test_math();
    test_mempool();
    printf("=== result: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
