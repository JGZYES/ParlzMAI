#include "infer/sample.h"

#include <math.h>
#include <stdlib.h>

static uint32_t g_rng = 2463534242u;

void mo_rng_seed(unsigned seed) { g_rng = seed ? seed : 2463534242u; }

static uint32_t next_rng(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

typedef struct { float v; int idx; } Scored;

static int cmp_desc(const void *a, const void *b) {
    const Scored *pa = (const Scored *)a, *pb = (const Scored *)b;
    if (pa->v > pb->v) return -1;
    if (pa->v < pb->v) return 1;
    return pa->idx - pb->idx;
}

int sample_top_k(const float *logits, int vocab, float temperature, int top_k) {
    if (temperature <= 0.0f) {
        /* 贪心 */
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    if (top_k <= 0 || top_k > vocab) top_k = vocab;

    Scored *sc = malloc((size_t)vocab * sizeof(Scored));
    for (int i = 0; i < vocab; i++) {
        sc[i].v = logits[i] / temperature;
        sc[i].idx = i;
    }
    qsort(sc, (size_t)vocab, sizeof(Scored), cmp_desc);

    /* softmax over 前 top_k 个 */
    float mxs = sc[0].v;
    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        sc[i].v = expf(sc[i].v - mxs);
        sum += sc[i].v;
    }
    float target = ((float)(next_rng() & 0xFFFFFFu) / (float)(1u << 24)) * sum;
    float acc = 0.0f;
    for (int i = 0; i < top_k; i++) {
        acc += sc[i].v;
        if (target <= acc) {
            int idx = sc[i].idx;
            free(sc);
            return idx;
        }
    }
    int idx = sc[top_k - 1].idx;
    free(sc);
    return idx;
}
