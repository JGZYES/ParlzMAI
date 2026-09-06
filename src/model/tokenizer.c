#include "model/tokenizer.h"

#include <stdlib.h>
#include <string.h>

int tokenizer_build(Tokenizer *t,
                    int vocab_size, const unsigned char **vocab, const int *vocab_len,
                    int num_merges, const uint32_t *merge_a, const uint32_t *merge_b) {
    if (!t || vocab_size <= 0) return -1;
    t->vocab_size = vocab_size;
    t->num_merges = num_merges;

    t->vocab = calloc((size_t)vocab_size, sizeof(unsigned char *));
    t->vocab_len = calloc((size_t)vocab_size, sizeof(int));
    if (!t->vocab || !t->vocab_len) return -1;
    for (int i = 0; i < vocab_size; i++) {
        t->vocab[i] = malloc((size_t)(vocab_len[i] ? vocab_len[i] : 1));
        if (!t->vocab[i]) return -1;
        memcpy(t->vocab[i], vocab[i], (size_t)vocab_len[i]);
        t->vocab_len[i] = vocab_len[i];
    }

    if (num_merges > 0) {
        t->merge_a = malloc((size_t)num_merges * sizeof(uint32_t));
        t->merge_b = malloc((size_t)num_merges * sizeof(uint32_t));
        if (!t->merge_a || !t->merge_b) return -1;
        memcpy(t->merge_a, merge_a, (size_t)num_merges * sizeof(uint32_t));
        memcpy(t->merge_b, merge_b, (size_t)num_merges * sizeof(uint32_t));
    } else {
        t->merge_a = NULL;
        t->merge_b = NULL;
    }
    return 0;
}

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->vocab_len);
    free(t->merge_a);
    free(t->merge_b);
    memset(t, 0, sizeof(*t));
}

/* 找到 id 下标 [st, ed] 的字节拼接长度，用于 rank 遍历时比对。这里直接比较 merge 的 (a,b) 与当前 token 的 (id[i],id[i+1])。 */
static int merge_rank(const Tokenizer *t, uint32_t a, uint32_t b) {
    for (int i = 0; i < t->num_merges; i++) {
        if (t->merge_a[i] == a && t->merge_b[i] == b) return i;
    }
    return -1;
}

int tokenizer_encode(const Tokenizer *t, const char *text, int32_t *out_ids, int max_out) {
    if (!t || !text || !out_ids) return -1;
    size_t n = strlen(text);
    if (n == 0) return 0;

    /* 每个字节一个初始 token，id=字节值(0..255) */
    int *tok = malloc((size_t)(n + 1) * sizeof(int));
    if (!tok) return -1;
    int len = 0;
    for (size_t i = 0; i < n; i++) {
        tok[len++] = (unsigned char)text[i];
    }

    const int base = 256;
    while (1) {
        int best = -1, best_rank = 0x7fffffff;
        for (int i = 0; i + 1 < len; i++) {
            int r = merge_rank(t, (uint32_t)tok[i], (uint32_t)tok[i + 1]);
            if (r >= 0 && r < best_rank) {
                best_rank = r;
                best = i;
            }
        }
        if (best < 0) break;
        tok[best] = base + best_rank;
        memmove(&tok[best + 1], &tok[best + 2], (size_t)(len - best - 2) * sizeof(int));
        len--;
    }

    if (len > max_out) {
        free(tok);
        return -2;
    }
    for (int i = 0; i < len; i++) out_ids[i] = tok[i];
    free(tok);
    return len;
}

int tokenizer_decode(const Tokenizer *t, const int32_t *ids, int n, char *out, int max_out) {
    if (!t || !ids || !out) return -1;
    int w = 0;
    for (int i = 0; i < n && w < max_out; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size || !t->vocab[id]) continue;
        int L = t->vocab_len[id];
        for (int j = 0; j < L && w < max_out; j++) {
            /* 简单替换非法 UTF-8 连续字节为 '?' 的占位处理：原样拷贝，由读取端决定 */
            out[w++] = (char)t->vocab[id][j];
        }
    }
    if (w < max_out) out[w] = '\0';
    else out[max_out - 1] = '\0';
    return w;
}
