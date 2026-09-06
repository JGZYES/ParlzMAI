#ifndef MO_TOKENIZER_H
#define MO_TOKENIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * byte-level BPE tokenizer（与 platform/tokenizer.py 完全一致）：
 *   id 0..255      = 单个字节 bytes([b])
 *   id 256+i       = 第 i 条 merge 合并出的 token
 *   merges[i] = (a,b) -> 新 id = 256 + i
 * 无特殊 token（n_special=0），直接对字节做 BPE。
 */
typedef struct {
    int vocab_size;
    int num_merges;
    unsigned char **vocab; /* vocab_size 个字符串 */
    int *vocab_len;
    uint32_t *merge_a;     /* num_merges */
    uint32_t *merge_b;
} Tokenizer;

/* 用外部数据深拷贝构建（vocab/vocab_len/merge_a/merge_b 会被复制走） */
int tokenizer_build(Tokenizer *t,
                    int vocab_size, const unsigned char **vocab, const int *vocab_len,
                    int num_merges, const uint32_t *merge_a, const uint32_t *merge_b);

void tokenizer_free(Tokenizer *t);

/* 编码 text -> ids（最多 max_out 个）。返回 token 数，负数为错 */
int tokenizer_encode(const Tokenizer *t, const char *text, int32_t *out_ids, int max_out);

/* 解码 ids -> utf-8 文本（最多 max_out 字节）。返回写入字节数 */
int tokenizer_decode(const Tokenizer *t, const int32_t *ids, int n, char *out, int max_out);

#ifdef __cplusplus
}
#endif

#endif /* MO_TOKENIZER_H */
