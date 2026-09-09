#ifndef MO_LLAMA_TOKENIZER_H
#define MO_LLAMA_TOKENIZER_H

#ifdef __cplusplus
extern "C" {
#endif

/* GPT-2 / llama 的 byte-level BPE 分词器（读 GGUF tokenizer.ggml.tokens/merges）。 */
typedef struct {
    int vocab_size;
    char **tokens;          /* 深拷贝，[vocab_size] */
    int  *byte_to_id;       /* [256] 字节 -> token id */
    int  *sort_idx;         /* [vocab_size] 按 tokens 字符串排序的索引（二分查找用） */
    int   num_merges;
    int  *ma, *mb;          /* [num_merges] 被合并的两个 token id */
} LlamaTokenizer;

/* 用 tokens/merges 字符串构建（深拷贝 tokens/merges）。merges 为 "a b" 形式。成功返回 0。 */
int llmtok_init(LlamaTokenizer *t, char **tokens, int vocab_size, char **merges, int num_merges);
void llmtok_free(LlamaTokenizer *t);

/* 编码：text -> ids（最多 max 个）。返回 token 数，负为错。 */
int llmtok_encode(const LlamaTokenizer *t, const char *text, int *out, int max);
/* 解码：ids -> utf-8（最多 max 字节）。返回字节数。 */
int llmtok_decode(const LlamaTokenizer *t, const int *ids, int n, char *out, int max);
/* 按 token 字符串精确查找 id（chat 模板特殊标记）。未命中返回 -1 */
int llmtok_find(const LlamaTokenizer *t, const char *s);

#ifdef __cplusplus
}
#endif

#endif /* MO_LLAMA_TOKENIZER_H */
