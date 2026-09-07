#ifndef MO_LLAMA_TOKENIZER_SPM_H
#define MO_LLAMA_TOKENIZER_SPM_H

#ifdef __cplusplus
extern "C" {
#endif

/* llama / SentencePiece (SPM) 分词器：▁ 空格 + score 优先 BPE + 字节回退（对齐 llama.cpp llm_tokenizer_spm） */
typedef struct {
    int   vocab_size;
    char **tokens;      /* 深拷贝 */
    int  *sort_idx;     /* 按字符串排序的词表索引，用于二分查找 */
    float *scores;      /* [vocab_size] 用于 BPE 优先级 */
} LlamaTokenizerSPM;

int llmtok_spm_init(LlamaTokenizerSPM *t, char **tokens, int vocab_size, float *scores);
void llmtok_spm_free(LlamaTokenizerSPM *t);
int llmtok_spm_encode(const LlamaTokenizerSPM *t, const char *text, int *out, int max);
int llmtok_spm_decode(const LlamaTokenizerSPM *t, const int *ids, int n, char *out, int max);

#ifdef __cplusplus
}
#endif

#endif /* MO_LLAMA_TOKENIZER_SPM_H */
