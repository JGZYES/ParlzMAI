#ifndef MO_GGUF_H
#define MO_GGUF_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF (llama.cpp) 容器读取。v2/v3。 */
typedef struct {
    const unsigned char *data;
    size_t size;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t meta_count;
    size_t data_offset;          /* 张量数据段起点（绝对文件偏移） */

    /* 张量表 */
    char **tensor_name;
    uint32_t *n_dims;
    uint64_t **dims;             /* [i][d] */
    uint32_t *ggml_type;
    uint64_t *offset;            /* 相对 data_offset */

    /* 元数据（供查询） */
    char **meta_key;
    uint32_t *meta_type;
    const unsigned char **meta_off;   /* value 起始（在 buffer 内） */
    size_t *meta_len;                 /* value 字节数 */
} Gguf;

/* 打开并解析文件。成功返回 0，失败返回负。 */
int gguf_open(Gguf *g, const char *path);
void gguf_close(Gguf *g);

/* 元数据查询：返回 1 命中，0 未命中 */
int gguf_meta_u32(const Gguf *g, const char *key, uint32_t *v);
int gguf_meta_f32(const Gguf *g, const char *key, float *v);
int gguf_meta_u32_array(const Gguf *g, const char *key, uint32_t *out, uint64_t max);
int gguf_meta_u8_array(const Gguf *g, const char *key, uint8_t *out, uint64_t max);
int gguf_meta_f32_array(const Gguf *g, const char *key, float *out, uint64_t max);
int gguf_meta_string(const Gguf *g, const char *key, char *out, size_t max);
/* 读字符串数组（如 tokenizer.ggml.tokens/merges）。成功返回 1，*out 需调用 gguf_free_string_array 释放 */
int gguf_meta_string_array(const Gguf *g, const char *key, char ***out, int *count);
void gguf_free_string_array(char **arr, int count);

/* 打印元数据与张量表到 out（任意 GGUF 都能看）。 */
void gguf_inspect(const Gguf *g, const char *path, FILE *out);

/* 把名为 name 的张量反量化成 f32 写入 out（out 长度需 rows*cols）。成功返回字节数，失败<0 */
/* 这里张量按 flat 元素看待，返回元素个数。 */
long gguf_tensor_to_f32(const Gguf *g, const char *name, float *out);
int gguf_has_tensor(const Gguf *g, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* MO_GGUF_H */
