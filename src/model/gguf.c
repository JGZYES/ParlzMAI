#include "model/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

/* ---- ggml_type 枚举（常用） ---- */
#define GGML_F32 0
#define GGML_F16 1
#define GGML_Q4_0 2
#define GGML_Q5_0 6
#define GGML_Q5_1 7
#define GGML_Q8_0 8
#define GGML_BF16 16

/* ---- 元数据值类型 ---- */
#define GGVAL_UINT8 0
#define GGVAL_INT8 1
#define GGVAL_UINT16 2
#define GGVAL_INT16 3
#define GGVAL_UINT32 4
#define GGVAL_INT32 5
#define GGVAL_FLOAT32 6
#define GGVAL_BOOL 7
#define GGVAL_STRING 8
#define GGVAL_ARRAY 9
#define GGVAL_UINT64 10
#define GGVAL_INT64 11
#define GGVAL_FLOAT64 12

static const unsigned char *gd(const Gguf *g, size_t off) { return g->data + off; }

static uint64_t rd_u64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static uint32_t rd_u32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static uint16_t rd_u16(const unsigned char *p) {
    uint16_t v; memcpy(&v, p, 2); return v;
}
static float rd_f32(const unsigned char *p) {
    float v; memcpy(&v, p, 4); return v;
}

static float bf16_to_f32(uint16_t h) {
    uint32_t f = (uint32_t)h << 16;
    float v; memcpy(&v, &f, 4); return v;
}

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    float f;
    if (exp == 0x1fu) { bits = (sign << 31) | 0x7f800000u | (mant << 13); memcpy(&f, &bits, 4); return f; }
    if (exp == 0) {
        if (mant == 0) { bits = sign << 31; memcpy(&f, &bits, 4); return f; }
        f = (float)mant * 5.9604644775390625e-8f;
        return sign ? -f : f;
    }
    bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    memcpy(&f, &bits, 4);
    return f;
}

/* 块大小 / 每块字节数（llama.cpp ggml 类型；Q4_0 等 block=32） */
static uint32_t type_block_bytes(uint32_t t) {
    switch (t) {
        case GGML_F32:   return 32 * 4;
        case GGML_F16:   return 32 * 2;
        case GGML_BF16:  return 32 * 2;
        case GGML_Q8_0:  return 2 + 32;              /* d(f16) + 32*int8 */
        case GGML_Q4_0:  return 2 + 16;              /* d(f16) + 16*4bit(=32) */
        case GGML_Q5_0:  return 2 + 16 + 4;          /* d + low + qh */
        case GGML_Q5_1:  return 2 + 2 + 16 + 4;      /* d + m + low + qh */
        default:         return 0;
    }
}

/* 反量化一个块（32 个）到 out */
static void dequant_block(uint32_t t, const unsigned char *p, float *out) {
    int i;
    switch (t) {
        case GGML_F32: {
            for (i = 0; i < 32; i++) out[i] = rd_f32(p + i * 4);
            break;
        }
        case GGML_F16: {
            for (i = 0; i < 32; i++) out[i] = fp16_to_f32(*(const uint16_t *)(p + i * 2));
            break;
        }
        case GGML_BF16: {
            for (i = 0; i < 32; i++) out[i] = bf16_to_f32(*(const uint16_t *)(p + i * 2));
            break;
        }
        case GGML_Q8_0: {
            float d = fp16_to_f32(*(const uint16_t *)p);
            const int8_t *q = (const int8_t *)(p + 2);
            for (i = 0; i < 32; i++) out[i] = (float)q[i] * d;
            break;
        }
        case GGML_Q4_0: {
            float d = fp16_to_f32(*(const uint16_t *)p);
            const unsigned char *q = p + 2;
            for (i = 0; i < 16; i++) {
                int lo = q[i] & 0x0F;
                int hi = (q[i] >> 4) & 0x0F;
                out[i * 2 + 0] = (float)(lo - 8) * d;
                out[i * 2 + 1] = (float)(hi - 8) * d;
            }
            break;
        }
        case GGML_Q5_0: {
            float d = fp16_to_f32(*(const uint16_t *)p);
            const unsigned char *ql = p + 2;
            const unsigned char *qh = p + 2 + 16;   /* 4 字节，bit w 对应权重 w 的高位 */
            for (int w = 0; w < 32; w++) {
                int low = (w & 1) ? (ql[w / 2] >> 4) & 0x0F : ql[w / 2] & 0x0F;
                int high = (qh[w / 8] >> (w % 8)) & 1;
                out[w] = (float)(low + (high << 4) - 16) * d;
            }
            break;
        }
        case GGML_Q5_1: {
            float d = fp16_to_f32(*(const uint16_t *)p);
            float m = fp16_to_f32(*(const uint16_t *)(p + 2));
            const unsigned char *ql = p + 4;
            const unsigned char *qh = p + 4 + 16;
            for (int w = 0; w < 32; w++) {
                int low = (w & 1) ? (ql[w / 2] >> 4) & 0x0F : ql[w / 2] & 0x0F;
                int high = (qh[w / 8] >> (w % 8)) & 1;
                out[w] = (float)(low + (high << 4)) * d + m;
            }
            break;
        }
        default: {
            for (i = 0; i < 32; i++) out[i] = 0.0f;
            break;
        }
    }
}

/* ---- 张量 ---- */

int gguf_has_tensor(const Gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        if (strcmp(g->tensor_name[i], name) == 0) return 1;
    }
    return 0;
}

/* 计算张量元素个数 = prod(dims) */
static uint64_t tensor_n(const Gguf *g, uint64_t i) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < g->n_dims[i]; d++) n *= g->dims[i][d];
    return n;
}

long gguf_tensor_to_f32(const Gguf *g, const char *name, float *out) {
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        if (strcmp(g->tensor_name[i], name) != 0) continue;
        uint64_t n = tensor_n(g, i);
        uint32_t t = g->ggml_type[i];
        uint32_t bb = type_block_bytes(t);
        if (bb == 0) return -1;
        const unsigned char *p = gd(g, g->data_offset + (size_t)g->offset[i]);
        uint64_t blocks = (n + 31) / 32;
        uint64_t k = 0;
        for (uint64_t b = 0; b < blocks; b++) {
            float tmp[32];
            dequant_block(t, p + (size_t)b * bb, tmp);
            for (int j = 0; j < 32 && k < n; j++) out[k++] = tmp[j];
        }
        return (long)n;
    }
    return -1;
}

/* ---- 解析 ---- */

/* 版本自适应长度：GGUF v1 用 u32，v2/v3 用 u64 */
static size_t rd_len(int ver, const unsigned char *p, size_t *off) {
    uint64_t v;
    if (ver >= 2) { v = rd_u64(p + *off); *off += 8; }
    else          { v = rd_u32(p + *off); *off += 4; }
    return (size_t)v;
}

/* 跳过 value，返回跳过的字节数。ver 用于数组/字符串长度宽度 */
static size_t skip_gval(int ver, const unsigned char *p, uint32_t vt) {
    switch (vt) {
        case GGVAL_UINT8: case GGVAL_INT8: case GGVAL_BOOL: return 1;
        case GGVAL_UINT16: case GGVAL_INT16: return 2;
        case GGVAL_UINT32: case GGVAL_INT32: case GGVAL_FLOAT32: return 4;
        case GGVAL_FLOAT64: case GGVAL_UINT64: case GGVAL_INT64: return 8;
        case GGVAL_STRING: { size_t o = 0; size_t n = rd_len(ver, p, &o); return o + n; }
        case GGVAL_ARRAY: {
            size_t o = 0;
            uint32_t it = rd_u32(p + o); o += 4;
            size_t n = rd_len(ver, p, &o);
            const unsigned char *q = p + o;
            for (size_t i = 0; i < n; i++) q += skip_gval(ver, q, it);
            return (size_t)(q - p);
        }
        default: return 0;
    }
}

static char *read_gstr(int ver, const unsigned char *p, size_t *off) {
    size_t n = rd_len(ver, p, off);
    char *s = malloc((size_t)n + 1);
    memcpy(s, p + *off, (size_t)n);
    s[n] = '\0';
    *off += (size_t)n;
    return s;
}

int gguf_open(Gguf *g, const char *path) {
    memset(g, 0, sizeof(*g));
    FILE *fh = fopen(path, "rb");
    if (!fh) return -1;
    fseek(fh, 0, SEEK_END);
    long sz = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    if (sz <= 0) { fclose(fh); return -2; }
    unsigned char *buf = malloc((size_t)sz);
    if (!buf) { fclose(fh); return -3; }
    if (fread(buf, 1, (size_t)sz, fh) != (size_t)sz) { free(buf); fclose(fh); return -4; }
    fclose(fh);

    if (sz < 24) { free(buf); return -5; }
    if (memcmp(buf, "GGUF", 4) != 0) { free(buf); return -6; }
    g->data = buf;
    g->size = (size_t)sz;

    size_t off = 4;
    g->version = rd_u32(buf + off); off += 4;
    g->tensor_count = rd_len(g->version, buf, &off);
    g->meta_count = rd_len(g->version, buf, &off);

    /* 探测段顺序：GGUF 生产者可能写“元数据在前”或“张量在前”。
       张量名总以 .weight 结尾；元数据键不会。 */
    int meta_first = 1;
    {
        size_t o = off;
        size_t n0 = rd_len(g->version, buf, &o);
        if (n0 >= 7 && memcmp(buf + o + n0 - 7, ".weight", 7) == 0) meta_first = 0;
    }

    /* 元数据 */
    g->meta_key = calloc((size_t)g->meta_count, sizeof(char *));
    g->meta_type = calloc((size_t)g->meta_count, sizeof(uint32_t));
    g->meta_off = calloc((size_t)g->meta_count, sizeof(const unsigned char *));
    g->meta_len = calloc((size_t)g->meta_count, sizeof(size_t));

    /* 张量信息 */
    g->tensor_name = calloc((size_t)g->tensor_count, sizeof(char *));
    g->n_dims = calloc((size_t)g->tensor_count, sizeof(uint32_t));
    g->dims = calloc((size_t)g->tensor_count, sizeof(uint64_t *));
    g->ggml_type = calloc((size_t)g->tensor_count, sizeof(uint32_t));
    g->offset = calloc((size_t)g->tensor_count, sizeof(uint64_t));

#define READ_METADATA() \
    for (uint64_t i = 0; i < g->meta_count; i++) { \
        g->meta_key[i] = read_gstr(g->version, buf, &off); \
        uint32_t vt = rd_u32(buf + off); off += 4; \
        g->meta_type[i] = vt; \
        g->meta_off[i] = buf + off; \
        size_t len = skip_gval(g->version, buf + off, vt); \
        g->meta_len[i] = len; \
        off += len; \
    }
#define READ_TENSORS() \
    for (uint64_t i = 0; i < g->tensor_count; i++) { \
        g->tensor_name[i] = read_gstr(g->version, buf, &off); \
        g->n_dims[i] = rd_u32(buf + off); off += 4; \
        g->dims[i] = malloc((size_t)g->n_dims[i] * sizeof(uint64_t)); \
        for (uint32_t d = 0; d < g->n_dims[i]; d++) { \
            g->dims[i][d] = rd_len(g->version, buf, &off); \
        } \
        g->ggml_type[i] = rd_u32(buf + off); off += 4; \
        g->offset[i] = rd_len(g->version, buf, &off); \
    }

    if (meta_first) { READ_METADATA(); READ_TENSORS(); }
    else            { READ_TENSORS(); READ_METADATA(); }

#undef READ_METADATA
#undef READ_TENSORS

    g->data_offset = off;

    return 0;
}

/* 查找元数据索引，未命中返回 -1 */
static ptrdiff_t meta_index(const Gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->meta_count; i++) {
        if (strcmp(g->meta_key[i], key) == 0) return (ptrdiff_t)i;
    }
    return -1;
}

int gguf_meta_u32(const Gguf *g, const char *key, uint32_t *v) {
    ptrdiff_t i = meta_index(g, key);
    if (i < 0) return 0;
    const unsigned char *p = g->meta_off[i];
    switch (g->meta_type[i]) {
        case GGVAL_UINT8:   *v = p[0]; return 1;
        case GGVAL_UINT16:  *v = rd_u16(p); return 1;
        case GGVAL_UINT32:  *v = rd_u32(p); return 1;
        case GGVAL_INT32:   *v = rd_u32(p); return 1;
        case GGVAL_UINT64:  *v = (uint32_t)rd_u64(p); return 1;
        case GGVAL_INT64:   *v = (uint32_t)rd_u64(p); return 1;
        default: return 0;
    }
}

int gguf_meta_f32(const Gguf *g, const char *key, float *v) {
    ptrdiff_t i = meta_index(g, key);
    if (i < 0) return 0;
    const unsigned char *p = g->meta_off[i];
    if (g->meta_type[i] == GGVAL_FLOAT32) { *v = rd_f32(p); return 1; }
    if (g->meta_type[i] == GGVAL_FLOAT64) {
        union { double d; uint64_t b; } u; memcpy(&u.b, p, 8); *v = (float)u.d; return 1;
    }
    return 0;
}

int gguf_meta_u32_array(const Gguf *g, const char *key, uint32_t *out, uint64_t max) {
    ptrdiff_t i = meta_index(g, key);
    if (i < 0) return 0;
    const unsigned char *p = g->meta_off[i];
    if (g->meta_type[i] != GGVAL_ARRAY) return 0;
    uint32_t it = rd_u32(p);
    size_t o = 4;
    size_t n = rd_len(g->version, p, &o);
    if (it != GGVAL_UINT32) return 0;
    size_t take = n < max ? n : max;
    for (size_t k = 0; k < take; k++) out[k] = rd_u32(p + o + k * 4);
    return 1;
}

int gguf_meta_u8_array(const Gguf *g, const char *key, uint8_t *out, uint64_t max) {
    ptrdiff_t i = meta_index(g, key);
    if (i < 0) return 0;
    const unsigned char *p = g->meta_off[i];
    if (g->meta_type[i] == GGVAL_ARRAY) {
        uint32_t it = rd_u32(p);
        size_t o = 4;
        size_t n = rd_len(g->version, p, &o);
        if (it != GGVAL_UINT8) return 0;
        size_t take = n < max ? n : max;
        for (size_t k = 0; k < take; k++) out[k] = p[o + k];
        return 1;
    }
    if (g->meta_type[i] == GGVAL_STRING) {
        size_t o = 0;
        size_t n = rd_len(g->version, p, &o);
        size_t take = n < max ? n : max;
        for (size_t k = 0; k < take; k++) out[k] = p[o + k];
        return 1;
    }
    return 0;
}

int gguf_meta_string(const Gguf *g, const char *key, char *out, size_t max) {
    ptrdiff_t i = meta_index(g, key);
    if (i < 0) return 0;
    const unsigned char *p = g->meta_off[i];
    if (g->meta_type[i] == GGVAL_STRING) {
        size_t o = 0;
        size_t n = rd_len(g->version, p, &o);
        size_t take = (size_t)n < max ? (size_t)n : max - 1;
        memcpy(out, p + o, take);
        out[take] = '\0';
        return 1;
    }
    return 0;
}

int gguf_meta_string_array(const Gguf *g, const char *key, char ***out, int *count) {
    ptrdiff_t i = meta_index(g, key);
    if (i < 0) return 0;
    const unsigned char *p = g->meta_off[i];
    if (g->meta_type[i] != GGVAL_ARRAY) return 0;
    uint32_t it = rd_u32(p);
    size_t o = 4;
    size_t n = rd_len(g->version, p, &o);
    if (it != GGVAL_STRING) return 0;
    char **arr = calloc((size_t)n, sizeof(char *));
    const unsigned char *q = p + o;
    for (size_t k = 0; k < n; k++) {
        size_t so = 0;
        size_t sl = rd_len(g->version, q, &so);
        arr[k] = malloc((size_t)sl + 1);
        memcpy(arr[k], q + so, (size_t)sl);
        arr[k][sl] = '\0';
        q += so + (size_t)sl;
    }
    if (out) *out = arr;
    if (count) *count = (int)n;
    return 1;
}

void gguf_free_string_array(char **arr, int count) {
    if (!arr) return;
    for (int i = 0; i < count; i++) free(arr[i]);
    free(arr);
}

void gguf_close(Gguf *g) {
    if (!g) return;
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        free(g->tensor_name[i]);
        free(g->dims[i]);
    }
    for (uint64_t i = 0; i < g->meta_count; i++) free(g->meta_key[i]);
    free(g->tensor_name);
    free(g->n_dims);
    free(g->dims);
    free(g->ggml_type);
    free(g->offset);
    free(g->meta_key);
    free(g->meta_type);
    free(g->meta_off);
    free(g->meta_len);
    free((void *)g->data);
    memset(g, 0, sizeof(*g));
}

static const char *type_name(uint32_t t) {
    switch (t) {
        case GGML_F32: return "F32";
        case GGML_F16: return "F16";
        case GGML_BF16: return "BF16";
        case GGML_Q8_0: return "Q8_0";
        case GGML_Q4_0: return "Q4_0";
        case GGML_Q5_0: return "Q5_0";
        case GGML_Q5_1: return "Q5_1";
        default: return "?";
    }
}

void gguf_inspect(const Gguf *g, const char *path, FILE *out) {
    fprintf(out, "==== GGUF 文件: %s ====\n", path);
    fprintf(out, "version=%u  tensors=%llu  metadata=%llu  data_offset=%zu\n",
            g->version,
            (unsigned long long)g->tensor_count,
            (unsigned long long)g->meta_count,
            g->data_offset);
    fprintf(out, "--- 张量表 ---\n");
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        uint64_t n = tensor_n(g, i);
        fprintf(out, "  %-32s dims=", g->tensor_name[i]);
        for (uint32_t d = 0; d < g->n_dims[i]; d++) fprintf(out, "%llu%s",
                (unsigned long long)g->dims[i][d], d + 1 < g->n_dims[i] ? "x" : "");
        fprintf(out, "  type=%s  elems=%llu  bytes~%llu\n",
                type_name(g->ggml_type[i]), (unsigned long long)n,
                (unsigned long long)(type_block_bytes(g->ggml_type[i]) * ((n + 31) / 32)));
    }
}
