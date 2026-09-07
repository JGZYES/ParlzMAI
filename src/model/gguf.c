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
#define GGML_Q2_K 10
#define GGML_Q3_K 11
#define GGML_Q4_K 12
#define GGML_Q5_K 13
#define GGML_Q6_K 14
#define GGML_Q8_K 15
#define GGML_BF16 16
#define QK_K 256
#define K_SCALE_SIZE 12

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
        case GGML_Q8_0:  return 2 + 32;
        case GGML_Q4_0:  return 2 + 16;
        case GGML_Q5_0:  return 2 + 16 + 4;
        case GGML_Q5_1:  return 2 + 2 + 16 + 4;
        case GGML_Q2_K:  return 84;   /* QK_K=256 */
        case GGML_Q3_K:  return 110;
        case GGML_Q4_K:  return 144;
        case GGML_Q5_K:  return 176;
        case GGML_Q6_K:  return 210;
        case GGML_Q8_K:  return 274;
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
                out[i]         = (float)(lo - 8) * d;   /* 低 nibble -> 0..15 */
                out[i + 16]    = (float)(hi - 8) * d;   /* 高 nibble -> 16..31 */
            }
            break;
        }
        case GGML_Q5_0: {
            float d = fp16_to_f32(*(const uint16_t *)p);
            const unsigned char *ql = p + 2;
            const unsigned char *qh = p + 2 + 16;
            for (i = 0; i < 16; i++) {
                int low = ql[i] & 0x0F;
                int high = (ql[i] >> 4) & 0x0F;
                int s0 = (qh[i * 2 / 2] >> 0) & 0x01;             /* bit of weight 2i? 直接按位 */
                int s0b = (qh[i] & 0x01) << 4;
                int s1b = (qh[i] & 0x08) << 2;
                out[i]      = (float)(low + s0b + 16 - 32) * d;   /* +16, -32 -> 中心化 */
                out[i + 16] = (float)(high + s1b + 16 - 32) * d;
            }
            break;
        }
        case GGML_Q5_1: {
            float d = fp16_to_f32(*(const uint16_t *)p);
            float m = fp16_to_f32(*(const uint16_t *)(p + 2));
            const unsigned char *ql = p + 4;
            const unsigned char *qh = p + 4 + 16;
            for (i = 0; i < 16; i++) {
                int low = ql[i] & 0x0F;
                int high = (ql[i] >> 4) & 0x0F;
                int s0b = (qh[i] & 0x01) << 4;
                int s1b = (qh[i] & 0x08) << 2;
                out[i]      = (float)(low + s0b) * d + m;
                out[i + 16] = (float)(high + s1b) * d + m;
            }
            break;
        }
        default: {
            for (i = 0; i < 32; i++) out[i] = 0.0f;
            break;
        }
    }
}

/* ---- K 系量化（QK_K=256 块）反量化 ---- */

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

static void dequant_k(uint32_t t, const unsigned char *p, float *out) {
    int i;
    if (t == GGML_Q8_K) {
        float d = fp16_to_f32(*(const uint16_t *)(p));
        const int8_t *qs = (const int8_t *)(p + 2);
        for (i = 0; i < 256; i++) out[i] = d * qs[i];
    } else if (t == GGML_Q6_K) {
        const uint8_t *ql = p;          /* 128 */
        const uint8_t *qh = p + 128;    /* 64  */
        const int8_t *sc0 = (const int8_t *)(p + 192); /* 16 */
        float d = fp16_to_f32(*(const uint16_t *)(p + 208));
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[n + l + 0]  = d * sc0[is + 0] * q1;
                out[n + l + 32] = d * sc0[is + 2] * q2;
                out[n + l + 64] = d * sc0[is + 4] * q3;
                out[n + l + 96] = d * sc0[is + 6] * q4;
            }
            ql += 64; qh += 32; sc0 += 8;
        }
    } else if (t == GGML_Q4_K || t == GGML_Q5_K) {
        float d = fp16_to_f32(*(const uint16_t *)(p));
        float mn = fp16_to_f32(*(const uint16_t *)(p + 2));
        const uint8_t *scales = p + 4;      /* 12 */
        const uint8_t *qh = (t == GGML_Q5_K) ? (p + 16) : NULL; /* 32 */
        const uint8_t *ql = (t == GGML_Q5_K) ? (p + 48) : (p + 16); /* 128 */
        if (t == GGML_Q5_K) {
            int is = 0; uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, &sc, &m);
                float d1 = d * sc, m1 = mn * m;
                get_scale_min_k4(is + 1, scales, &sc, &m);
                float d2 = d * sc, m2 = mn * m;
                for (int l = 0; l < 32; l++) { out[l] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1; out[l + 32] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2; }
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2; out += 64;
            }
        } else { /* Q4_K */
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, &sc, &m);
                float d1 = d * sc, m1 = mn * m;
                get_scale_min_k4(is + 1, scales, &sc, &m);
                float d2 = d * sc, m2 = mn * m;
                for (int l = 0; l < 32; l++) { out[l] = d1 * (ql[l] & 0xF) - m1; out[l + 32] = d2 * (ql[l] >> 4) - m2; }
                ql += 32; is += 2; out += 64;
            }
        }
    } else if (t == GGML_Q2_K) {
        float d = fp16_to_f32(*(const uint16_t *)(p));
        float mn = fp16_to_f32(*(const uint16_t *)(p + 2));
        const uint8_t *scales = p + 4; /* 16 */
        const uint8_t *q = p + 20;     /* 64 */
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
                for (int l = 0; l < 16; l++) *out++ = dl * (float)((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = mn * (sc >> 4);
                for (int l = 0; l < 16; l++) *out++ = dl * (float)((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    } else { /* Q3_K */
        const uint8_t *hm = p;          /* hmask 32 */
        const uint8_t *q3 = p + 32;     /* qs 64 */
        const uint8_t *sc3 = p + 96;    /* scales 12 */
        float d = fp16_to_f32(*(const uint16_t *)(p + 108));
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint32_t aux[4]; const int8_t *scales = (const int8_t *)aux;
        uint32_t tmp;
        memcpy(aux, sc3, 12);
        tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0; uint8_t m = 1;
            for (int j = 0; j < 4; j++) {
                float dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) { *out++ = dl * ((int8_t)((q3[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)); }
                dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) { *out++ = dl * ((int8_t)((q3[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4)); }
                shift += 2; m <<= 1;
            }
            q3 += 32;
        }
    }
}

static int is_k_type(int t) {
    return t == GGML_Q2_K || t == GGML_Q3_K || t == GGML_Q4_K || t == GGML_Q5_K || t == GGML_Q6_K || t == GGML_Q8_K;
}

/* ggml 行大小：张量最后一维(ne0) 按块对齐补零后的字节数 */
static size_t ggml_row_size(uint32_t t, uint64_t ne0) {
    switch (t) {
        case GGML_F32:  return (size_t)ne0 * 4;
        case GGML_F16:  return (size_t)ne0 * 2;
        case GGML_BF16: return (size_t)ne0 * 2;
        default: {
            uint32_t bs = is_k_type(t) ? 256u : 32u;
            uint64_t nblocks = (ne0 + bs - 1) / bs;
            return (size_t)nblocks * type_block_bytes(t);
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
        uint32_t t = g->ggml_type[i];
        uint32_t bb = type_block_bytes(t);
        if (bb == 0) return -1;
        uint64_t ne0 = g->dims[i][0];
        uint64_t rows = 1;
        for (uint32_t d = 1; d < g->n_dims[i]; d++) rows *= g->dims[i][d];
        size_t row_size = ggml_row_size(t, ne0);
        const unsigned char *p = gd(g, g->data_offset + (size_t)g->offset[i]);
        int kq = is_k_type(t);
        uint32_t bs = kq ? 256u : 32u;
        uint64_t nblocks = (ne0 + bs - 1) / bs;
        uint64_t k = 0;
        for (uint64_t r = 0; r < rows; r++) {
            const unsigned char *pr = p + r * row_size;
            for (uint64_t b = 0; b < nblocks; b++) {
                int remain = (int)(ne0 - b * bs);
                int take = remain < (int)bs ? remain : (int)bs;
                if (kq) {
                    float tmp[256];
                    dequant_k(t, pr + b * bb, tmp);
                    for (int j = 0; j < take; j++) out[k++] = tmp[j];
                } else {
                    float tmp[32];
                    dequant_block(t, pr + b * bb, tmp);
                    for (int j = 0; j < take; j++) out[k++] = tmp[j];
                }
            }
        }
        return (long)k;
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

    /* GGUF 张量数据区起点需按对齐值向上取整：data_offset = align_up(off, alignment)。
       对齐值取元数据 general.alignment（必须非 0 且为 2 的幂），否则用默认 32。
       若不对齐，所有张量数据整体偏移若干字节 → 反量化读到垃圾 → 输出 <unk>。 */
    {
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < g->meta_count; i++) {
            if (strcmp(g->meta_key[i], "general.alignment") == 0) {
                alignment = g->meta_type[i] == GGVAL_UINT32
                    ? (uint64_t)rd_u32(g->meta_off[i])
                    : (uint64_t)rd_u64(g->meta_off[i]);
                break;
            }
        }
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) alignment = 32;
        g->data_offset = (off + alignment - 1) & ~(alignment - 1);
    }

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

int gguf_meta_f32_array(const Gguf *g, const char *key, float *out, uint64_t max) {
    ptrdiff_t i = meta_index(g, key);
    if (i < 0) return 0;
    const unsigned char *p = g->meta_off[i];
    if (g->meta_type[i] != GGVAL_ARRAY) return 0;
    uint32_t it = rd_u32(p);
    size_t o = 4;
    size_t n = rd_len(g->version, p, &o);
    if (it != GGVAL_FLOAT32) return 0;
    size_t take = n < max ? n : max;
    for (size_t k = 0; k < take; k++) out[k] = rd_f32(p + o + k * 4);
    return 1;
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
        case GGML_Q2_K: return "Q2_K";
        case GGML_Q3_K: return "Q3_K";
        case GGML_Q4_K: return "Q4_K";
        case GGML_Q5_K: return "Q5_K";
        case GGML_Q6_K: return "Q6_K";
        case GGML_Q8_K: return "Q8_K";
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
