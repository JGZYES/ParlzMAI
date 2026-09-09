#include "model/model.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "utils/portable.h"

#ifdef _WIN32
/* Windows：无 mmap，退化为读整文件到内存（mmap 只是加载优化，不影响正确性） */
#include <io.h>
static unsigned char *mo_map_file(const char *path, size_t *out_sz) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int64_t n = mo_file_size(f);   /* 64 位，避免 Windows long(32 位) 对 >2GB 溢出 */
    if (n <= 0) { fclose(f); return NULL; }
    unsigned char *b = malloc((size_t)n);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f);
    *out_sz = (size_t)n;
    return b;
}
static void mo_unmap_file(unsigned char *b, size_t sz) { (void)sz; free(b); }
#else
#include <sys/mman.h>
static unsigned char *mo_map_file(const char *path, size_t *out_sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    off_t sz = lseek(fd, 0, SEEK_END); lseek(fd, 0, SEEK_SET);
    if (sz <= 0) { close(fd); return NULL; }
    unsigned char *b = mmap(NULL, (size_t)sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (b == MAP_FAILED) return NULL;
    *out_sz = (size_t)sz;
    return b;
}
static void mo_unmap_file(unsigned char *b, size_t sz) { if (b && b != MAP_FAILED) munmap(b, sz); }
#endif

#include "model/gguf.h"

#define PAP_MAGIC "PAP1"

static uint32_t rd_u32(const unsigned char *b, size_t *off) {
    uint32_t v;
    memcpy(&v, b + *off, 4);
    *off += 4;
    return v;
}

static float rd_f32(const unsigned char *b, size_t *off) {
    float v;
    memcpy(&v, b + *off, 4);
    *off += 4;
    return v;
}

static int expect_magic(const unsigned char *b, size_t *off) {
    if (memcmp(b + *off, PAP_MAGIC, 4) != 0) return -1;
    *off += 4;
    return 0;
}

/* 精确匹配：sscanf 成功赋值且整个名字都被消费（避免尾部字面量未匹配却部分成功） */
static int m1(const char *name, const char *fmt, int *a) {
    int n = 0;
    int r = sscanf(name, fmt, a, &n);
    return (r == 1) && name[n] == '\0';
}
static int m2(const char *name, const char *fmt, int *a, int *b) {
    int n = 0;
    int r = sscanf(name, fmt, a, b, &n);
    return (r == 2) && name[n] == '\0';
}

/* 把第 layer 层第 expert 个 MoE 专家的 w1/w2 指针（按需分配）填入 m。 */
static void ensure_moe_expert(Model *m, int layer, int expert) {
    if (!m->exp_w1[layer][expert]) {
        m->exp_w1[layer][expert] = malloc((size_t)m->config.d_model * m->config.d_expert * sizeof(float));
        m->exp_w2[layer][expert] = malloc((size_t)m->config.d_expert * m->config.d_model * sizeof(float));
    }
}

static void load_weight(Model *m, const char *name, uint32_t rows, uint32_t cols, const float *src) {
    (void)rows;
    (void)cols;
    ModelConfig *c = &m->config;
    int l, e;

    if (strcmp(name, "wte") == 0) {
        memcpy(m->wte, src, (size_t)c->vocab_size * c->d_model * sizeof(float));
    } else if (strcmp(name, "wpe") == 0) {
        memcpy(m->wpe, src, (size_t)c->max_seq_len * c->d_model * sizeof(float));
    } else if (strcmp(name, "norm_final") == 0) {
        memcpy(m->norm_final, src, (size_t)c->d_model * sizeof(float));
    } else if (strcmp(name, "lm_head") == 0) {
        memcpy(m->lm_head, src, (size_t)c->vocab_size * c->d_model * sizeof(float));
    } else if (m1(name, "l%d.norm1%n", &l)) {
        memcpy(m->l_norm1[l], src, (size_t)c->d_model * sizeof(float));
    } else if (m1(name, "l%d.attn.wq%n", &l)) {
        memcpy(m->wq[l], src, (size_t)c->d_model * c->d_model * sizeof(float));
    } else if (m1(name, "l%d.attn.wk%n", &l)) {
        memcpy(m->wk[l], src, (size_t)c->d_model * c->d_model * sizeof(float));
    } else if (m1(name, "l%d.attn.wv%n", &l)) {
        memcpy(m->wv[l], src, (size_t)c->d_model * c->d_model * sizeof(float));
    } else if (m1(name, "l%d.attn.wo%n", &l)) {
        memcpy(m->wo[l], src, (size_t)c->d_model * c->d_model * sizeof(float));
    } else if (m1(name, "l%d.norm2%n", &l)) {
        memcpy(m->l_norm2[l], src, (size_t)c->d_model * sizeof(float));
    } else if (m1(name, "l%d.moe.router%n", &l)) {
        memcpy(m->router[l], src, (size_t)c->d_model * c->moe_n_experts * sizeof(float));
    } else if (m2(name, "l%d.moe.exp%d.w1%n", &l, &e)) {
        ensure_moe_expert(m, l, e);
        memcpy(m->exp_w1[l][e], src, (size_t)c->d_model * c->d_expert * sizeof(float));
    } else if (m2(name, "l%d.moe.exp%d.w2%n", &l, &e)) {
        ensure_moe_expert(m, l, e);
        memcpy(m->exp_w2[l][e], src, (size_t)c->d_expert * c->d_model * sizeof(float));
    } else if (m1(name, "l%d.ffn.w1%n", &l)) {
        memcpy(m->ffn_w1[l], src, (size_t)c->d_model * c->d_ff * sizeof(float));
    } else if (m1(name, "l%d.ffn.w2%n", &l)) {
        memcpy(m->ffn_w2[l], src, (size_t)c->d_ff * c->d_model * sizeof(float));
    } else {
        fprintf(stderr, "[warn] 忽略未知权重: %s\n", name);
    }
}

/* fp16 -> fp32 */
static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    float f;
    if (exp == 0x1fu) {  /* inf / nan */
        bits = (sign << 31) | 0x7f800000u | (mant << 13);
        memcpy(&f, &bits, 4);
        return f;
    }
    if (exp == 0) {      /* 0 或次正规 */
        if (mant == 0) { bits = sign << 31; memcpy(&f, &bits, 4); return f; }
        f = (float)mant * 5.9604644775390625e-8f;  /* mant * 2^-24 */
        return sign ? -f : f;
    }
    bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    memcpy(&f, &bits, 4);
    return f;
}

/* 把原始权重字节按 dtype 反量化成 f32 写入 dst，返回读走的字节数 */
static size_t read_weight_f32(const unsigned char *p, uint32_t rows, uint32_t cols,
                              uint32_t dtype, float *dst) {
    size_t n = (size_t)rows * cols;
    if (dtype == 0) {
        memcpy(dst, p, n * 4);
        return n * 4;
    }
    if (dtype == 1) {
        for (size_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, p + i * 2, 2);
            dst[i] = half_to_float(h);
        }
        return n * 2;
    }
    if (dtype == 2) {   /* q8: 每块 [fp16 scale][32 int8] */
        size_t k = 0, po = 0;
        size_t nb = (n + 31) / 32;
        for (size_t b = 0; b < nb; b++) {
            uint16_t hs;
            memcpy(&hs, p + po, 2);
            po += 2;
            float scale = half_to_float(hs);
            size_t cnt = (n - k) < 32 ? (n - k) : 32;
            for (size_t j = 0; j < cnt; j++) {
                dst[k++] = (float)(int8_t)p[po + j] * (scale / 127.0f);
            }
            po += cnt;
        }
        return po;
    }
    return 0;
}

/* 根据已就绪的 config 分配全部权重存储（供 .pap/.gguf 加载器复用）。失败返回 -1 */
static int alloc_weights(Model *m) {
    ModelConfig *c = &m->config;
    int L = c->n_layer;
    m->wte = malloc((size_t)c->vocab_size * c->d_model * sizeof(float));
    m->wpe = malloc((size_t)c->max_seq_len * c->d_model * sizeof(float));
    m->l_norm1 = calloc(L, sizeof(float *));
    m->wq = calloc(L, sizeof(float *));
    m->wk = calloc(L, sizeof(float *));
    m->wv = calloc(L, sizeof(float *));
    m->wo = calloc(L, sizeof(float *));
    m->l_norm2 = calloc(L, sizeof(float *));
    m->router = calloc(L, sizeof(float *));
    m->exp_w1 = calloc(L, sizeof(float **));
    m->exp_w2 = calloc(L, sizeof(float **));
    m->ffn_w1 = calloc(L, sizeof(float *));
    m->ffn_w2 = calloc(L, sizeof(float *));
    m->norm_final = malloc((size_t)c->d_model * sizeof(float));
    m->lm_head = malloc((size_t)c->vocab_size * c->d_model * sizeof(float));

    for (int l = 0; l < L; l++) {
        m->l_norm1[l] = malloc((size_t)c->d_model * sizeof(float));
        m->wq[l] = malloc((size_t)c->d_model * c->d_model * sizeof(float));
        m->wk[l] = malloc((size_t)c->d_model * c->d_model * sizeof(float));
        m->wv[l] = malloc((size_t)c->d_model * c->d_model * sizeof(float));
        m->wo[l] = malloc((size_t)c->d_model * c->d_model * sizeof(float));
        m->l_norm2[l] = malloc((size_t)c->d_model * sizeof(float));
        m->exp_w1[l] = calloc((size_t)c->moe_n_experts, sizeof(float *));
        m->exp_w2[l] = calloc((size_t)c->moe_n_experts, sizeof(float *));
        if (c->moe_mask[l]) {
            m->router[l] = malloc((size_t)c->d_model * c->moe_n_experts * sizeof(float));
        } else {
            m->ffn_w1[l] = malloc((size_t)c->d_model * c->d_ff * sizeof(float));
            m->ffn_w2[l] = malloc((size_t)c->d_ff * c->d_model * sizeof(float));
        }
    }
    return 0;
}

int model_load(Model *m, const char *path) {
    if (!m || !path) return -1;
    memset(m, 0, sizeof(*m));

    size_t map_sz = 0;
    unsigned char *buf = mo_map_file(path, &map_sz);
    if (!buf) { return -4; }
    size_t nrd = map_sz;   /* 整文件已映射/读入 */
    (void)nrd;

    size_t off = 0;
    ModelConfig *c = &m->config;
    config_init(c);
    if (expect_magic(buf, &off) != 0) { mo_unmap_file(buf, map_sz); return -10; }
    uint32_t version = rd_u32(buf, &off);
    if (version != 1) { mo_unmap_file(buf, map_sz); return -11; }
    uint32_t dtype = rd_u32(buf, &off);
    if (dtype > 2) { mo_unmap_file(buf, map_sz); return -12; }   /* 0=f32,1=fp16,2=q8 */

    c->vocab_size     = (int)rd_u32(buf, &off);
    c->n_layer        = (int)rd_u32(buf, &off);
    c->d_model        = (int)rd_u32(buf, &off);
    c->n_head         = (int)rd_u32(buf, &off);
    c->d_ff           = (int)rd_u32(buf, &off);
    c->moe_n_experts  = (int)rd_u32(buf, &off);
    c->moe_top_k      = (int)rd_u32(buf, &off);
    c->d_expert       = (int)rd_u32(buf, &off);
    c->max_seq_len    = (int)rd_u32(buf, &off);
    c->num_merges     = (int)rd_u32(buf, &off);
    c->n_special      = (int)rd_u32(buf, &off);
    c->tie_weights    = buf[off]; off += 4; /* 1 byte + 3 reserve */
    c->rmsnorm_eps    = rd_f32(buf, &off);
    c->head_dim       = c->d_model / c->n_head;

    /* moe_mask */
    c->moe_mask = malloc((size_t)c->n_layer);
    memcpy(c->moe_mask, buf + off, (size_t)c->n_layer);
    off += (size_t)c->n_layer;

    /* ---- vocab blob ---- */
    uint32_t blob_len = rd_u32(buf, &off);
    size_t boff = 0;
    const unsigned char *blob = buf + off;
    off += blob_len;

    unsigned char **vbytes = calloc((size_t)c->vocab_size, sizeof(unsigned char *));
    int *vlen = calloc((size_t)c->vocab_size, sizeof(int));
    for (int i = 0; i < c->vocab_size; i++) {
        uint32_t tl = rd_u32(blob, &boff);
        vbytes[i] = (unsigned char *)(blob + boff);   /* 指向 buffer 内 */
        vlen[i] = (int)tl;
        boff += tl;
    }
    uint32_t *ma = malloc((size_t)(c->num_merges > 0 ? c->num_merges : 1) * sizeof(uint32_t));
    uint32_t *mb = malloc((size_t)(c->num_merges > 0 ? c->num_merges : 1) * sizeof(uint32_t));
    for (int i = 0; i < c->num_merges; i++) {
        ma[i] = rd_u32(blob, &boff);
        mb[i] = rd_u32(blob, &boff);
    }
    tokenizer_build(&m->tok, c->vocab_size, (const unsigned char **)vbytes, vlen,
                    c->num_merges, ma, mb);
    free(vbytes);
    free(vlen);
    free(ma);
    free(mb);

    /* ---- weights 段 ---- */
    uint32_t n_tensors = rd_u32(buf, &off);

    if (alloc_weights(m) != 0) { mo_unmap_file(buf, map_sz); return -14; }

    for (uint32_t t = 0; t < n_tensors; t++) {
        uint32_t nl = rd_u32(buf, &off);
        char name[128];
        if (nl >= sizeof(name)) nl = sizeof(name) - 1;
        memcpy(name, buf + off, nl);
        name[nl] = '\0';
        off += nl;
        uint32_t rows = rd_u32(buf, &off);
        uint32_t cols = rd_u32(buf, &off);
        size_t n = (size_t)rows * cols;
        float *tmp = malloc(n * sizeof(float));
        if (!tmp) { mo_unmap_file(buf, map_sz); return -13; }
        size_t used = read_weight_f32(buf + off, rows, cols, dtype, tmp);
        off += used;
        load_weight(m, name, rows, cols, tmp);
        free(tmp);
    }

    m->is_loaded = 1;
    mo_unmap_file(buf, map_sz);
    return 0;
}

/* 从 GGUF 加载。要求使用我们定义的元数据键（见 platform/gguf_format.py）*/
int model_load_gguf(Model *m, const char *path) {
    if (!m || !path) return -1;
    memset(m, 0, sizeof(*m));

    Gguf g;
    if (gguf_open(&g, path) != 0) { fprintf(stderr, "[err] 无法解析 GGUF: %s\n", path); return -20; }

    ModelConfig *c = &m->config;
    config_init(c);

    uint32_t u;
    if (!gguf_meta_u32(&g, "pap.vocab_size", &u)) { gguf_close(&g); return -21; } c->vocab_size = (int)u;
    if (!gguf_meta_u32(&g, "pap.n_layer", &u)) { gguf_close(&g); return -21; } c->n_layer = (int)u;
    if (!gguf_meta_u32(&g, "pap.d_model", &u)) { gguf_close(&g); return -21; } c->d_model = (int)u;
    if (!gguf_meta_u32(&g, "pap.n_head", &u)) { gguf_close(&g); return -21; } c->n_head = (int)u;
    if (!gguf_meta_u32(&g, "pap.d_ff", &u)) { gguf_close(&g); return -21; } c->d_ff = (int)u;
    if (!gguf_meta_u32(&g, "pap.moe_n_experts", &u)) { gguf_close(&g); return -21; } c->moe_n_experts = (int)u;
    if (!gguf_meta_u32(&g, "pap.moe_top_k", &u)) { gguf_close(&g); return -21; } c->moe_top_k = (int)u;
    if (!gguf_meta_u32(&g, "pap.d_expert", &u)) { gguf_close(&g); return -21; } c->d_expert = (int)u;
    if (!gguf_meta_u32(&g, "pap.max_seq_len", &u)) { gguf_close(&g); return -21; } c->max_seq_len = (int)u;
    if (!gguf_meta_u32(&g, "pap.num_merges", &u)) { gguf_close(&g); return -21; } c->num_merges = (int)u;
    gguf_meta_f32(&g, "pap.rmsnorm_eps", &c->rmsnorm_eps);
    c->head_dim = c->d_model / c->n_head;

    /* moe_mask */
    if (c->n_layer > 0) {
        c->moe_mask = malloc((size_t)c->n_layer);
        uint64_t got = (uint64_t)c->n_layer;
        if (!gguf_meta_u8_array(&g, "pap.moe_mask", c->moe_mask, got)) {
            /* 缺省：全部 MoE */
            for (int l = 0; l < c->n_layer; l++) c->moe_mask[l] = (c->moe_n_experts > 0) ? 1 : 0;
        }
    }

    if (alloc_weights(m) != 0) { gguf_close(&g); return -22; }

    /* 逐张量反量化并按名装载（未知名字会因精确匹配被忽略） */
    for (uint64_t i = 0; i < g.tensor_count; i++) {
        uint64_t n = 1;
        for (uint32_t d = 0; d < g.n_dims[i]; d++) n *= g.dims[i][d];
        float *tmp = malloc((size_t)n * sizeof(float));
        if (gguf_tensor_to_f32(&g, g.tensor_name[i], tmp) > 0) {
            load_weight(m, g.tensor_name[i], 1, (uint32_t)n, tmp);
        }
        free(tmp);
    }

    /* 词表 + merges（从元数据还原） */
    if (c->vocab_size > 0) {
        uint32_t *vlen = calloc((size_t)c->vocab_size, sizeof(uint32_t));
        uint64_t blob_len = 0;
        if (gguf_meta_u32_array(&g, "pap.vocab_len", vlen, (uint64_t)c->vocab_size)) {
            for (int i = 0; i < c->vocab_size; i++) blob_len += vlen[i];
            uint8_t *blob = malloc((size_t)blob_len);
            if (gguf_meta_u8_array(&g, "pap.vocab_blob", blob, blob_len)) {
                unsigned char **vbytes = calloc((size_t)c->vocab_size, sizeof(unsigned char *));
                int *bytes = calloc((size_t)c->vocab_size, sizeof(int));
                size_t off = 0;
                for (int i = 0; i < c->vocab_size; i++) {
                    vbytes[i] = blob + off;
                    bytes[i] = (int)vlen[i];
                    off += vlen[i];
                }
                uint32_t *ma = calloc((size_t)(c->num_merges > 0 ? c->num_merges : 1), sizeof(uint32_t));
                uint32_t *mb = calloc((size_t)(c->num_merges > 0 ? c->num_merges : 1), sizeof(uint32_t));
                if (c->num_merges > 0) {
                    gguf_meta_u32_array(&g, "pap.merges_a", ma, (uint64_t)c->num_merges);
                    gguf_meta_u32_array(&g, "pap.merges_b", mb, (uint64_t)c->num_merges);
                }
                tokenizer_build(&m->tok, c->vocab_size, (const unsigned char **)vbytes, bytes,
                                c->num_merges, ma, mb);
                free(vbytes); free(bytes); free(ma); free(mb);
            }
            free(blob);
        }
        free(vlen);
    }

    gguf_close(&g);
    m->is_loaded = 1;
    return 0;
}

void model_free(Model *m) {
    if (!m) return;
    int L = m->config.n_layer;
    tokenizer_free(&m->tok);
    free(m->wte);
    free(m->wpe);
    for (int l = 0; l < L; l++) {
        free(m->l_norm1[l]);
        free(m->wq[l]);
        free(m->wk[l]);
        free(m->wv[l]);
        free(m->wo[l]);
        free(m->l_norm2[l]);
        free(m->router[l]);
        if (m->exp_w1[l]) {
            for (int e = 0; e < m->config.moe_n_experts; e++) {
                free(m->exp_w1[l][e]);
                free(m->exp_w2[l][e]);
            }
        }
        free(m->exp_w1[l]);
        free(m->exp_w2[l]);
        free(m->ffn_w1[l]);
        free(m->ffn_w2[l]);
    }
    free(m->l_norm1);
    free(m->wq);
    free(m->wk);
    free(m->wv);
    free(m->wo);
    free(m->l_norm2);
    free(m->router);
    free(m->exp_w1);
    free(m->exp_w2);
    free(m->ffn_w1);
    free(m->ffn_w2);
    free(m->norm_final);
    free(m->lm_head);
    config_free(&m->config);
    memset(m, 0, sizeof(*m));
}

int64_t model_param_count(const Model *m) {
    const ModelConfig *c = &m->config;
    int64_t n = 0;
    n += (int64_t)c->vocab_size * c->d_model;      /* wte */
    n += (int64_t)c->max_seq_len * c->d_model;     /* wpe */
    for (int l = 0; l < c->n_layer; l++) {
        n += c->d_model;                            /* norm1 */
        n += 4LL * c->d_model * c->d_model;         /* qkv+o */
        n += c->d_model;                            /* norm2 */
        if (c->moe_mask[l]) {
            n += (int64_t)c->d_model * c->moe_n_experts;   /* router */
            n += 2LL * c->moe_n_experts * c->d_model * c->d_expert;
        } else {
            n += 2LL * c->d_model * c->d_ff;
        }
    }
    n += c->d_model;                                /* norm_final */
    n += (int64_t)c->vocab_size * c->d_model;       /* lm_head */
    return n;
}
