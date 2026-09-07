#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model/gguf.h"
#include "model/llama_tokenizer.h"

static const char *type_name_c(uint32_t t);

/* ---- 逐字照搬的 llama.cpp Q6_K 反量化（权威参考，验证公式一致） ---- */
#define QK_K 256
typedef uint16_t ggml_half;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K;
static float f16f(uint16_t h) {
    uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,b; float f;
    if(e==0x1f){b=(s<<31)|0x7f800000|(m<<13);memcpy(&f,&b,4);return f;}
    if(e==0){ if(m==0){b=s<<31;memcpy(&f,&b,4);return f;} f=(float)m*5.9604644775390625e-8f; return s?-f:f;}
    b=(s<<31)|((e-15+127)<<23)|(m<<13); memcpy(&f,&b,4); return f;
}
static void dq6k_ref(const block_q6_K *x, float *y) {
    const float d = f16f(x->d);
    const uint8_t *ql = x->ql, *qh = x->qh;
    const int8_t *sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l/16;
            int8_t q1=(int8_t)((ql[l+0]&0xF)|(((qh[l]>>0)&3)<<4))-32;
            int8_t q2=(int8_t)((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
            int8_t q3=(int8_t)((ql[l+0]>>4)|(((qh[l]>>4)&3)<<4))-32;
            int8_t q4=(int8_t)((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
            y[l+0]=d*sc[is+0]*q1; y[l+32]=d*sc[is+2]*q2;
            y[l+64]=d*sc[is+4]*q3; y[l+96]=d*sc[is+6]*q4;
        }
        y+=128; ql+=64; qh+=32; sc+=8;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: test_llmtok <model.gguf> [text...]\n"); return 1; }
    char text[4096] = "Hello, world! Today is a good day.";
    if (argc > 2) { text[0] = 0; for (int i = 2; i < argc; i++) { strcat(text, argv[i]); if (i + 1 < argc) strcat(text, " "); } }

    Gguf g;
    if (gguf_open(&g, argv[1]) != 0) { fprintf(stderr, "gguf open fail\n"); return 2; }
    char **toks = NULL; int ntok = 0;
    if (!gguf_meta_string_array(&g, "tokenizer.ggml.tokens", &toks, &ntok)) { fprintf(stderr, "no tokens meta\n"); return 3; }
    char **mrg = NULL; int nmrg = 0;
    gguf_meta_string_array(&g, "tokenizer.ggml.merges", &mrg, &nmrg);

    LlamaTokenizer t;
    if (llmtok_init(&t, toks, ntok, mrg, nmrg) != 0) { fprintf(stderr, "tok init fail\n"); return 4; }
    gguf_free_string_array(toks, ntok);
    gguf_free_string_array(mrg, nmrg);

    int ids[512];
    int n = llmtok_encode(&t, text, ids, 512);
    printf("text: %s\nids:  ", text);
    for (int i = 0; i < n; i++) printf("%d ", ids[i]);
    printf("\n");
    char buf[4096];
    int b = llmtok_decode(&t, ids, n, buf, sizeof(buf));
    printf("round-trip: %.*s\n", b < 0 ? 0 : b, buf);

    /* 反量化：第一个 Q4_K/Q6_K 张量，Q6_K 附逐字参考对比（公式一致性验证） */
    for (uint64_t ti = 0; ti < g.tensor_count; ti++) {
        if (g.ggml_type[ti] == 12 || g.ggml_type[ti] == 14) {
            uint64_t nn = 1;
            for (uint32_t d = 0; d < g.n_dims[ti]; d++) nn *= g.dims[ti][d];
            float *fv = malloc((size_t)nn * sizeof(float));
            if (gguf_tensor_to_f32(&g, g.tensor_name[ti], fv) > 0) {
                int nan = 0; float mn = 1e30f, mx = -1e30f;
                for (uint64_t j = 0; j < nn && j < 4096; j++) {
                    if (fv[j] != fv[j]) nan++;
                    if (fv[j] < mn) mn = fv[j]; if (fv[j] > mx) mx = fv[j];
                }
                fprintf(stderr, "[dequant] %s (%s) NaN=%d range=[%.3f, %.3f]\n",
                        g.tensor_name[ti], type_name_c(g.ggml_type[ti]), nan, mn, mx);
                if (g.ggml_type[ti] == 14) {  /* Q6_K 参考对比（首位差异=0 表示与 llama.cpp 一致） */
                    const unsigned char *tp = g.data + g.data_offset + (size_t)g.offset[ti];
                    float refrow[1536];
                    for (int blk = 0; blk < 6; blk++) dq6k_ref((const block_q6_K *)(tp + blk*210), refrow + blk*256);
                    int diff = -1;
                    for (int j = 0; j < 1536; j++) if (fabsf(refrow[j] - fv[j]) > 0.01f) { diff = j; break; }
                    fprintf(stderr, "[ref] Q6_K vs llama.cpp 逐字参考: 首个差异@%d (0=公式一致)\n", diff);
                }
            }
            free(fv);
            break;
        }
    }

    llmtok_free(&t);
    gguf_close(&g);
    return 0;
}

static const char *type_name_c(uint32_t t) {
    switch (t) { case 12: return "Q4_K"; case 14: return "Q6_K"; default: return "?"; }
}
