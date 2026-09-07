#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model/gguf.h"
#include "model/llama_tokenizer.h"

static const char *type_name_c(uint32_t t);

/* 独立验证 C 端 GPT-2 byte-level BPE 分词器（不依赖完整模型加载/架构）。 */
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: test_llmtok <model.gguf> [text...]\n"); return 1; }
    /* 从第 2 个参数起拼接成测试文本（若无则用默认）。 */
    char text[4096] = "Hello, world! Today is a good day.";
    if (argc > 2) { text[0] = 0; for (int i = 2; i < argc; i++) { strcat(text, argv[i]); if (i + 1 < argc) strcat(text, " "); } }

    Gguf g;
    if (gguf_open(&g, argv[1]) != 0) { fprintf(stderr, "gguf open fail\n"); return 2; }
    char **toks = NULL; int ntok = 0;
    if (!gguf_meta_string_array(&g, "tokenizer.ggml.tokens", &toks, &ntok)) { fprintf(stderr, "no tokens meta\n"); return 3; }
    char **mrg = NULL; int nmrg = 0;
    gguf_meta_string_array(&g, "tokenizer.ggml.merges", &mrg, &nmrg);
    fprintf(stderr, "[info] vocab=%d merges=%d\n", ntok, nmrg);

    LlamaTokenizer t;
    if (llmtok_init(&t, toks, ntok, mrg, nmrg) != 0) { fprintf(stderr, "tok init fail\n"); return 4; }
    gguf_free_string_array(toks, ntok);
    gguf_free_string_array(mrg, nmrg);

    int ids[512];
    int n = llmtok_encode(&t, text, ids, 512);
    printf("text: %s\n", text);
    printf("ids:  ");
    for (int i = 0; i < n; i++) printf("%d ", ids[i]);
    printf("\n");

    char buf[4096];
    int b = llmtok_decode(&t, ids, n, buf, sizeof(buf));
    printf("round-trip: %.*s\n", b < 0 ? 0 : b, buf);

    /* 反量化检查：对第一个 Q4_K/Q6_K 张量做反量化，报告有限值统计 */
    for (uint64_t ti = 0; ti < g.tensor_count; ti++) {
        if (g.ggml_type[ti] == 12 || g.ggml_type[ti] == 14) {  /* Q4_K / Q6_K */
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
