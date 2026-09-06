#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model/gguf.h"
#include "model/llama_tokenizer.h"

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

    llmtok_free(&t);
    gguf_close(&g);
    return 0;
}
