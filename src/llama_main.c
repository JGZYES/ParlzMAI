#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model/llama.h"
#include "utils/portable.h"

int main(int argc, char **argv) {
    mo_set_console_utf8();   /* 控制台按 UTF-8 显示，避免中文乱码 */
    if (argc < 2) {
        fprintf(stderr, "用法: pmai-llama <model.gguf> [n_new] [prompt 或 init_id ...]\n");
        return 1;
    }
    LlamaModel m;
    if (llama_load(&m, argv[1]) != 0) {
        fprintf(stderr, "加载失败: %s\n", argv[1]);
        return 2;
    }
    int n_new = argc > 2 ? atoi(argv[2]) : 10;
    int ids[512];
    int n = 0;
    int has_tok = m.is_spm ? (m.spm_tok.vocab_size > 0) : (m.tok.vocab_size > 0);

    /* --tok <text>：仅分词，打印 token id 后退出（用于验证分词器） */
    if (has_tok && argc >= 5 && strcmp(argv[3], "--tok") == 0) {
        n = llama_tokenize(&m, argv[4], ids, 512);
        for (int i = 0; i < n; i++) printf("%d", ids[i]), putchar(i + 1 < n ? ' ' : '\n');
        llama_free(&m);
        return 0;
    }

    /* 有分词器且第 3 个参数是字符串 -> 当作 prompt 编码 */
    if (has_tok && argc >= 4) {
        n = llama_tokenize(&m, argv[3], ids, 512);
        if (n <= 0) { ids[0] = 1; n = 1; }
    } else {
        for (int i = 3; i < argc && n < 512; i++) ids[n++] = atoi(argv[i]);
        if (n == 0) ids[n++] = 1;
    }

    int out[512];
    int p = llama_generate(&m, ids, n, n_new, 0.0f, 0, out);

    if (has_tok) {
        char buf[4096];
        int blen = llama_detokenize(&m, out, p, buf, sizeof(buf));
        fwrite(buf, 1, (size_t)(blen < 0 ? 0 : blen), stdout);
        printf("\n");
    } else {
        for (int i = 0; i < p; i++) printf("%d", out[i]), putchar(i + 1 < p ? ' ' : '\n');
    }
    llama_free(&m);
    return 0;
}
