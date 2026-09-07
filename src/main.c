#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>

#include "infer/generate.h"
#include "infer/sample.h"
#include "model/gguf.h"
#include "model/model.h"
#include "model/llama.h"
#ifdef MOE_HTTP
#include "server/api.h"
#endif
#include "utils/logger.h"

static void usage(void) {
    printf("用法: pmai --model <file.pap|file.gguf> [选项]\n"
           "  --model <path>      模型文件（必填；.gguf 自动识别）\n"
           "  --inspect <path>    查看任意 GGUF 的元数据/张量表\n"
           "  --prompt <text>     提示词（默认: \"The\"）\n"
           "  --n_tokens <N>      生成 token 数（默认 100）\n"
           "  --temperature <T>   采样温度（默认 0.8，<=0 为贪心）\n"
           "  --top_k <K>         top-k 采样（默认 40，<=0 不限制）\n"
           "  --seed <S>          随机种子\n"
           "  --interactive        交互式对话（输入一行→生成→继续，exit/quit 退出）\n"
           "  --chat               聊天模式（User:/Assistant: 格式，带上下文，存记录）\n"
           "  --output <file>      把生成文本写到文件（默认在 output/ 目录）\n"
           "  --serve             以 HTTP 服务方式运行（需 --model）\n"
           "  --port <P>          HTTP 端口（默认 11434）\n"
           "  --log-level <lvl>   debug/info/warn/error\n");
}

static int has_gguf_magic(const char *path) {
    const char m[4] = {'G', 'G', 'U', 'F'};
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char b[4];
    int n = (int)fread(b, 1, 4, f);
    fclose(f);
    return (n == 4 && memcmp(b, m, 4) == 0);
}

/* 判断是否为 llama 架构 GGUF（存在 llama.block_count 元数据；.pap 导出的是 pap.* 前缀） */
static int is_llama_gguf(const char *path) {
    Gguf g;
    if (gguf_open(&g, path) != 0) return 0;
    uint32_t bc = 0;
    int r = gguf_meta_u32(&g, "llama.block_count", &bc);
    gguf_close(&g);
    return r;
}

/* llama：编码提示（开头补 BOS）→ 生成 → 解码输出到 stdout */
static void llama_emit(LlamaModel *m, const char *prompt, int n_new, float temp, int top_k) {
    int ids[512];
    int n = 0;
    if (m->c.bos_id >= 0) ids[n++] = m->c.bos_id;
    if (prompt && *prompt) {
        int k = llama_tokenize(m, prompt, ids + n, 512 - n);
        if (k > 0) n += k;
    }
    if (n == 0) ids[n++] = (m->c.bos_id >= 0) ? m->c.bos_id : 1;
    int out[512];
    int p = llama_generate(m, ids, n, n_new, temp, top_k, out);
    char buf[4096];
    int bl = llama_detokenize(m, out, p, buf, sizeof(buf));
    fwrite(buf, 1, (size_t)(bl < 0 ? 0 : bl), stdout);
    printf("\n");
}

static int arg_i(const char *s, int def) { return s ? atoi(s) : def; }
static float arg_f(const char *s, float def) { return s ? (float)atof(s) : def; }

static int write_text_file(const char *path, const char *content) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(content, 1, strlen(content), f);
    fclose(f);
    return 0;
}

/* 从简单 key: value 的 config 文件加载默认值（host/port/model/n_tokens/temperature/top_k） */
static void load_config(const char *path, unsigned *port, int *n_tokens, float *temperature,
                        int *top_k, char *model, size_t model_cap) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char *c = strchr(line, '#');
        if (c) *c = '\0';
        if ((line[0] == '\n' || line[0] == '\r') && !line[1]) continue;
        char key[160] = {0}, val[384] = {0};
        if (sscanf(line, "%159[^:]: %383[^\n]", key, val) != 2) continue;
        for (char *p = key; *p; p++) if (*p == ' ' || *p == '\t') { *p = '\0'; break; }
        if (!strcmp(key, "port")) *port = (unsigned)atoi(val);
        else if (!strcmp(key, "model")) snprintf(model, model_cap, "%s", val);
        else if (!strcmp(key, "n_tokens")) *n_tokens = atoi(val);
        else if (!strcmp(key, "temperature")) *temperature = (float)atof(val);
        else if (!strcmp(key, "top_k")) *top_k = atoi(val);
    }
    fclose(f);
}

/* 聊天式生成：从 prompt 前缀开始生成，遇到下一个 "\nUser:"（下一轮用户话）停止，返回助手回复。调用方 free() */
static char *run_chat(Model *m, const char *prompt, int max_new, float temperature, int top_k) {
    GenStream gs;
    if (gen_stream_begin(&gs, m, prompt, max_new, temperature, top_k) < 0) return strdup("");
    size_t cap = (size_t)max_new * 16 + 256;
    char *buf = malloc(cap);
    size_t w = 0;
    buf[0] = '\0';
    char tok[64];
    while (gen_stream_next(&gs, tok, sizeof(tok)) == 1) {
        size_t tl = strlen(tok);
        if (w + tl + 1 > cap) {
            cap = cap * 2 + tl + 64;
            buf = realloc(buf, cap);
        }
        memcpy(buf + w, tok, tl);
        w += tl;
        buf[w] = '\0';
        if (strstr(buf, "\nUser:")) break;   /* 已生成到下一轮用户话，停止 */
    }
    gen_stream_end(&gs);
    char *np = strstr(buf, "\nUser:");
    if (np) *np = '\0';                       /* 截掉后面的下一轮 */
    return buf;
}

int main(int argc, char **argv) {
    const char *model = NULL, *prompt = NULL, *loglvl = NULL, *inspect = NULL;
    int n_tokens = 100;
    float temperature = 0.8f;
    int top_k = 40;
    int seed = 0;
    int serve = 0;
    int interactive = 0;
    int chat = 0;
    unsigned port = 11434;
    const char *output = NULL;
    const char *config = NULL;
    char config_model[512] = "";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--inspect") && i + 1 < argc) inspect = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--n_tokens") && i + 1 < argc) n_tokens = arg_i(argv[++i], 100);
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc) temperature = arg_f(argv[++i], 0.8f);
        else if (!strcmp(argv[i], "--top_k") && i + 1 < argc) top_k = arg_i(argv[++i], 40);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = arg_i(argv[++i], 0);
        else if (!strcmp(argv[i], "--interactive")) interactive = 1;
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) output = argv[++i];
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--serve")) serve = 1;
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = (unsigned)arg_i(argv[++i], 11434);
        else if (!strcmp(argv[i], "--log-level") && i + 1 < argc) loglvl = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(); return 0; }
    }

    /* 便利：未用 --model 时，把第一个非 '-' 位置参数当作模型路径 */
    if (config) {
        load_config(config, &port, &n_tokens, &temperature, &top_k, config_model, sizeof(config_model));
        if (!model && config_model[0]) model = config_model;
    }
    if (!model && argc > 1 && argv[1][0] != '-') model = argv[1];

    if (loglvl) {
        if (!strcasecmp(loglvl, "debug")) mo_log_set_level(MO_LOG_DEBUG);
        else if (!strcasecmp(loglvl, "info")) mo_log_set_level(MO_LOG_INFO);
        else if (!strcasecmp(loglvl, "warn")) mo_log_set_level(MO_LOG_WARN);
        else if (!strcasecmp(loglvl, "error")) mo_log_set_level(MO_LOG_ERROR);
    }

    if (inspect) {
        Gguf g;
        if (gguf_open(&g, inspect) != 0) { MO_LOGE("GGUF 打开失败: %s", inspect); return 2; }
        gguf_inspect(&g, inspect, stdout);
        gguf_close(&g);
        return 0;
    }

    if (!model) { usage(); return 1; }

    /* llama GGUF：由 pmai 直接跑（不再需要 pmai-llama） */
    if (has_gguf_magic(model) && is_llama_gguf(model)) {
        LlamaModel lm;
        if (llama_load(&lm, model) != 0) { MO_LOGE("加载 llama 模型失败: %s", model); return 2; }
        MO_LOGI("已加载 llama 模型: %s", model);
        MO_LOGI("  config: vocab=%d layers=%d n_embd=%d n_head=%d(kv=%d) ffn=%d max_seq=%d",
                lm.c.vocab, lm.c.n_layer, lm.c.n_embd, lm.c.n_head, lm.c.n_head_kv,
                lm.c.ffn_dim, lm.c.max_seq);
        MO_LOGI("  params: %lld", (long long)llama_param_count(&lm));
        if (!prompt) prompt = "The";
        if (serve) {
#ifdef MOE_HTTP
            MO_LOGE("llama 模型的 HTTP 服务暂未接入，请先使用单次/交互模式");
#endif
            llama_free(&lm);
            return 2;
        }
        if (interactive) {
            char line[4096];
            for (;;) {
                printf("\n>> ");
                fflush(stdout);
                if (!fgets(line, sizeof(line), stdin)) break;
                size_t len = strlen(line);
                while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
                if (!len) continue;
                if (!strcmp(line, "exit") || !strcmp(line, "quit")) break;
                llama_emit(&lm, line, n_tokens, temperature, top_k);
            }
        } else {
            llama_emit(&lm, prompt, n_tokens, temperature, top_k);
        }
        llama_free(&lm);
        return 0;
    }

    mo_rng_seed((unsigned)seed);

    Model m;
    int rc = has_gguf_magic(model) ? model_load_gguf(&m, model) : model_load(&m, model);
    if (rc != 0) {
        MO_LOGE("加载模型失败(%d): %s", rc, model);
        return 2;
    }
    ModelConfig *c = &m.config;
    MO_LOGI("已加载模型: %s", model);
    MO_LOGI("  config: vocab=%d layers=%d d_model=%d heads=%d max_seq=%d",
            c->vocab_size, c->n_layer, c->d_model, c->n_head, c->max_seq_len);
    MO_LOGI("  MoE: experts=%d top_k=%d d_expert=%d",
            c->moe_n_experts, c->moe_top_k, c->d_expert);
    MO_LOGI("  params: %lld", (long long)model_param_count(&m));

    if (!prompt) prompt = "The";
    MO_LOGI("生成中: n_tokens=%d temperature=%.2f top_k=%d", n_tokens, temperature, top_k);

    if (serve) {
#ifdef MOE_HTTP
        moe_server_run(&m, model, port);
#else
        MO_LOGE("未启用 HTTP 服务（缺 libmicrohttpd/jansson，用 -DBUILD_HTTP=ON 重建）");
        (void)port;
#endif
        model_free(&m);
        return 0;
    }

    if (chat) {
        mkdir("output", 0755);
        char outpath[512];
        if (output) snprintf(outpath, sizeof(outpath), "%s", output);
        else snprintf(outpath, sizeof(outpath), "output/chat-%ld.txt", (long)time(NULL));
        FILE *logf = fopen(outpath, "wb");
        if (logf) fprintf(logf, "# pmai 聊天记录\n\n");
        MO_LOGI("聊天模式已启动。输入内容即对话；exit/quit 退出。记录保存到 %s", outpath);

        size_t hcap = 4096, hlen = 0;
        char *hist = calloc(1, hcap);
        char line[4096];
        for (;;) {
            printf("\n你: ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            size_t len = strlen(line);
            while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
            if (!len) continue;
            if (!strcmp(line, "exit") || !strcmp(line, "quit")) break;

            /* prompt = history + "User: <line>\nAssistant:" */
            size_t need = hlen + len + 64;
            if (need > hcap) { while (need > hcap) hcap *= 2; hist = realloc(hist, hcap); }
            size_t p = (size_t)snprintf(hist + hlen, hcap - hlen, "User: %s\nAssistant:", line);
            hlen += p;

            char *reply = run_chat(&m, hist, n_tokens, temperature, top_k);
            if (!reply || !reply[0]) { free(reply); reply = strdup("(我还没想好，换句话试试？)"); }
            printf("%s\n", reply);

            /* 回填历史，供下一轮携带上下文；保留尾部避免超出上下文 */
            size_t adds = (size_t)snprintf(hist + hlen, hcap - hlen, "\nAssistant: %s\n", reply ? reply : "");
            hlen += adds;
            if (hlen > 2048) { size_t keep = 1536; memmove(hist, hist + hlen - keep, keep + 1); hlen = keep; }
            if (logf && reply) { fprintf(logf, "User: %s\nAssistant: %s\n\n", line, reply); fflush(logf); }
            free(reply);
        }
        free(hist);
        if (logf) fclose(logf);
        model_free(&m);
        return 0;
    }

    if (interactive) {
        mkdir("output", 0755);
        char outpath[512];
        if (output) snprintf(outpath, sizeof(outpath), "%s", output);
        else snprintf(outpath, sizeof(outpath), "output/chat-%ld.txt", (long)time(NULL));
        FILE *logf = fopen(outpath, "wb");
        if (logf) fprintf(logf, "# pmai 交互记录\n\n");
        MO_LOGI("交互模式已启动。输入一行后回车即生成；输入 exit/quit 退出。对话保存到 %s", outpath);
        char line[4096];
        for (;;) {
            printf("\n>> ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            size_t len = strlen(line);
            while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
            if (!len) continue;
            if (!strcmp(line, "exit") || !strcmp(line, "quit")) break;
            char *out = moe_generate_text(&m, line, n_tokens, temperature, top_k);
            printf("%s\n", out ? out : "");
            if (logf && out) { fprintf(logf, ">> %s\n%s\n\n", line, out); fflush(logf); }
            free(out);
        }
        if (logf) fclose(logf);
        model_free(&m);
        return 0;
    }

    if (output) {
        mkdir("output", 0755);
        char *out = moe_generate_text(&m, prompt, n_tokens, temperature, top_k);
        int rc = write_text_file(output, out ? out : "");
        MO_LOGI("生成已写入 %s (%s)", output, rc == 0 ? "ok" : "fail");
        printf("%s\n", out ? out : "");
        free(out);
        model_free(&m);
        return 0;
    }

    moe_generate(&m, prompt, n_tokens, temperature, top_k);

    model_free(&m);
    return 0;
}
