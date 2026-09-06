#include "server/api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <jansson.h>
#include <microhttpd.h>

#include "infer/generate.h"
#include "utils/logger.h"

/* 简单监控计数（单线程 demo 用） */
static unsigned long g_requests = 0;
static unsigned long g_tokens = 0;

/* ---- 请求上下文 ---- */
typedef struct {
    char *body;
    size_t len;
} ConnInfo;

/* ---- JSON 字符串转义（写入 dst） ---- */
static size_t json_escape(const char *src, char *dst, size_t cap) {
    size_t w = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p && w + 8 < cap; p++) {
        unsigned char c = *p;
        if (c == '"') { dst[w++] = '\\'; dst[w++] = '"'; }
        else if (c == '\\') { dst[w++] = '\\'; dst[w++] = '\\'; }
        else if (c == '\n') { dst[w++] = '\\'; dst[w++] = 'n'; }
        else if (c == '\r') { dst[w++] = '\\'; dst[w++] = 'r'; }
        else if (c == '\t') { dst[w++] = '\\'; dst[w++] = 't'; }
        else if (c < 0x20) { dst[w++] = ' '; }
        else { dst[w++] = (char)c; }
    }
    dst[w] = '\0';
    return w;
}

int moe_parse_generate_request(const char *json, char *prompt, size_t prompt_cap,
                               int *max_new, float *temperature, int *top_k) {
    json_error_t err;
    json_t *root = json_loads(json, 0, &err);
    if (!root || !json_is_object(root)) {
        if (root) json_decref(root);
        return -1;
    }
    int rc = 0;
    json_t *p = json_object_get(root, "prompt");
    json_t *opts = json_object_get(root, "options");
    if (!p || !json_is_string(p)) { rc = -2; goto done; }
    const char *s = json_string_value(p);
    size_t n = strlen(s);
    if (n >= prompt_cap) n = prompt_cap - 1;
    memcpy(prompt, s, n);
    prompt[n] = '\0';

    if (opts && json_is_object(opts)) {
        json_t *mt = json_object_get(opts, "max_tokens");
        if (mt && json_is_integer(mt)) *max_new = (int)json_integer_value(mt);
        json_t *tmp = json_object_get(opts, "temperature");
        if (tmp && json_is_real(tmp)) *temperature = (float)json_real_value(tmp);
        json_t *tk = json_object_get(opts, "top_k");
        if (tk && json_is_integer(tk)) *top_k = (int)json_integer_value(tk);
    }
done:
    json_decref(root);
    return rc;
}

/* ---- SSE 回调：逐个 token 拉取并写入 SSE 行（返回写入字节数 / 结束标记） ---- */
static ssize_t sse_cb(void *cls, uint64_t pos, char *buf, size_t max) {
    (void)pos;
    GenStream *gs = (GenStream *)cls;
    size_t w = 0;
    char tok[64];
    char esc[256];

    while (w + 96 < max) {
        int r = gen_stream_next(gs, tok, sizeof(tok));
        if (r <= 0) {
            /* 先冲掉已缓冲数据；否则结束流 */
            return (ssize_t)(w > 0 ? w : MHD_CONTENT_READER_END_OF_STREAM);
        }
        g_tokens++;                       /* 监控：累计生成 token */
        json_escape(tok, esc, sizeof(esc));
        int n = snprintf(buf + w, max - w, "data: {\"response\":\"%s\"}\n\n", esc);
        w += (size_t)n;
    }
    return (ssize_t)w;
}

static void sse_free(void *cls) {
    GenStream *gs = (GenStream *)cls;
    if (gs) {
        gen_stream_end(gs);
        free(gs);
    }
}

/* ---- 连接完成时释放 ConnInfo ---- */
static void completion_cb(void *cls, struct MHD_Connection *conn, void **con_cls,
                          enum MHD_RequestTerminationCode code) {
    (void)cls; (void)conn; (void)code;
    ConnInfo *ci = *con_cls;
    if (ci) {
        free(ci->body);
        free(ci);
        *con_cls = NULL;
    }
}

static char *read_body(ConnInfo *ci, const char *upload, size_t *upload_size) {
    if (*upload_size == 0) return NULL;
    ci->body = realloc(ci->body, ci->len + *upload_size + 1);
    memcpy(ci->body + ci->len, upload, *upload_size);
    ci->len += *upload_size;
    ci->body[ci->len] = '\0';
    size_t n = *upload_size;
    *upload_size = 0;
    (void)n;
    return ci->body;
}

static enum MHD_Result send_text(struct MHD_Connection *conn, unsigned status, const char *ct, const char *body) {
    struct MHD_Response *resp = MHD_create_response_from_buffer(strlen(body), (void *)body, MHD_RESPMEM_MUST_COPY);
    if (!resp) return MHD_NO;
    MHD_add_response_header(resp, "Content-Type", ct);
    enum MHD_Result ret = MHD_queue_response(conn, status, resp);
    MHD_destroy_response(resp);
    return ret;
}

static enum MHD_Result access_handler(void *cls, struct MHD_Connection *conn,
                                      const char *url, const char *method,
                                      const char *version, const char *upload_data,
                                      size_t *upload_data_size, void **con_cls) {
    (void)version;
    const Model *m = (const Model *)cls;
    ConnInfo *ci = (ConnInfo *)*con_cls;
    if (!ci) {
        ci = calloc(1, sizeof(ConnInfo));
        *con_cls = ci;
        return MHD_YES;
    }

    /* 读取 POST 请求体 */
    if (strcmp(method, MHD_HTTP_METHOD_POST) == 0) {
        if (*upload_data_size) {
            read_body(ci, upload_data, upload_data_size);
            return MHD_YES;
        }
    }

    /* GET /health */
    if (strcmp(method, MHD_HTTP_METHOD_GET) == 0 && strcmp(url, "/health") == 0)
        return send_text(conn, MHD_HTTP_OK, "text/plain", "ok");

    /* GET /api/models */
    if (strcmp(method, MHD_HTTP_METHOD_GET) == 0 && strcmp(url, "/api/models") == 0) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "{\"model\":\"moe-0.1b\",\"params\":%lld,\"arch\":\"moe-pap\",\"n_experts\":%d}",
                 (long long)model_param_count(m), m->config.moe_n_experts);
        return send_text(conn, MHD_HTTP_OK, "application/json", buf);
    }

    /* GET /api/metrics */
    if (strcmp(method, MHD_HTTP_METHOD_GET) == 0 && strcmp(url, "/api/metrics") == 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "{\"requests\":%lu,\"tokens\":%lu}", g_requests, g_tokens);
        return send_text(conn, MHD_HTTP_OK, "application/json", buf);
    }

    /* POST /api/generate (SSE) */
    if (strcmp(method, MHD_HTTP_METHOD_POST) == 0 && strcmp(url, "/api/generate") == 0) {
        g_requests++;
        if (!ci->body) return send_text(conn, MHD_HTTP_BAD_REQUEST, "text/plain", "empty body");
        char prompt[2048];
        int max_new = 100, top_k = 40;
        float temperature = 0.8f;
        int rc = moe_parse_generate_request(ci->body, prompt, sizeof(prompt),
                                            &max_new, &temperature, &top_k);
        if (rc != 0) return send_text(conn, MHD_HTTP_BAD_REQUEST, "text/plain", "bad json");

        GenStream *gs = calloc(1, sizeof(GenStream));
        if (gen_stream_begin(gs, m, prompt, max_new, temperature, top_k) < 0) {
            free(gs);
            return send_text(conn, MHD_HTTP_INTERNAL_SERVER_ERROR, "text/plain", "gen init fail");
        }
        struct MHD_Response *resp = MHD_create_response_from_callback(MHD_SIZE_UNKNOWN, 4096, &sse_cb, gs, &sse_free);
        if (!resp) { gen_stream_end(gs); free(gs); return MHD_NO; }
        MHD_add_response_header(resp, "Content-Type", "text/event-stream");
        MHD_add_response_header(resp, "Cache-Control", "no-cache");
        enum MHD_Result ret = MHD_queue_response(conn, MHD_HTTP_OK, resp);
        MHD_destroy_response(resp);
        return ret;
    }

    return send_text(conn, MHD_HTTP_NOT_FOUND, "text/plain", "not found");
}

int moe_server_run(const Model *m, const char *model_name, unsigned port) {
    (void)model_name;
    struct MHD_Daemon *daemon = MHD_start_daemon(
        MHD_USE_THREAD_PER_CONNECTION,
        (unsigned short)port, NULL, NULL, &access_handler, (void *)m,
        MHD_OPTION_NOTIFY_COMPLETED, &completion_cb, NULL,
        MHD_OPTION_END);
    if (!daemon) {
        MO_LOGE("HTTP 服务启动失败: %u", port);
        return 1;
    }
    MO_LOGI("pmai HTTP 已启动: http://127.0.0.1:%u  模型=%s", port, model_name);
    MO_LOGI("  GET  /health");
    MO_LOGI("  GET  /api/models");
    MO_LOGI("  GET  /api/metrics");
    MO_LOGI("  POST /api/generate  (SSE)   Ctrl-C 退出");
    for (;;) sleep(3600);   /* 阻塞直到 Ctrl-C 终止进程 */
    return 0;
}
