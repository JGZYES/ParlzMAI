#define _GNU_SOURCE
#include "model/llama_tokenizer_spm.h"

#include <stdlib.h>
#include <string.h>
#include "utils/portable.h"

/* ---- GPT-2 byte_to_unicode（与 llama_tokenizer.c 相同）---- */
static void build_b2u(char b2u[256][5], int code2byte[512]) {
    int bs[256], nb = 0;
    for (int i = '!'; i <= '~'; i++) bs[nb++] = i;
    for (int i = 0xA1; i <= 0xAC; i++) bs[nb++] = i;
    for (int i = 0xAE; i <= 0xFF; i++) bs[nb++] = i;
    for (int i = 0; i < 512; i++) code2byte[i] = -1;
    int byte2code[256];
    for (int b = 0; b < 256; b++) byte2code[b] = -1;
    for (int i = 0; i < nb; i++) byte2code[bs[i]] = bs[i];
    int n = 0;
    for (int b = 0; b < 256; b++) if (byte2code[b] < 0) byte2code[b] = 256 + n++;
    for (int b = 0; b < 256; b++) {
        int code = byte2code[b];
        code2byte[code] = b;
        if (code < 0x80) { b2u[b][0] = (char)code; b2u[b][1] = 0; }
        else if (code < 0x800) {
            b2u[b][0] = (char)(0xC0 | (code >> 6));
            b2u[b][1] = (char)(0x80 | (code & 0x3F));
            b2u[b][2] = 0;
        } else {
            b2u[b][0] = (char)(0xE0 | (code >> 12));
            b2u[b][1] = (char)(0x80 | ((code >> 6) & 0x3F));
            b2u[b][2] = (char)(0x80 | (code & 0x3F));
            b2u[b][3] = 0;
        }
    }
}

static int cmp_idx(const void *a, const void *b, void *arg) {
    char **tokens = (char **)arg;
    const int *ia = (const int *)a, *ib = (const int *)b;
    return strcmp(tokens[*ia], tokens[*ib]);
}
static int lookup(const LlamaTokenizerSPM *t, const char *s, int len) {
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const char *tk = t->tokens[t->sort_idx[mid]];
        int c = strncmp(tk, s, (size_t)len);
        if (c == 0 && tk[len] == '\0') return t->sort_idx[mid];
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

int llmtok_spm_init(LlamaTokenizerSPM *t, char **tokens, int vocab_size, float *scores) {
    memset(t, 0, sizeof(*t));
    t->vocab_size = vocab_size;
    t->tokens = calloc((size_t)vocab_size, sizeof(char *));
    for (int i = 0; i < vocab_size; i++) {
        size_t n = strlen(tokens[i]);
        t->tokens[i] = malloc(n + 1);
        memcpy(t->tokens[i], tokens[i], n + 1);
    }
    t->sort_idx = malloc((size_t)vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) t->sort_idx[i] = i;
    mo_qsort_r(t->sort_idx, (size_t)vocab_size, sizeof(int), cmp_idx, t->tokens);
    t->scores = malloc((size_t)vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) t->scores[i] = scores ? scores[i] : 0.0f;
    return 0;
}
void llmtok_spm_free(LlamaTokenizerSPM *t) {
    if (!t) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->tokens[i]);
    free(t->tokens);
    free(t->sort_idx);
    free(t->scores);
    memset(t, 0, sizeof(*t));
}

/* 可调用的 symbol: [start, len) 字符串 */
typedef struct { const char *s; int len; } Sym;

#define SPACE "\xE2\x96\x81"  /* U+2581 */

/* 预计算目标文本（" "+text 中空格->▁），返回 UTF-8 字符数组 */
static int preprocess(const char *text, char **outbuf, Sym **symarr, int *nchars) {
    size_t tlen = strlen(text);
    /* 结果 = " " + text，其中每个 ' ' (0x20) 换成 SPACE(3字节) */
    size_t cap = 2 + tlen * 3 + 8;
    char *buf = malloc(cap);
    size_t w = 0;
    memcpy(buf + w, SPACE, 3); w += 3;   /* 前导空格 -> ▁（SentencePiece 伪空格） */
    for (size_t i = 0; i < tlen; i++) {
        if (text[i] == ' ') { memcpy(buf + w, SPACE, 3); w += 3; }
        else buf[w++] = text[i];
    }
    buf[w] = '\0';
    /* 按 UTF-8 字符切分 */
    int n = 0;
    Sym *arr = malloc(((w / 1) + 8) * sizeof(Sym));
    for (size_t i = 0; i < w;) {
        unsigned char c = (unsigned char)buf[i];
        int l = (c < 0x80) ? 1 : ((c & 0xE0) == 0xC0 ? 2 : ((c & 0xF0) == 0xE0 ? 3 : 4));
        arr[n].s = buf + i; arr[n].len = l;
        n++; i += l;
    }
    *outbuf = buf; *symarr = arr; *nchars = n;
    return 0;
}

int llmtok_spm_encode(const LlamaTokenizerSPM *t, const char *text, int *out, int max) {
    char *buf; Sym *arr; int n;
    preprocess(text, &buf, &arr, &n);
    if (n == 0) { free(buf); free(arr); return 0; }

    /* BPE：反复找相邻拼接在词表中且 score 最高者，合并 */
    while (1) {
        int best = -1; float bestsc = -1e30f; char bestbuf[2048]; int bestlen = 0; int bestid = -1;
        for (int i = 0; i + 1 < n; i++) {
            int L = arr[i].len + arr[i + 1].len;
            if (L >= (int)sizeof(bestbuf)) continue;
            memcpy(bestbuf, arr[i].s, (size_t)arr[i].len);
            memcpy(bestbuf + arr[i].len, arr[i + 1].s, (size_t)arr[i + 1].len);
            bestbuf[L] = '\0';
            int id = lookup(t, bestbuf, L);
            if (id >= 0 && t->scores[id] > bestsc) { bestsc = t->scores[id]; best = i; bestlen = L; bestid = id; }
        }
        if (best < 0) break;
        arr[best].s = t->tokens[bestid]; arr[best].len = bestlen;
        memmove(&arr[best + 1], &arr[best + 2], (size_t)(n - best - 2) * sizeof(Sym));
        n--;
    }

    /* 映射 / 字节回退 */
    char b2u[256][5]; int code2byte[512];
    build_b2u(b2u, code2byte);
    int cnt = 0;
    for (int i = 0; i < n && cnt < max; i++) {
        int id = lookup(t, arr[i].s, arr[i].len);
        if (id >= 0) { out[cnt++] = id; continue; }
        /* 逐字节回退：byte -> b2u 字符 -> token id */
        for (int j = 0; j < arr[i].len; j++) {
            int b = (unsigned char)arr[i].s[j];
            const char *u = b2u[b];
            int id2 = lookup(t, u, (int)strlen(u));
            out[cnt++] = id2 >= 0 ? id2 : 0;
        }
    }
    free(buf); free(arr);
    return cnt;
}

int llmtok_spm_decode(const LlamaTokenizerSPM *t, const int *ids, int n, char *out, int max) {
    char b2u[256][5]; int code2byte[512];
    build_b2u(b2u, code2byte);
    int w = 0;
    for (int i = 0; i < n && w < max; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size) continue;
        const char *s = t->tokens[id];
        while (*s && w < max) {
            unsigned char c = (unsigned char)*s;
            /* ▁ (U+2581, UTF-8 E2 96 81) -> 空格（token 内部也有，不仅仅是纯 ▁） */
            if (c == 0xE2 && (unsigned char)s[1] == 0x96 && (unsigned char)s[2] == 0x81) {
                out[w++] = ' '; s += 3; continue;
            }
            int l = (c < 0x80) ? 1 : ((c & 0xE0) == 0xC0 ? 2 : ((c & 0xF0) == 0xE0 ? 3 : 4));
            if (l == 1) { out[w++] = (char)c; s += 1; continue; }
            /* 多字节：byte-proxy（b2u 代理一个原始字节）-> 输出该字节；否则原样输出原 UTF-8 字节 */
            int cp = (l == 2) ? (((c & 0x1F) << 6) | (s[1] & 0x3F))
                  : (l == 3) ? (((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F))
                  : (((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F));
            if (cp >= 0 && cp < 512 && code2byte[cp] >= 0) out[w++] = (char)code2byte[cp];
            else { for (int k = 0; k < l; k++) out[w++] = s[k]; }
            s += l;
        }
    }
    if (w < max) out[w] = '\0'; else out[max - 1] = '\0';
    return w;
}
