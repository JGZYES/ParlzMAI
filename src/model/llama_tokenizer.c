#define _GNU_SOURCE
#include "model/llama_tokenizer.h"

#include <stdlib.h>
#include <string.h>

/* ---- GPT-2 byte_to_unicode 映射 ---- */

/* b2u_str[b]：byte b 映射成的 unicode 字符的 UTF-8 串；同时生成反向 code2byte[码点]=字节 */
static void build_b2u(char b2u[256][5], int code2byte[512]) {
    int bs[256], nb = 0;
    for (int i = '!'; i <= '~'; i++) bs[nb++] = i;
    for (int i = 0xA1; i <= 0xAC; i++) bs[nb++] = i;
    for (int i = 0xAE; i <= 0xFF; i++) bs[nb++] = i;

    for (int i = 0; i < 512; i++) code2byte[i] = -1;
    int byte2code[256];
    for (int b = 0; b < 256; b++) byte2code[b] = -1;
    for (int i = 0; i < nb; i++) byte2code[bs[i]] = bs[i];   /* 可打印区：字节映射到自身 */
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

/* 二分查找 token 字符串 */
static int lookup(const LlamaTokenizer *t, const char *s) {
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int c = strcmp(t->tokens[t->sort_idx[mid]], s);
        if (c == 0) return t->sort_idx[mid];
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

static int cmp_idx(const void *a, const void *b, void *arg) {
    char **tokens = (char **)arg;
    const int *ia = (const int *)a, *ib = (const int *)b;
    return strcmp(tokens[*ia], tokens[*ib]);
}

int llmtok_init(LlamaTokenizer *t, char **tokens, int vocab_size, char **merges, int num_merges) {
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
    qsort_r(t->sort_idx, (size_t)vocab_size, sizeof(int), cmp_idx, t->tokens);

    /* byte -> id */
    char b2u[256][5];
    int code2byte[512];
    build_b2u(b2u, code2byte);
    t->byte_to_id = malloc(256 * sizeof(int));
    for (int b = 0; b < 256; b++) t->byte_to_id[b] = lookup(t, b2u[b]);

    /* merges "a b" -> id 对 */
    t->num_merges = num_merges;
    t->ma = malloc((size_t)(num_merges > 0 ? num_merges : 1) * sizeof(int));
    t->mb = malloc((size_t)(num_merges > 0 ? num_merges : 1) * sizeof(int));
    for (int i = 0; i < num_merges; i++) {
        const char *s = merges[i];
        const char *sp = strchr(s, ' ');
        int la = sp ? (int)(sp - s) : (int)strlen(s);
        char a[512], b[512];
        memcpy(a, s, (size_t)la); a[la] = 0;
        strcpy(b, sp ? sp + 1 : "");
        t->ma[i] = lookup(t, a);
        t->mb[i] = lookup(t, b);
    }
    return 0;
}

void llmtok_free(LlamaTokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->tokens[i]);
    free(t->tokens);
    free(t->byte_to_id);
    free(t->sort_idx);
    free(t->ma);
    free(t->mb);
    memset(t, 0, sizeof(*t));
}

int llmtok_encode(const LlamaTokenizer *t, const char *text, int *out, int max) {
    size_t n = strlen(text);
    if (n == 0) return 0;
    int *ids = malloc((size_t)(n + 8) * sizeof(int));
    int len = 0;
    for (size_t i = 0; i < n; i++) ids[len++] = t->byte_to_id[(unsigned char)text[i]];

    while (1) {
        int best = -1, bestr = 0x7fffffff;
        for (int i = 0; i + 1 < len; i++) {
            int a = ids[i], b = ids[i + 1];
            for (int r = 0; r < t->num_merges; r++) {
                if (t->ma[r] == a && t->mb[r] == b && r < bestr) { bestr = r; best = i; break; }
            }
        }
        if (best < 0) break;
        int a = ids[best], b = ids[best + 1];
        int la = (int)strlen(t->tokens[a]), lb = (int)strlen(t->tokens[b]);
        char buf[2048];
        memcpy(buf, t->tokens[a], (size_t)la);
        memcpy(buf + la, t->tokens[b], (size_t)lb + 1);
        int merged = lookup(t, buf);
        if (merged < 0) break;
        ids[best] = merged;
        memmove(&ids[best + 1], &ids[best + 2], (size_t)(len - best - 2) * sizeof(int));
        len--;
    }

    if (len > max) { free(ids); return -1; }
    for (int i = 0; i < len; i++) out[i] = ids[i];
    free(ids);
    return len;
}

/* 把一个 UTF-8 字符解码成码点 */
static int u8_to_cp(const unsigned char *s, int *len) {
    unsigned char c = s[0];
    if (c < 0x80) { *len = 1; return c; }
    if ((c & 0xE0) == 0xC0) { *len = 2; return ((c & 0x1F) << 6) | (s[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0) { *len = 3; return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    *len = 1; return c;
}

int llmtok_decode(const LlamaTokenizer *t, const int *ids, int n, char *out, int maxcat) {
    /* 反 byte_to_unicode：码点 -> 字节 */
    char b2u[256][5];
    int code2byte[512];
    build_b2u(b2u, code2byte);
    int w = 0;
    for (int i = 0; i < n && w < maxcat; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size) continue;
        const char *s = t->tokens[id];
        while (*s && w < maxcat) {
            int l; int cp = u8_to_cp((const unsigned char *)s, &l);
            int b = (cp >= 0 && cp < 512) ? code2byte[cp] : (cp & 0xFF);
            out[w++] = (char)b;
            s += l;
        }
    }
    if (w < maxcat) out[w] = '\0'; else out[maxcat - 1] = '\0';
    return w;
}
