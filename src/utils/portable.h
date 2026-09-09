/* 跨平台可移植辅助：把 Windows 缺少的 POSIX 函数桥接到 UCRT/MinGW 版本。
   每个源文件按需 include 本头即可。 */
#ifndef MO_PORTABLE_H
#define MO_PORTABLE_H

#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>

/* posix_memalign(PPTR, align, n) -> Windows: _aligned_malloc */
static inline int mo_posix_memalign(void **p, size_t align, size_t n) {
    void *q = _aligned_malloc(n, align);
    if (!q) return 1;
    *p = q;
    return 0;
}
static inline void mo_aligned_free(void *p) { _aligned_free(p); }

/* localtime_r(const time_t*, struct tm*) -> Windows: localtime_s(struct tm*, const time_t*) */
static inline struct tm *mo_localtime_r(const time_t *t, struct tm *tm_) {
    localtime_s(tm_, t);
    return tm_;
}

/* qsort_r(base, n, sz, cmp(const,const,void*), ctx) -> Windows: 用静态 trampoline 桥接。
   （仅用于模型加载等单线程一次性调用，非线程安全。） */
static int (*_mo_qsort_cmp)(const void *, const void *, void *);
static void *_mo_qsort_ctx;
static int _mo_qsort_tramp(const void *a, const void *b) {
    return _mo_qsort_cmp(a, b, _mo_qsort_ctx);
}
static inline void mo_qsort_r(void *base, size_t n, size_t sz,
                              int (*cmp)(const void *, const void *, void *), void *ctx) {
    _mo_qsort_cmp = cmp;
    _mo_qsort_ctx = ctx;
    qsort(base, n, sz, _mo_qsort_tramp);
}

/* mkdir(path, 0755) -> Windows: 只取一个参数 */
#include <direct.h>
static inline int mo_makedirs(const char *p) { return _mkdir(p); }

/* 文件大小(64 位)。Windows 的 long 是 32 位，ftell 对 >2GB 文件会溢出，必须用 _ftelli64 */
#include <stdio.h>
static inline int64_t mo_file_size(FILE *f) {
    _fseeki64(f, 0, SEEK_END);
    int64_t n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    return n;
}

/* 让控制台按 UTF-8 解释输出：cmd/PowerShell 默认 GBK(936)，程序输出的是 UTF-8 字节
   若不设置，中文会显示成 "鍔犺浇" 这类乱码。仅在真有控制台时生效，管道/重定向无影响。 */
#include <windows.h>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
static inline void mo_set_console_utf8(void) {
    SetConsoleOutputCP(65001);   /* CP_UTF8 */
    SetConsoleCP(65001);
}

#else
#define mo_posix_memalign posix_memalign
#define mo_aligned_free  free
#define mo_localtime_r   localtime_r
#define mo_qsort_r       qsort_r
#include <sys/stat.h>
#define mo_makedirs(p)   mkdir((p), 0755)
static inline int64_t mo_file_size(FILE *f) {
    if (fseek(f, 0, SEEK_END) != 0) return -1;
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    return (int64_t)n;
}
static inline void mo_set_console_utf8(void) {}
#endif

#endif /* MO_PORTABLE_H */
