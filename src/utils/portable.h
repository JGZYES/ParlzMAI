/* 跨平台可移植辅助：把 Windows 缺少的 POSIX 函数桥接到 UCRT/MinGW 版本。
   每个源文件按需 include 本头即可。 */
#ifndef MO_PORTABLE_H
#define MO_PORTABLE_H

#include <stddef.h>
#include <stdlib.h>
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

#else
#define mo_posix_memalign posix_memalign
#define mo_aligned_free  free
#define mo_localtime_r   localtime_r
#define mo_qsort_r       qsort_r
#include <sys/stat.h>
#define mo_makedirs(p)   mkdir((p), 0755)
#endif

#endif /* MO_PORTABLE_H */
