#include "utils/logger.h"

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

static MoLogLevel g_level = MO_LOG_INFO;
static int g_timestamps = 1;

static const char *level_name(MoLogLevel l) {
    switch (l) {
        case MO_LOG_DEBUG: return "DEBUG";
        case MO_LOG_INFO:  return "INFO";
        case MO_LOG_WARN:  return "WARN";
        case MO_LOG_ERROR: return "ERROR";
        default:           return "????";
    }
}

void mo_log_set_level(MoLogLevel level) { g_level = level; }
MoLogLevel mo_log_get_level(void) { return g_level; }
void mo_log_set_timestamps(int enabled) { g_timestamps = enabled; }

void mo_log_msg(MoLogLevel level, const char *fmt, ...) {
    if (level < g_level) return;

    FILE *out = (level >= MO_LOG_WARN) ? stderr : stdout;

    if (g_timestamps) {
        time_t t = time(NULL);
        struct tm tm_buf;
        char ts[32];
        localtime_r(&t, &tm_buf); /* POSIX；Linux/glibc 下可用 */
        strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_buf);
        fprintf(out, "%s [%5s] ", ts, level_name(level));
    } else {
        fprintf(out, "[%5s] ", level_name(level));
    }

    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);
    fputc('\n', out);
    fflush(out);
}
