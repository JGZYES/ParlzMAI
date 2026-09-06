#ifndef MO_LOGGER_H
#define MO_LOGGER_H

#ifdef __cplusplus
extern "C" {
#endif

/* 日志级别 */
typedef enum {
    MO_LOG_DEBUG = 0,
    MO_LOG_INFO,
    MO_LOG_WARN,
    MO_LOG_ERROR,
    MO_LOG_NONE
} MoLogLevel;

/* 控制全局日志级别 */
void mo_log_set_level(MoLogLevel level);
MoLogLevel mo_log_get_level(void);

/* 是否打印时间戳（默认开） */
void mo_log_set_timestamps(int enabled);

/* 一条日志；内部追加换行。level 低于全局级别时忽略 */
void mo_log_msg(MoLogLevel level, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#define MO_LOGD(...) mo_log_msg(MO_LOG_DEBUG, __VA_ARGS__)
#define MO_LOGI(...) mo_log_msg(MO_LOG_INFO,  __VA_ARGS__)
#define MO_LOGW(...) mo_log_msg(MO_LOG_WARN,  __VA_ARGS__)
#define MO_LOGE(...) mo_log_msg(MO_LOG_ERROR, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* MO_LOGGER_H */
