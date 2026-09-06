#ifndef MO_API_H
#define MO_API_H

#include <stddef.h>

#include "model/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 启动 HTTP 服务并阻塞。返回 0 正常退出，非0为失败。 */
int moe_server_run(const Model *m, const char *model_name, unsigned port);

/* 供测试：解析 JSON 请求体为 (prompt, max_new, temperature, top_k)。 */
/* 返回 0 成功，非0 失败。 */
int moe_parse_generate_request(const char *json, char *prompt, size_t prompt_cap,
                               int *max_new, float *temperature, int *top_k);

#ifdef __cplusplus
}
#endif

#endif /* MO_API_H */
