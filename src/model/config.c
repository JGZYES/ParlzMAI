#include "model/config.h"

#include <stdlib.h>
#include <string.h>

void config_init(ModelConfig *c) {
    memset(c, 0, sizeof(*c));
    c->head_dim = 0;
    c->rmsnorm_eps = 1e-5f;
}

void config_free(ModelConfig *c) {
    if (!c) return;
    free(c->moe_mask);
    c->moe_mask = NULL;
}
