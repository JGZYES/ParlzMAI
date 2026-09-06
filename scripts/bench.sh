#!/usr/bin/env bash
# 基准：测 pmai 生成速度（tokens/s 与端到端延迟）
# 用法: scripts/bench.sh <model> [n_tokens]
set -euo pipefail
MODEL="${1:-../models/moe-0.1b-fp16.pap}"
N="${2:-100}"
CDIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$CDIR/output/pmai"

echo "== pmai 基准: $MODEL  n=$N =="
start=$(date +%s.%N)
"$BIN" "$MODEL" --prompt "The" --n_tokens "$N" --temperature 0 --top_k 0 > /tmp/bench_out.txt
end=$(date +%s.%N)

dt=$(echo "$end - $start" | bc)
tok=$(echo "$N" | bc)
tps=$(echo "scale=2; $tok / $dt" | bc)
echo "耗时: ${dt}s   约 ${tps} tokens/s（含模型加载）"

# 仅生成时间（预热一次）
start=$(date +%s.%N)
"$BIN" "$MODEL" --prompt "The" --n_tokens "$N" --temperature 0 --top_k 0 > /dev/null
end=$(date +%s.%N)
echo "第二次（已缓存）: $(echo "$end - $start" | bc)s  ~$(echo "scale=2; $N / ($end - $start)" | bc) tokens/s"
