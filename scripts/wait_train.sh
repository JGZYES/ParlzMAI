#!/usr/bin/env bash
# 等训练日志出现关键行：语料/词表/参数/步数
for i in $(seq 1 60); do
  if grep -qE "step +1|已导出|Traceback|Error|CUDA" /tmp/pap_train.log 2>/dev/null; then
    echo "FOUND_AT_${i}"; break
  fi
  sleep 5
done
echo "----- tail -----"
tail -30 /tmp/pap_train.log 2>/dev/null | tr -d '\0'
