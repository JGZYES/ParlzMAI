#!/usr/bin/env bash
# 等待 pap-venv 的 torch/numpy 装好（用于 WSL 内训练）
for i in $(seq 1 40); do
  if /opt/pap-venv/bin/python -c 'import torch,numpy' >/dev/null 2>&1; then
    echo "READY"
    /opt/pap-venv/bin/python -c 'import torch,numpy;print("torch",torch.__version__,"cuda",torch.cuda.is_available());print("numpy",numpy.__version__)' 2>&1 | head -3
    exit 0
  fi
  sleep 6
done
echo "NOT_READY_AFTER_240s"
exit 1
