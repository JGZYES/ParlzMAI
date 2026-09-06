# -*- coding: utf-8 -*-
"""把自研 MoE 模型的 .pap 导出成标准 GGUF 容器。

用法:
  /opt/pap-venv/bin/python platform/export_gguf.py \
      --src models/moe-0.1b.pap --dst models/moe-0.1b.gguf --dtype fp16   # 或 q8 / f32
"""
from __future__ import annotations

import argparse
import os

import pap_format
import gguf_format

GML = {"f32": 0, "fp16": 1, "q8": 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dtype", default="fp16", choices=GML.keys())
    args = ap.parse_args()

    cfg, vocab, merges, tensors = pap_format.read_pap(args.src)
    gguf_format.save_gguf(args.dst, cfg, vocab, merges, tensors, GML[args.dtype])

    in_sz = os.path.getsize(args.src)
    out_sz = os.path.getsize(args.dst)
    print(f"已导出: {args.src} ({in_sz/1e6:.1f} MB) -> {args.dst} ({out_sz/1e6:.1f} MB)")

    # 读回验证
    rt, meta, ver = gguf_format.read_gguf(args.dst)
    print(f"读回校验: version={ver} tensors={len(rt)} metadata={len(meta)}")
    print(f"  general.architecture = {meta.get('general.architecture')}")
    print(f"  pap.vocab_size = {meta.get('pap.vocab_size')}  pap.moe_n_experts = {meta.get('pap.moe_n_experts')}")


if __name__ == "__main__":
    main()
