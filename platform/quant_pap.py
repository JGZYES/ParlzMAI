# -*- coding: utf-8 -*-
"""把 .pap 模型转成更小的 dtype（能力基本不变）。

用法:
  /opt/pap-venv/bin/python platform/quant_pap.py \
      --src models/moe-0.1b.pap --dst models/moe-0.1b-fp16.pap --dtype fp16

  --dtype fp16:  f32 -> 半精度, 体积约减半 (437MB -> ~219MB), 能力几乎不变
  --dtype q8 :   f32 -> int8 块量化(每32个共享fp16 scale), 约 437/4 (~110MB), 损失略大
"""
from __future__ import annotations

import argparse
import os

import pap_format

DTYPES = {"f32": 0, "fp16": 1, "q8": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dtype", default="fp16", choices=DTYPES.keys())
    args = ap.parse_args()

    dst_dtype = DTYPES[args.dtype]
    pap_format.pap_quant(args.src, args.dst, dst_dtype)

    in_sz = os.path.getsize(args.src)
    out_sz = os.path.getsize(args.dst)
    print(f"已转换: {args.src} ({in_sz/1e6:.1f} MB) -> {args.dst} ({out_sz/1e6:.1f} MB)")
    print(f"  压缩比 {in_sz/out_sz:.2f}x  dtype={args.dtype} (code {dst_dtype})")


if __name__ == "__main__":
    main()
