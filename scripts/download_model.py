# -*- coding: utf-8 -*-
"""从 HF 镜像下载 GGUF / AI 模型文件。
用法: /opt/pap-venv/bin/python scripts/download_model.py --repo Qwen/Qwen2.5-0.5B-Instruct-GGUF \
        --file qwen2.5-0.5b-instruct-q4_k_m.gguf --out models/
"""
import argparse
import os
import sys

import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF 仓库（如 ggml-org/stories15M_MOE）")
    ap.add_argument("--file", required=True, help="仓库内文件名")
    ap.add_argument("--out", default="models/", help="输出目录")
    ap.add_argument("--base", default="https://hf-mirror.com", help="镜像")
    args = ap.parse_args()

    url = f"{args.base}/{args.repo}/resolve/main/{args.file}"
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, os.path.basename(args.file))
    print(f"下载 {url} -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"\r  {total/1e6:.1f} MB", end="")
    print(f"\n完成: {dst} ({total/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
