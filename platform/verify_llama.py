# -*- coding: utf-8 -*-
"""对拍：llama 架构 C(torch 数学) — 合成随机 MoE 模型 → GGUF → C 加载运行 → 与 torch 贪心 ids 一致。"""
from __future__ import annotations

import argparse
import subprocess

import llama_model as lm
import gguf_format


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/f/ParlzMAI/models/_llama_smoke.gguf")
    ap.add_argument("--cbin", default="/mnt/f/ParlzMAI/build/moe-llama")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--init", default="1 42 100")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch = __import__("torch")
    torch.manual_seed(1234)

    cfg = lm.LlamaConfig(vocab_size=512, n_layer=2, n_embd=128, n_head=8, n_head_kv=4,
                         ffn_dim=256, n_expert=4, n_expert_used=2, max_seq=24,
                         rmsnorm_eps=1e-5, rope_theta=10000.0)

    model = lm.LlamaMoE(cfg)
    tens = lm.export_tensors(model)
    # 字节 tokenizer（合成）: tokens = chr(b), merges = []
    tokens = [chr(b) for b in range(256)]
    merges = []
    gguf_format.save_llama_gguf(args.out, cfg, tens, gml_type=1, tokens=tokens, merges=merges)

    # torch 端：从 GGUF 重建（严格圆回）
    m2 = lm.load_llama_from_gguf(args.out, cfg)
    m2.to("cpu")
    init = [int(x) for x in args.init.split()]
    py_ids = m2.generate_from_ids(init, args.n, temperature=0)   # 贪心
    py_gen = py_ids[len(init):]                                  # 去掉 init 前缀

    # C 端
    cmd = [args.cbin, args.out, str(args.n)] + [str(x) for x in init]
    out = subprocess.run(cmd, capture_output=True, text=True)
    c_ids = [int(x) for x in out.stdout.split()]

    print("=== torch 贪心 ids ===")
    print(py_ids)
    print("=== C 贪心 ids ===")
    print(c_ids)
    ok = py_gen == c_ids
    print(f"\n[MATCH] {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
