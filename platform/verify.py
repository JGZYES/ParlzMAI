# -*- coding: utf-8 -*-
"""对拍：torch 贪心 vs C 贪心，验证两端数学一致。"""
from __future__ import annotations

import argparse
import subprocess

import model as model_mod
import tokenizer as tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pap", required=True)
    ap.add_argument("--prompt", default="First Citizen:")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--cbin", default="/mnt/f/ParlzMAI/build/moe-serve")
    args = ap.parse_args()

    m, vocab, merges = model_mod.load_pap_model(args.pap)
    init = tok.encode(args.prompt, m.cfg["vocab_size"], merges)
    ids = m.generate_from_ids(init, args.n, temperature=0)   # greedy
    py_text = tok.decode(ids, vocab)

    out = subprocess.run(
        [args.cbin, "--model", args.pap, "--prompt", args.prompt,
         "--n_tokens", str(args.n), "--temperature", "0", "--top_k", "0"],
        capture_output=True)
    c_all = out.stdout.decode("utf-8", errors="replace")
    idx = c_all.rfind(args.prompt)
    c_text = c_all[idx:] if idx >= 0 else c_all

    print("=== torch greedy ===")
    print(py_text)
    print("=== C greedy ===")
    print(c_text.strip())
    ok = c_text.strip() == py_text.strip()
    print(f"\n[MATCH] {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
