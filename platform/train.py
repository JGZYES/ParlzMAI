# -*- coding: utf-8 -*-
"""
ParlzAIPlatformPAP — 训练器。

从 (可选的) 语料文件训练一个 GPT+MoE 模型，并导出为 ParlzMAI 可运行的 .pap 文件。

用法（在 WSL 内，使用含 torch 的 venv）:
  /opt/pap-venv/bin/python platform/train.py --data data/tinyshakespeare.txt \
      --out models/moe-0.1b.pap --vocab 2048 --layers 6 --d_model 512 --heads 8 \
      --experts 16 --top_k 2 --d_expert 1024 --max_seq 256 --steps 1500 --batch 8
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch

import tokenizer as tok
import pap_format
from model import ParlzGPTMoE, export_tensors


def build_config(vocab_size, n_layer, d_model, n_head, d_ff, moe_n_experts,
                 moe_top_k, d_expert, max_seq, eps, moe_every):
    cfg = {
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "d_model": d_model,
        "n_head": n_head,
        "d_ff": d_ff,
        "moe_n_experts": moe_n_experts,
        "moe_top_k": moe_top_k,
        "d_expert": d_expert,
        "max_seq_len": max_seq,
        "rmsnorm_eps": eps,
        # 每层是否 MoE；moe_every=1 表示全部 MoE，=2 表示隔层
        "moe_mask": [1 if (l % moe_every == 0) else 0 for l in range(n_layer)],
    }
    return cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_batch(data_t, start, block_size, batch_size, device):
    """取一个 batch: x = [batch, block_size], y 右移一位。"""
    ix = torch.randint(0, data_t.numel() - block_size, (batch_size,))
    x = torch.stack([data_t[i:i + block_size] for i in ix])
    y = torch.stack([data_t[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--out", default="models/moe-0.1b.pap")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--d_expert", type=int, default=1024)
    ap.add_argument("--max_seq", type=int, default=256)
    ap.add_argument("--moe_every", type=int, default=1)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU", flush=True)
        device = "cpu"

    # ---- 读语料 + BPE（带缓存） ----
    if not os.path.exists(args.data):
        raise SystemExit(f"语料不存在: {args.data}")
    text = open(args.data, "r", encoding="utf-8").read()
    print(f"语料: {len(text)} 字符", flush=True)

    cache = args.data + ".papcache"
    if os.path.exists(cache):
        import pickle as _pk
        with open(cache, "rb") as f:
            vocab, merges, data_ids = _pk.load(f)
        vocab_size = len(vocab)
        print(f"加载预处理缓存: vocab={vocab_size} merges={len(merges)}", flush=True)
    else:
        import pickle as _pk
        vocab, merges = tok.train_bpe(text, args.vocab)
        vocab_size = len(vocab)
        data_ids = tok.encode(text, vocab_size, merges)
        with open(cache, "wb") as f:
            _pk.dump((vocab, merges, data_ids), f)
        print(f"词表: {vocab_size} (字节+合并)", flush=True)

    # ---- 构建模型 ----
    cfg = build_config(vocab_size, args.layers, args.d_model, args.heads, args.d_ff,
                       args.experts, args.top_k, args.d_expert, args.max_seq,
                       args.eps, args.moe_every)
    model = ParlzGPTMoE(cfg).to(device)
    nparams = count_params(model)
    print(f"模型参数: {nparams/1e6:.1f} M", flush=True)

    # ---- 数据 ----
    data_ids = tok.encode(text, vocab_size, merges)
    data_t = torch.tensor(data_ids, dtype=torch.long)
    block = args.max_seq

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                                  betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / max(args.warmup, 1), 1.0))
    model.train()

    print(f"开始训练 {args.steps} 步 (device={device}) ...", flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(data_t, 0, block, args.batch, device)
        loss, logits = (None, None)
        out = model(x, y)
        loss = out[1]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sched.step()

        if step % args.eval_every == 0 or step == 1:
            el = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step {step:5d}|\tloss {loss.item():.4f}\tlr {lr:.2e}\t{el:.1f}s", flush=True)
            model.eval()
            init_ids = tok.encode("The", vocab_size, merges)
            gen_ids = model.generate_from_ids(init_ids, 60, temperature=0.8, top_k=40)
            gen = tok.decode(gen_ids, vocab)
            if gen:
                print(f"    [sample] {gen}", flush=True)
            model.train()

    print(f"训练完成，耗时 {time.time()-t0:.1f}s", flush=True)
    model.eval()

    # ---- 导出 .pap ----
    tens = export_tensors(model)
    pcfg = pap_format.ModelConfig()
    pcfg.vocab_size = vocab_size
    pcfg.n_layer = args.layers
    pcfg.d_model = args.d_model
    pcfg.n_head = args.heads
    pcfg.d_ff = args.d_ff
    pcfg.moe_n_experts = args.experts
    pcfg.moe_top_k = args.top_k
    pcfg.d_expert = args.d_expert
    pcfg.max_seq_len = args.max_seq
    pcfg.num_merges = len(merges)
    pcfg.tie_weights = 0
    pcfg.rmsnorm_eps = args.eps
    pcfg.moe_mask = [int(v) for v in cfg["moe_mask"]]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pap_format.save_pap(args.out, pcfg, vocab, merges, tens)
    print(f"已导出 .pap -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
