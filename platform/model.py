# -*- coding: utf-8 -*-
"""
ParlzAIPlatformPAP — GPT + MoE 模型 (PyTorch)。

与 C 引擎 (src/core/transformer.c) 数学一对一:
  - pre-norm transformer, RMSNorm
  - 学习式绝对位置编码 (无 RoPE)
  - 缩放点积注意力 (无 GQA), 因果掩码
  - 每个 block: norm1 -> attn -> +res, norm2 -> ffn/moe -> +res
  - FFN 层可选 MoE: router 线性 + softmax -> top-k -> 专家加权和
  - 激活: tanh 近似 GELU (与 C 的 mo_gelu 一致)

导出 .pap 时所有 nn.Linear 权重做 .t() 转置成 (in,out)，C 端直接 x @ W。
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """y = x / sqrt(mean(x^2)+eps) * w，沿最后一维。"""
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(mean_sq + eps)
    return w * x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class MoEExpert(nn.Module):
    def __init__(self, d_model: int, d_expert: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_expert, bias=False)  # 上投影
        self.w2 = nn.Linear(d_expert, d_model, bias=False)  # 下投影

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


class MoELayer(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, d_expert: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([MoEExpert(d_model, d_expert) for _ in range(n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        logits = self.router(x)                       # (B,T,n_experts)
        probs = F.softmax(logits, dim=-1)             # 路由概率
        topk = torch.topk(probs, k=self.top_k, dim=-1)  # values, indices
        sel_w = topk.values                           # (B,T,top_k)
        sel_i = topk.indices                          # (B,T,top_k)
        out = torch.zeros_like(x)
        for t in range(self.top_k):
            e_idx = sel_i[..., t].reshape(-1)         # (B*T,)
            wt = sel_w[..., t].reshape(-1, 1)         # (B*T,1)
            src = x.reshape(B * T, C)
            # 用 gather 取每行对应专家的输出
            exp_out = torch.zeros(B * T, C, device=x.device)
            for e in range(self.n_experts):
                sel = (e_idx == e)
                if sel.any():
                    sub = self.experts[e](src[sel])
                    exp_out[sel] = sub
            out += (wt * exp_out).view(B, T, C)
        return out


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, cfg, is_moe: bool):
        super().__init__()
        self.is_moe = is_moe
        self.ln1 = RMSNorm(d_model, cfg["rmsnorm_eps"])
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model, cfg["rmsnorm_eps"])
        if is_moe:
            self.moe = MoELayer(d_model, cfg["moe_n_experts"], cfg["moe_top_k"], cfg["d_expert"])
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, cfg["d_ff"], bias=False),
                nn.GELU(approximate="tanh"),
                nn.Linear(cfg["d_ff"], d_model, bias=False),
            )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.is_moe:
            x = x + self.moe(self.ln2(x))
        else:
            x = x + self.ffn(self.ln2(x))
        return x


class ParlzGPTMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg["vocab_size"]
        self.max_seq_len = cfg["max_seq_len"]
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.wpe = nn.Embedding(cfg["max_seq_len"], cfg["d_model"])
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([
            Block(cfg["d_model"], cfg["n_head"], cfg, cfg["moe_mask"][l])
            for l in range(cfg["n_layer"])
        ])
        self.ln_f = RMSNorm(cfg["d_model"], cfg["rmsnorm_eps"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        assert T <= self.max_seq_len
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss
        return logits, None

    @torch.no_grad()
    def generate_from_ids(self, init_ids, max_new, temperature=0.8, top_k=40):
        """自回归采样生成（训练途中示例用，无需 tokenizer）。返回 id 列表。"""
        device = next(self.parameters()).device
        x = torch.tensor([init_ids], dtype=torch.long, device=device)
        out = list(init_ids)
        for _ in range(max_new):
            logits, _ = self(x[:, -self.max_seq_len:])
            logits = logits[0, -1]
            if temperature <= 0:
                nxt = torch.argmax(logits).item()
                out.append(nxt)
                x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
                continue
            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        return out


def export_tensors(model: "ParlzGPTMoE"):
    """收集权重为 .pap 命名张量；Linear 权重转置为 (in,out) 行主序。"""
    tens: Dict[str, torch.Tensor] = {}
    tens["wte"] = model.wte.weight.detach()            # (vocab,d)
    tens["wpe"] = model.wpe.weight.detach()            # (max_seq,d)
    for l, blk in enumerate(model.blocks):
        p = f"l{l}"
        tens[f"{p}.norm1"] = blk.ln1.weight.detach()
        tens[f"{p}.attn.wq"] = blk.attn.wq.weight.detach().t()
        tens[f"{p}.attn.wk"] = blk.attn.wk.weight.detach().t()
        tens[f"{p}.attn.wv"] = blk.attn.wv.weight.detach().t()
        tens[f"{p}.attn.wo"] = blk.attn.wo.weight.detach().t()
        tens[f"{p}.norm2"] = blk.ln2.weight.detach()
        if blk.is_moe:
            tens[f"{p}.moe.router"] = blk.moe.router.weight.detach().t()
            for e, expert in enumerate(blk.moe.experts):
                tens[f"{p}.moe.exp{e}.w1"] = expert.w1.weight.detach().t()
                tens[f"{p}.moe.exp{e}.w2"] = expert.w2.weight.detach().t()
        else:
            tens[f"{p}.ffn.w1"] = blk.ffn[0].weight.detach().t()
            tens[f"{p}.ffn.w2"] = blk.ffn[2].weight.detach().t()
    tens["norm_final"] = model.ln_f.weight.detach()
    tens["lm_head"] = model.lm_head.weight.detach().t()  # (vocab,d)->(d,vocab)? no
    # 注意: lm_head 是 (vocab,d)，C 端 h @ lm_head 需 (d,vocab)。torch Linear 权重是 (vocab,d)，
    # 转置应为 (d,vocab)。但 export 里我们要求 (in,out)；lm_head 的 "输入"是 d，"输出"是 vocab。
    tens["lm_head"] = model.lm_head.weight.detach().t()  # 保证 (d_model, vocab_size)? 见下
    return tens


def load_pap_model(path: str):
    """从 .pap 重建 torch 模型（反推 export_tensors 的转置）。返回 (model, vocab, merges)。"""
    import pap_format

    cfgp, vocab, merges, tens = pap_format.read_pap(path)
    cfg = {
        "vocab_size": cfgp.vocab_size,
        "n_layer": cfgp.n_layer,
        "d_model": cfgp.d_model,
        "n_head": cfgp.n_head,
        "d_ff": cfgp.d_ff,
        "moe_n_experts": cfgp.moe_n_experts,
        "moe_top_k": cfgp.moe_top_k,
        "d_expert": cfgp.d_expert,
        "max_seq_len": cfgp.max_seq_len,
        "rmsnorm_eps": cfgp.rmsnorm_eps,
        "moe_mask": list(cfgp.moe_mask),
    }
    model = ParlzGPTMoE(cfg)
    sd = {}
    sd["wte.weight"] = torch.from_numpy(tens["wte"])
    sd["wpe.weight"] = torch.from_numpy(tens["wpe"])
    for l in range(cfg["n_layer"]):
        sd[f"blocks.{l}.ln1.weight"] = torch.from_numpy(tens[f"l{l}.norm1"]).reshape(-1)
        sd[f"blocks.{l}.attn.wq.weight"] = torch.from_numpy(tens[f"l{l}.attn.wq"].T)
        sd[f"blocks.{l}.attn.wk.weight"] = torch.from_numpy(tens[f"l{l}.attn.wk"].T)
        sd[f"blocks.{l}.attn.wv.weight"] = torch.from_numpy(tens[f"l{l}.attn.wv"].T)
        sd[f"blocks.{l}.attn.wo.weight"] = torch.from_numpy(tens[f"l{l}.attn.wo"].T)
        sd[f"blocks.{l}.ln2.weight"] = torch.from_numpy(tens[f"l{l}.norm2"]).reshape(-1)
        if cfg["moe_mask"][l]:
            sd[f"blocks.{l}.moe.router.weight"] = torch.from_numpy(tens[f"l{l}.moe.router"].T)
            for e in range(cfg["moe_n_experts"]):
                sd[f"blocks.{l}.moe.experts.{e}.w1.weight"] = torch.from_numpy(tens[f"l{l}.moe.exp{e}.w1"].T)
                sd[f"blocks.{l}.moe.experts.{e}.w2.weight"] = torch.from_numpy(tens[f"l{l}.moe.exp{e}.w2"].T)
        else:
            sd[f"blocks.{l}.ffn.0.weight"] = torch.from_numpy(tens[f"l{l}.ffn.w1"].T)
            sd[f"blocks.{l}.ffn.2.weight"] = torch.from_numpy(tens[f"l{l}.ffn.w2"].T)
    sd["ln_f.weight"] = torch.from_numpy(tens["norm_final"]).reshape(-1)
    sd["lm_head.weight"] = torch.from_numpy(tens["lm_head"].T)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, vocab, merges
