# -*- coding: utf-8 -*-
"""
ParlzAIPlatformPAP — llama/Mixtral 架构模型 (PyTorch)。

与 C 引擎 (src/core/llama.c) 数学一比一：
  - pre-norm RMSNorm, 学习式绝对位置编码 -> 无；用 **RoPE** 旋转位置
  - **GQA**: n_head 查询 / n_head_kv 键值
  - **SwiGLU** FFN: down(silu(gate(x)) * up(x))
  - **Mixtral MoE**: router softmax -> top-k(used) -> 重归一化 -> 专家加权和
激活: SiLU (x*sigmoid(x))。参数均以 (in, out) 行主序存储，C 端直接 x @ W。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    n_layer: int = 4
    n_embd: int = 512
    n_head: int = 8
    n_head_kv: int = 4
    ffn_dim: int = 1024
    n_expert: int = 8
    n_expert_used: int = 2
    max_seq: int = 256
    rmsnorm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return w * (x * torch.rsqrt(ms + eps))


def build_rope_cache(cfg: LlamaConfig, device=None):
    """cos/sin: [max_seq, head_dim/2]  (NeoX half-rotate)"""
    D = cfg.head_dim
    half = D // 2
    freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(cfg.max_seq, dtype=torch.float32)
    angles = torch.outer(t, freqs)          # [max_seq, half]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, T: int) -> torch.Tensor:
    """x: [B, H, T, D]（T 在 dim2）"""
    D = x.size(-1)
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]
    c = cos[:T].unsqueeze(0).unsqueeze(1)     # [1,1,T,D/2]
    s = sin[:T].unsqueeze(0).unsqueeze(1)
    out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return out


class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        H, HK, D = cfg.n_head, cfg.n_head_kv, cfg.head_dim
        self.attn_norm = nn.Parameter(torch.ones(cfg.n_embd))
        self.attn_q = nn.Parameter(torch.randn(cfg.n_embd, H * D) * 0.02)   # (in,out)
        self.attn_k = nn.Parameter(torch.randn(cfg.n_embd, HK * D) * 0.02)
        self.attn_v = nn.Parameter(torch.randn(cfg.n_embd, HK * D) * 0.02)
        self.attn_o = nn.Parameter(torch.randn(cfg.n_embd, cfg.n_embd) * 0.02)

        self.ffn_norm = nn.Parameter(torch.ones(cfg.n_embd))
        if cfg.n_expert > 0:
            self.ffn_gate_inp = nn.Parameter(torch.randn(cfg.n_embd, cfg.n_expert) * 0.02)  # router
            self.ffn_experts = nn.ModuleList()
            for _ in range(cfg.n_expert):
                e = nn.Module()
                e.ffn_gate = nn.Parameter(torch.randn(cfg.n_embd, cfg.ffn_dim) * 0.02)
                e.ffn_up = nn.Parameter(torch.randn(cfg.n_embd, cfg.ffn_dim) * 0.02)
                e.ffn_down = nn.Parameter(torch.randn(cfg.ffn_dim, cfg.n_embd) * 0.02)
                self.ffn_experts.append(e)
        else:
            self.ffn_gate = nn.Parameter(torch.randn(cfg.n_embd, cfg.ffn_dim) * 0.02)
            self.ffn_up = nn.Parameter(torch.randn(cfg.n_embd, cfg.ffn_dim) * 0.02)
            self.ffn_down = nn.Parameter(torch.randn(cfg.ffn_dim, cfg.n_embd) * 0.02)

    def forward(self, x, cos, sin):
        cfg = self.cfg
        B, T, C = x.shape
        H, HK, D = cfg.n_head, cfg.n_head_kv, cfg.head_dim
        hk = H // HK  # 每组 kv 对应几个查询头

        a = _rmsnorm(x, self.attn_norm, cfg.rmsnorm_eps)
        q = a @ self.attn_q                                        # [B,T,H*D]
        k = a @ self.attn_k
        v = a @ self.attn_v
        q = q.view(B, T, H, D).transpose(1, 2)                     # [B,H,T,D]
        k = k.view(B, T, HK, D).transpose(1, 2)
        v = v.view(B, T, HK, D).transpose(1, 2)
        q = _apply_rope(q, cos, sin, T)
        k = _apply_rope(k, cos, sin, T)

        weight = torch.zeros(B, H, T, T, device=x.device)
        for h in range(H):
            kvh = h // hk
            qh = q[:, h]        # [B,T,D]
            kh = k[:, kvh]      # [B,T,D]
            sc = (qh @ kh.transpose(-2, -1)) / math.sqrt(D)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            sc = sc.masked_fill(~mask, float("-inf"))
            w = F.softmax(sc, dim=-1)
            weight[:, h] = w
        # 聚合
        att = torch.zeros(B, T, H * D, device=x.device)
        for h in range(H):
            kvh = h // hk
            vh = v[:, kvh]      # [B,T,D]
            o = weight[:, h] @ vh  # [B,T,D]
            att[:, :, h * D:(h + 1) * D] = o
        att = att.view(B, T, C)
        x = x + att @ self.attn_o

        f = _rmsnorm(x, self.ffn_norm, cfg.rmsnorm_eps)
        if cfg.n_expert > 0:
            logits = f @ self.ffn_gate_inp                          # [B,T,n_expert]
            probs = F.softmax(logits, dim=-1)
            top = torch.topk(probs, k=cfg.n_expert_used, dim=-1)
            wtop = top.values                                       # [B,T,used]
            idx = top.indices
            wtop = wtop / (wtop.sum(dim=-1, keepdim=True) + 1e-9)   # 重归一化
            N = B * T
            out = torch.zeros(N, C, device=x.device)
            f2 = f.reshape(N, C)
            for t in range(cfg.n_expert_used):
                eidx = idx[..., t].reshape(-1)
                w = wtop[..., t].reshape(-1, 1)
                for e in range(cfg.n_expert):
                    sel = (eidx == e)
                    if sel.any():
                        exp = self.ffn_experts[e]
                        z = f2[sel] @ exp.ffn_gate
                        z = F.silu(z) * (f2[sel] @ exp.ffn_up)
                        out[sel] += w[sel] * (z @ exp.ffn_down)
            x = x + out.view(B, T, C)
        else:
            g = F.silu(f @ self.ffn_gate)
            u = f @ self.ffn_up
            up = g * u
            x = x + (up @ self.ffn_down)
        return x


class LlamaMoE(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embd = nn.Parameter(torch.randn(cfg.vocab_size, cfg.n_embd) * 0.02)
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layer)])
        self.output_norm = nn.Parameter(torch.ones(cfg.n_embd))
        self.output = nn.Parameter(torch.randn(cfg.n_embd, cfg.vocab_size) * 0.02)
        self._cos = None
        self._sin = None

    def _rope(self):
        if self._cos is None:
            c, s = build_rope_cache(self.cfg)
            self._cos, self._sin = c, s
        return self._cos, self._sin

    def forward(self, idx):
        T = idx.size(1)
        x = self.token_embd[idx]                                    # [B,T,C]
        cos, sin = self._rope()
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = _rmsnorm(x, self.output_norm, self.cfg.rmsnorm_eps)
        logits = x @ self.output
        return logits

    @torch.no_grad()
    def generate_from_ids(self, init_ids, max_new, temperature=0.8, top_k=40):
        device = next(self.parameters()).device
        self.to(device)
        x = torch.tensor([init_ids], dtype=torch.long, device=device)
        out = list(init_ids)
        for _ in range(max_new):
            logits = self.forward(x[:, -self.cfg.max_seq:])[:, -1, :]
            if temperature <= 0:
                nxt = torch.argmax(logits).item()
            else:
                if temperature > 0: logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[-1]] = float("-inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        return out


def export_tensors(model: LlamaMoE) -> Dict[str, torch.Tensor]:
    """收集为 GGUF llama 命名的 (in,out) 张量。"""
    cfg = model.cfg
    tens: Dict[str, torch.Tensor] = {}
    tens["token_embd.weight"] = model.token_embd.detach()
    for i, b in enumerate(model.blocks):
        p = f"blk.{i}"
        tens[f"{p}.attn_norm.weight"] = b.attn_norm.detach()
        tens[f"{p}.attn_q.weight"] = b.attn_q.detach()
        tens[f"{p}.attn_k.weight"] = b.attn_k.detach()
        tens[f"{p}.attn_v.weight"] = b.attn_v.detach()
        tens[f"{p}.attn_o.weight"] = b.attn_o.detach()
        tens[f"{p}.ffn_norm.weight"] = b.ffn_norm.detach()
        if cfg.n_expert > 0:
            tens[f"{p}.ffn_gate_inp.weight"] = b.ffn_gate_inp.detach()
            for e, ex in enumerate(b.ffn_experts):
                tens[f"{p}.ffn_experts.{e}.ffn_gate.weight"] = ex.ffn_gate.detach()
                tens[f"{p}.ffn_experts.{e}.ffn_up.weight"] = ex.ffn_up.detach()
                tens[f"{p}.ffn_experts.{e}.ffn_down.weight"] = ex.ffn_down.detach()
        else:
            tens[f"{p}.ffn_gate.weight"] = b.ffn_gate.detach()
            tens[f"{p}.ffn_up.weight"] = b.ffn_up.detach()
            tens[f"{p}.ffn_down.weight"] = b.ffn_down.detach()
    tens["output_norm.weight"] = model.output_norm.detach()
    tens["output.weight"] = model.output.detach()
    return tens


def load_llama_from_gguf(path: str, cfg: LlamaConfig) -> LlamaMoE:
    """从 GGUF 张量重建 LlamaMoE（用于 C/torch 对拍）。"""
    import gguf_format
    tens, meta, ver = gguf_format.read_gguf(path)
    model = LlamaMoE(cfg)
    sd: Dict[str, torch.Tensor] = {}

    def T(k):
        return torch.from_numpy(tens[k])

    sd["token_embd"] = T("token_embd.weight")
    sd["output_norm"] = T("output_norm.weight").reshape(-1)
    sd["output"] = T("output.weight")
    for i in range(cfg.n_layer):
        b = f"blocks.{i}"
        g = f"blk.{i}"
        sd[f"{b}.attn_norm"] = T(f"{g}.attn_norm.weight").reshape(-1)
        sd[f"{b}.attn_q"] = T(f"{g}.attn_q.weight")
        sd[f"{b}.attn_k"] = T(f"{g}.attn_k.weight")
        sd[f"{b}.attn_v"] = T(f"{g}.attn_v.weight")
        sd[f"{b}.attn_o"] = T(f"{g}.attn_o.weight")
        sd[f"{b}.ffn_norm"] = T(f"{g}.ffn_norm.weight").reshape(-1)
        if cfg.n_expert > 0:
            sd[f"{b}.ffn_gate_inp"] = T(f"{g}.ffn_gate_inp.weight")
            for e in range(cfg.n_expert):
                sd[f"{b}.ffn_experts.{e}.ffn_gate"] = T(f"{g}.ffn_experts.{e}.ffn_gate.weight")
                sd[f"{b}.ffn_experts.{e}.ffn_up"] = T(f"{g}.ffn_experts.{e}.ffn_up.weight")
                sd[f"{b}.ffn_experts.{e}.ffn_down"] = T(f"{g}.ffn_experts.{e}.ffn_down.weight")
        else:
            sd[f"{b}.ffn_gate"] = T(f"{g}.ffn_gate.weight")
            sd[f"{b}.ffn_up"] = T(f"{g}.ffn_up.weight")
            sd[f"{b}.ffn_down"] = T(f"{g}.ffn_down.weight")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model
