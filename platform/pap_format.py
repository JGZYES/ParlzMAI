# -*- coding: utf-8 -*-
"""
ParlzAIPlatformPAP — .pap 模型文件格式 (version 1)

二进制布局 (little-endian):

    off  size  字段
    0    4     magic = b"PAP1"
    4    4     version (u32) = 1
    8    4     dtype (u32) = 0   # 0=f32, 1=Q8(未实现)
    12   4     vocab_size (u32)
    16   4     n_layer (u32)
    20   4     d_model (u32)
    24   4     n_head (u32)
    28   4     d_ff (u32)          # 稠密 FFN 中间层
    32   4     moe_n_experts (u32)
    36   4     moe_top_k (u32)
    40   4     d_expert (u32)      # MoE 专家中间层
    44   4     max_seq_len (u32)
    48   4     num_merges (u32)
    52   4     n_special (u32)
    56   1     tie_weights (u8)
    57   1     reserved
    58   1     reserved
    59   1     reserved
    60   4     rmsnorm_eps (f32)
    64   n     moe_mask: n_layer bytes (1=MoE, 0=dense FFN)

    --- vocab 段 ---
    u32 vocab_blob_len
    blob: for id in 0..vocab_size-1: u32 len + len bytes
          for m in 0..num_merges-1 : u32 a + u32 b

    --- weights 段 ---
    u32 n_tensors
    for each: u32 name_len + name + u32 rows + u32 cols + rows*cols f32(row-major)

张量命名（行主序，列为 (in, out)，C 端直接 x @ W）：
  wte (vocab,d)  wpe (max_seq,d)
  l{L}.norm1, l{L}.attn.{wq,wk,wv,wo}, l{L}.norm2
  l{L}.moe.router (d,n_experts) ; l{L}.moe.exp{E}.{w1(d,d_expert),w2(d_expert,d)}
  l{L}.ffn.{w1(d,d_ff),w2(d_ff,d)}   (稠密层)
  norm_final (d) ; lm_head (vocab,d)
"""
from __future__ import annotations

import struct
from typing import Dict, List, Tuple

PARZ_MAGIC = b"PAP1"
VERSION = 1

_M = "<"  # little-endian


def _u32(v: int) -> bytes:
    return struct.pack(_M + "I", int(v) & 0xFFFFFFFF)


def _f32(v: float) -> bytes:
    return struct.pack(_M + "f", float(v))


class ModelConfig:
    def __init__(self):
        self.vocab_size = 0
        self.n_layer = 0
        self.d_model = 0
        self.n_head = 0
        self.d_ff = 0
        self.moe_n_experts = 0
        self.moe_top_k = 0
        self.d_expert = 0
        self.max_seq_len = 0
        self.num_merges = 0
        self.n_special = 0
        self.tie_weights = 0
        self.rmsnorm_eps = 1e-5
        self.moe_mask: List[int] = []

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


def _tensor_to_f32(weight) -> "object":
    """任意 torch/numpy/list -> numpy float32 2D (in,out) 数组。"""
    import numpy as np
    if hasattr(weight, "detach"):       # torch.Tensor
        weight = weight.detach().cpu().numpy()
    return np.asarray(weight, dtype=np.float32)


Q8_BLOCK = 32  # int8 量化块大小


def _tensor_bytes(w, dtype: int) -> bytes:
    """把一个 2D f32 数组编码成 dtype 字节流（行主序展平）。"""
    import numpy as np
    w = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    n = w.size
    if dtype == 0:                       # f32
        return w.tobytes()
    if dtype == 1:                       # fp16
        return w.astype(np.float16).astype("<f2").tobytes()
    if dtype == 2:                       # q8: 每块 [fp16 scale][32×int8]
        pad = (-n) % Q8_BLOCK
        wp = np.pad(w, (0, pad), "constant") if pad else w
        nb = wp.size // Q8_BLOCK
        b = wp.reshape(nb, Q8_BLOCK)
        scale = np.maximum(np.abs(b).max(axis=1), 1e-12)
        q = np.clip(np.round(b / scale[:, None] * 127.0), -127, 127).astype(np.int8)
        out = bytearray()
        for i in range(nb):
            out += np.float32(scale[i]).astype("<f2").tobytes()   # 2 字节 fp16 scale
            out += q[i].tobytes()
        return bytes(out)
    raise ValueError(f"未知 dtype {dtype}")


def _tensor_stored_bytes(rows, cols, dtype: int) -> int:
    n = rows * cols
    if dtype == 0:
        return n * 4
    if dtype == 1:
        return n * 2
    if dtype == 2:
        nb = (n + Q8_BLOCK - 1) // Q8_BLOCK
        return n + nb * 2
    raise ValueError(f"未知 dtype {dtype}")


def save_pap(path: str, cfg: ModelConfig, vocab: List[bytes],
             merges: List[Tuple[int, int]], tensors: Dict[str, "object"], dtype: int = 0) -> None:
    """写出 .pap 文件。tensors 的每个矩阵须为 (in, out) 行主序（C 端直接 x@W）。
    dtype: 0=f32, 1=fp16, 2=int8(块量化)。"""
    import numpy as np

    with open(path, "wb") as f:
        f.write(PARZ_MAGIC)
        f.write(_u32(VERSION))
        f.write(_u32(dtype))               # dtype
        f.write(_u32(cfg.vocab_size))
        f.write(_u32(cfg.n_layer))
        f.write(_u32(cfg.d_model))
        f.write(_u32(cfg.n_head))
        f.write(_u32(cfg.d_ff))
        f.write(_u32(cfg.moe_n_experts))
        f.write(_u32(cfg.moe_top_k))
        f.write(_u32(cfg.d_expert))
        f.write(_u32(cfg.max_seq_len))
        f.write(_u32(cfg.num_merges))
        f.write(_u32(cfg.n_special))
        f.write(bytes([cfg.tie_weights & 0xFF, 0, 0, 0]))
        f.write(_f32(cfg.rmsnorm_eps))
        # moe_mask
        mask = bytes(cfg.moe_mask)
        assert len(mask) == cfg.n_layer
        f.write(mask)

        # --- vocab blob ---
        blob = bytearray()
        for tok in vocab:
            blob += _u32(len(tok))
            blob += tok
        for (a, b) in merges:
            blob += _u32(a)
            blob += _u32(b)
        f.write(_u32(len(blob)))
        f.write(bytes(blob))

        # --- weights ---
        names = sorted(tensors.keys())
        f.write(_u32(len(names)))
        for name in names:
            w = _tensor_to_f32(tensors[name])
            if w.ndim == 1:
                w = w.reshape(1, -1)   # 一维 norm 向量按单行保存
            rows, cols = w.shape
            nb = name.encode("utf-8")
            f.write(_u32(len(nb)))
            f.write(nb)
            f.write(_u32(rows))
            f.write(_u32(cols))
            f.write(_tensor_bytes(w, dtype))


def read_pap(path: str):
    """读回 .pap，返回 (cfg, vocab, merges, tensors)。用于校验或转回 torch。"""
    import numpy as np

    tensors: Dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    off_r = lambda n: None

    def r_u32(off):
        return (struct.unpack_from(_M + "I", data, off)[0], off + 4)

    r_u8 = None

    magic = data[0:4]
    assert magic == PARZ_MAGIC, f"bad magic {magic}"
    cfg = ModelConfig()

    def u32(off):
        v, n = struct.unpack_from(_M + "I", data, off)[0], off + 4
        return v, n

    def f32(off):
        v, n = struct.unpack_from(_M + "f", data, off)[0], off + 4
        return v, n

    v, off = u32(4)
    assert v == VERSION
    dtype, off = u32(8)
    cfg.vocab_size, off = u32(12)
    cfg.n_layer, off = u32(16)
    cfg.d_model, off = u32(20)
    cfg.n_head, off = u32(24)
    cfg.d_ff, off = u32(28)
    cfg.moe_n_experts, off = u32(32)
    cfg.moe_top_k, off = u32(36)
    cfg.d_expert, off = u32(40)
    cfg.max_seq_len, off = u32(44)
    cfg.num_merges, off = u32(48)
    cfg.n_special, off = u32(52)
    cfg.tie_weights = data[56]
    off = 60
    cfg.rmsnorm_eps, off = f32(60)
    off = 64
    cfg.moe_mask = list(data[off:off + cfg.n_layer])
    off += cfg.n_layer

    blob_len, off = u32(off)
    blob = data[off:off + blob_len]
    off += blob_len

    def b32(o):  # 在 blob 内读取
        return struct.unpack_from(_M + "I", blob, o)[0], o + 4

    boff = 0
    vocab = []
    for _ in range(cfg.vocab_size):
        tl, boff = b32(boff)
        vocab.append(blob[boff:boff + tl])
        boff += tl
    merges = []
    for _ in range(cfg.num_merges):
        a, boff = b32(boff)
        b, boff = b32(boff)
        merges.append((a, b))

    n_t, off = u32(off)
    for _ in range(n_t):
        nl, off = u32(off)
        name = data[off:off + nl].decode("utf-8")
        off += nl
        rows, off = u32(off)
        cols, off = u32(off)
        n = rows * cols
        if dtype == 0:
            arr = np.frombuffer(data, dtype=np.float32, count=n, offset=off).reshape(rows, cols).copy()
            off += n * 4
        elif dtype == 1:
            arr = np.frombuffer(data, dtype=np.float16, count=n, offset=off).astype(np.float32).reshape(rows, cols).copy()
            off += n * 2
        elif dtype == 2:
            nb = (n + Q8_BLOCK - 1) // Q8_BLOCK
            vals = np.empty(n, dtype=np.float32)
            po = off
            k = 0
            for _ in range(nb):
                scale = np.frombuffer(data, dtype=np.float16, count=1, offset=po).astype(np.float32)[0]
                po += 2
                cnt = min(Q8_BLOCK, n - k)
                q = np.frombuffer(data, dtype=np.int8, count=cnt, offset=po)
                po += cnt
                vals[k:k + cnt] = q.astype(np.float32) * (scale / 127.0)
                k += cnt
            arr = vals.reshape(rows, cols).copy()
            off += nb * (2 + Q8_BLOCK)   # 每块 2 字节 scale + 32 字节 q
        else:
            raise ValueError(f"未知 dtype {dtype}")
        tensors[name] = arr
    return cfg, vocab, merges, tensors


def pap_quant(src: str, dst: str, dtype: int) -> None:
    """把已有 .pap 转成另一种 dtype（不改变词表/结构）。dtype: 1=fp16, 2=q8。"""
    cfg, vocab, merges, tensors = read_pap(src)
    save_pap(dst, cfg, vocab, merges, tensors, dtype=dtype)
