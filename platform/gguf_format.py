# -*- coding: utf-8 -*-
"""
GGUF (llama.cpp) v2/v3 容器 读取/写入，用于把 ParlzAIPlatformPAP 的自研 MoE 模型
导出成标准 GGUF，并能在 ParlzMAI (C 引擎) 中加载运行。也支持读任意 GGUF 做查看/量化分析。

张量命名沿用 .pap 的名字。元数据约定：
  pap.* (见 model.py / C 端 model_load_gguf)
"""
from __future__ import annotations

import struct
from typing import Dict, List, Tuple

import numpy as np

try:
    import pap_format as pf
except Exception:  # pragma: no cover
    pf = None

_M = "<"
MAGIC = b"GGUF"

# ggml 类型
GML_F32 = 0
GML_F16 = 1
GML_Q4_0 = 2
GML_Q5_0 = 6
GML_Q5_1 = 7
GML_Q8_0 = 8

# gguf 元数据值类型
GV_U8 = 0
GV_U32 = 4
GV_F32 = 6
GV_STRING = 8
GV_ARRAY = 9
TYPE_NAME = {0: "F32", 1: "F16", 2: "Q4_0", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0"}


def _u64(v):
    return struct.pack(_M + "Q", int(v))
def _u32(v):
    return struct.pack(_M + "I", int(v) & 0xFFFFFFFF)
def _f32(v):
    return struct.pack(_M + "f", float(v))
def _gstr(s: str) -> bytes:
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def _meta(vt: int, value) -> bytes:
    if vt == GV_U8:
        return bytes([int(value)])
    if vt == GV_U32:
        return _u32(value)
    if vt == GV_F32:
        return _f32(value)
    if vt == GV_STRING:
        return _gstr(value)
    if vt == GV_ARRAY:
        inner, arr = value
        if inner == GV_U8:
            bb = bytes(arr)
            return _u32(GV_U8) + _u64(len(bb)) + bb
        if inner == GV_U32:
            return _u32(GV_U32) + _u64(len(arr)) + b"".join(_u32(x) for x in arr)
        if inner == GV_STRING:
            return _u32(GV_STRING) + _u64(len(arr)) + b"".join(_gstr(s) for s in arr)
        raise ValueError("不支持的数组内类型")
    raise ValueError(f"未知元数据类型 {vt}")


def _tensor_bytes(arr: np.ndarray, gml_type: int) -> bytes:
    w = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    n = w.size
    if gml_type == GML_F32:
        return w.tobytes()
    if gml_type == GML_F16:
        return w.astype(np.float16).astype("<f2").tobytes()
    if gml_type == GML_Q8_0:
        pad = (-n) % 32
        wp = np.pad(w, (0, pad), "constant") if pad else w
        nb = wp.size // 32
        b = wp.reshape(nb, 32)
        maxabs = np.maximum(np.abs(b).max(axis=1), 1e-12)
        d = (maxabs / 127.0).astype(np.float16)
        # 每块: [fp16 d][32 int8]
        out = bytearray()
        for i in range(nb):
            di = d[i]
            q = np.clip(np.round(b[i] / float(di) / 1.0), -127, 127).astype(np.int8)
            out += np.float16(di).astype("<f2").tobytes()
            out += q.tobytes()
        return bytes(out)
    raise ValueError(f"未知 ggml 类型 {gml_type}")


def save_gguf(path: str, cfg, vocab: List[bytes], merges: List[Tuple[int, int]],
              tensors: Dict[str, np.ndarray], gml_type: int = GML_F16) -> None:
    """导出 .gguf。cfg 为 pap_format.ModelConfig。gml_type 选 F16/Q8_0/F32。"""
    from pap_format import ModelConfig
    rows, cols = (cfg.d_model, cfg.vocab_size)  # 占位断言，未用
    names = sorted(tensors.keys())

    # 计算张量表与 data 偏移
    tensor_meta = []
    data_chunks = []
    offset = 0
    for name in names:
        w = np.ascontiguousarray(tensors[name], dtype=np.float32)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        r, c = w.shape
        encoded = _tensor_bytes(w, gml_type)
        # ggml dims: [ne0=列, ne1=行]
        tensor_meta.append((name, [c, r], gml_type, offset))
        data_chunks.append(encoded)
        offset += len(encoded)

    # 元数据
    meta_kv: List[Tuple[str, int, object]] = [
        ("general.architecture", GV_STRING, "moe-pap"),
        ("general.name", GV_STRING, "ParlzAIPlatformPAP MoE 0.1B"),
        ("pap.vocab_size", GV_U32, cfg.vocab_size),
        ("pap.n_layer", GV_U32, cfg.n_layer),
        ("pap.d_model", GV_U32, cfg.d_model),
        ("pap.n_head", GV_U32, cfg.n_head),
        ("pap.d_ff", GV_U32, cfg.d_ff),
        ("pap.moe_n_experts", GV_U32, cfg.moe_n_experts),
        ("pap.moe_top_k", GV_U32, cfg.moe_top_k),
        ("pap.d_expert", GV_U32, cfg.d_expert),
        ("pap.max_seq_len", GV_U32, cfg.max_seq_len),
        ("pap.num_merges", GV_U32, cfg.num_merges),
        ("pap.rmsnorm_eps", GV_F32, cfg.rmsnorm_eps),
        ("pap.moe_mask", GV_ARRAY, (GV_U8, cfg.moe_mask)),
        ("pap.vocab_len", GV_ARRAY, (GV_U32, [len(t) for t in vocab])),
        ("pap.vocab_blob", GV_ARRAY, (GV_U8, b"".join(vocab))),
        ("pap.merges_a", GV_ARRAY, (GV_U32, [a for a, _ in merges])),
        ("pap.merges_b", GV_ARRAY, (GV_U32, [b for _, b in merges])),
    ]

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(_u32(3))                       # version
        f.write(_u64(len(tensor_meta)))        # tensor_count
        f.write(_u64(len(meta_kv)))            # metadata_kv_count

        for (k, vt, val) in meta_kv:
            f.write(_gstr(k))
            f.write(_u32(vt))
            f.write(_meta(vt, val))

        for (name, dims, t, off) in tensor_meta:
            f.write(_gstr(name))
            f.write(_u32(len(dims)))
            for d in dims:
                f.write(_u64(d))
            f.write(_u32(t))
            f.write(_u64(off))

        # data 段（顺序与 tensor 表一致）
        for chunk in data_chunks:
            f.write(chunk)


def read_gguf(path: str):
    """读取 GGUF，返回 (tensors: dict, metadata: dict)。用于验证/查看。"""
    data = open(path, "rb").read()
    off = 0

    def u32(o):
        return struct.unpack_from(_M + "I", data, o)[0], o + 4
    def u64(o):
        return struct.unpack_from(_M + "Q", data, o)[0], o + 8
    def gstr(o):
        n, o = u64(o)
        return data[o:o + n].decode("utf-8"), o + n

    assert data[0:4] == MAGIC
    version, off = u32(4)
    tensor_count, off = u64(8)
    meta_count, off = u64(16)

    # 探测顺序：张量名总以 .weight 结尾
    n0, po = u64(off)
    tensor_first = data[po:po + n0].endswith(b".weight")

    def _meta_loop(off):
        metadata = {}
        for _ in range(meta_count):
            k, off = gstr(off)
            vt, off = u32(off)
            v, off = _read_val(data, off, vt)
            metadata[k] = v
        return metadata, off

    def _tensor_loop(off):
        tm = []
        for _ in range(tensor_count):
            name, off = gstr(off)
            ndims, off = u32(off)
            dims = []
            for _ in range(ndims):
                d, off = u64(off)
                dims.append(d)
            t, off = u32(off)
            o, off = u64(off)
            tm.append((name, dims, t, o))
        return tm, off

    if tensor_first:
        tensor_meta, off = _tensor_loop(off)
        metadata, off = _meta_loop(off)
    else:
        metadata, off = _meta_loop(off)
        tensor_meta, off = _tensor_loop(off)

    data_off = off
    tensors = {}
    for (name, dims, t, o) in tensor_meta:
        n = 1
        for d in dims:
            n *= d
        arr = _dequant(data, data_off + o, n, t)
        if len(dims) >= 2:
            arr = arr.reshape(dims[1], dims[0])   # ggml 行主序: (ne1 rows, ne0 cols) = (in,out)
        tensors[name] = arr
    return tensors, metadata, version


def read_gguf_meta(path: str):
    """只解析 header + 元数据 + 张量信息，不做反量化（用于读分词器/结构）。"""
    data = open(path, "rb").read()
    off = 0

    def u32(o):
        return struct.unpack_from(_M + "I", data, o)[0], o + 4
    def u64(o):
        return struct.unpack_from(_M + "Q", data, o)[0], o + 8
    def gstr(o):
        n, o = u64(o)
        return data[o:o + n].decode("utf-8", errors="replace"), o + n

    assert data[0:4] == MAGIC
    version, off = u32(4)
    tensor_count, off = u64(8)
    meta_count, off = u64(16)
    n0, po = u64(off)
    tensor_first = data[po:po + n0].endswith(b".weight")

    def _meta(off):
        md = {}
        for _ in range(meta_count):
            k, off = gstr(off)
            vt, off = u32(off)
            v, off = _read_val(data, off, vt)
            md[k] = v
        return md, off

    def _tens(off):
        tm = []
        for _ in range(tensor_count):
            name, off = gstr(off)
            nd, off = u32(off)
            dims = []
            for _ in range(nd):
                d, off = u64(off)
                dims.append(d)
            t, off = u32(off)
            o, off = u64(off)
            tm.append((name, dims, t, o))
        return tm, off

    if tensor_first:
        tmeta, off = _tens(off)
        md, off = _meta(off)
    else:
        md, off = _meta(off)
        tmeta, off = _tens(off)
    return md, tmeta, version


def _read_val(data, off, vt):
    def _u32(o):
        return struct.unpack_from(_M + "I", data, o)[0], o + 4
    if vt == GV_U8:
        return data[off], off + 1
    if vt == 1:   # INT8
        return struct.unpack_from(_M + "b", data, off)[0], off + 1
    if vt == 2:   # UINT16
        return struct.unpack_from(_M + "H", data, off)[0], off + 2
    if vt == 3:   # INT16
        return struct.unpack_from(_M + "h", data, off)[0], off + 2
    if vt == GV_U32:
        return struct.unpack_from(_M + "I", data, off)[0], off + 4
    if vt == 5:   # INT32
        return struct.unpack_from(_M + "i", data, off)[0], off + 4
    if vt == GV_F32:
        return struct.unpack_from(_M + "f", data, off)[0], off + 4
    if vt == 7:   # BOOL
        return bool(data[off]), off + 1
    if vt == GV_STRING:
        n, o = struct.unpack_from(_M + "Q", data, off)[0], off + 8
        return data[o:o + n].decode("utf-8", errors="replace"), o + n
    if vt == GV_ARRAY:
        inner, o = _u32(off)
        n, o = struct.unpack_from(_M + "Q", data, o)[0], o + 8
        items = []
        for _ in range(n):
            it, o = _read_val(data, o, inner)
            items.append(it)
        return items, o
    if vt == 10 or vt == 11:   # UINT64 / INT64
        return struct.unpack_from(_M + "q", data, off)[0], off + 8
    if vt == 12:   # FLOAT64
        return struct.unpack_from(_M + "d", data, off)[0], off + 8
    raise ValueError(f"未知元数据类型 {vt}")


def _dequant(data, off, n, gml_type):
    if gml_type == GML_F32:
        return np.frombuffer(data, np.float32, count=n, offset=off).astype(np.float32)
    if gml_type == GML_F16:
        return np.frombuffer(data, np.float16, count=n, offset=off).astype(np.float32)
    if gml_type == GML_Q8_0:
        nb = (n + 31) // 32
        vals = np.empty(n, dtype=np.float32)
        po = off
        k = 0
        for _ in range(nb):
            scale = np.frombuffer(data, np.float16, count=1, offset=po).astype(np.float32)[0]
            po += 2
            cnt = min(32, n - k)
            q = np.frombuffer(data, np.int8, count=cnt, offset=po)
            po += cnt
            vals[k:k + cnt] = q.astype(np.float32) * scale
            k += cnt
        return vals
    raise ValueError(f"不支持读取 ggml 类型 {gml_type}")


def save_llama_gguf(path: str, cfg, tensors: Dict[str, "object"], gml_type: int = GML_F16,
                    tokens=None, merges=None, bos: int = 1, eos: int = 2,
                    name: str = "llama-moe"):
    """写 llama/Mixtral 架构的 GGUF。cfg 为 platform.llama_model.LlamaConfig。"""
    import numpy as np
    names = sorted(tensors.keys())
    tensor_meta = []
    data_chunks = []
    offset = 0
    for nm in names:
        w = np.ascontiguousarray(tensors[nm], dtype=np.float32)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        r, c = w.shape
        enc = _tensor_bytes(w, gml_type)
        tensor_meta.append((nm, [c, r], gml_type, offset))
        data_chunks.append(enc)
        offset += len(enc)

    meta_kv = [
        ("general.architecture", GV_STRING, "llama"),
        ("general.name", GV_STRING, name),
        ("llama.block_count", GV_U32, cfg.n_layer),
        ("llama.embedding_length", GV_U32, cfg.n_embd),
        ("llama.attention.head_count", GV_U32, cfg.n_head),
        ("llama.attention.head_count_kv", GV_U32, cfg.n_head_kv),
        ("llama.attention.layer_norm_rms_epsilon", GV_F32, cfg.rmsnorm_eps),
        ("llama.feed_forward_length", GV_U32, cfg.ffn_dim),
        ("llama.expert_count", GV_U32, cfg.n_expert),
        ("llama.expert_used_count", GV_U32, cfg.n_expert_used),
        ("llama.context_length", GV_U32, cfg.max_seq),
        ("llama.rope.dimension_count", GV_U32, cfg.head_dim),
        ("llama.rope.freq_base", GV_F32, cfg.rope_theta),
        ("llama.vocab_size", GV_U32, cfg.vocab_size),
        ("tokenizer.ggml.model", GV_STRING, "llama"),
        ("tokenizer.ggml.pre", GV_STRING, "default"),
    ]
    if tokens is not None:
        meta_kv.append(("tokenizer.ggml.tokens", GV_ARRAY, (GV_STRING, list(tokens))))
    if merges is not None:
        meta_kv.append(("tokenizer.ggml.merges", GV_ARRAY, (GV_STRING, list(merges))))
    meta_kv.append(("tokenizer.ggml.bos_token_id", GV_U32, bos))
    meta_kv.append(("tokenizer.ggml.eos_token_id", GV_U32, eos))

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(_u32(3))
        f.write(_u64(len(tensor_meta)))
        f.write(_u64(len(meta_kv)))
        for (k, vt, val) in meta_kv:
            f.write(_gstr(k))
            f.write(_u32(vt))
            f.write(_meta(vt, val))
        for (nm, dims, t, off) in tensor_meta:
            f.write(_gstr(nm))
            f.write(_u32(len(dims)))
            for d in dims:
                f.write(_u64(d))
            f.write(_u32(t))
            f.write(_u64(off))
        for chunk in data_chunks:
            f.write(chunk)
