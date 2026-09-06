# -*- coding: utf-8 -*-
"""GPT-2 / llama 的 byte-level BPE 分词器（读 GGUF tokenizer.ggml.tokens/merges）。"""
from __future__ import annotations
from typing import List, Tuple


def byte_to_unicode() -> dict:
    """GPT-2 的 byte→unicode 映射（与 llama.cpp’llama’ tokenizer 一致）。"""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\u00a1"), ord("\u00ac") + 1)) +
          list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip([b for b in bs], [chr(c) for c in cs]))


def unicode_to_byte() -> dict:
    return {v: k for k, v in byte_to_unicode().items()}


class LlamaTokenizer:
    def __init__(self, tokens: List[str], merges: List[str]):
        self.tokens = tokens
        # 字符 -> token id（只收录单字符 token）
        self.symbol2id = {t: i for i, t in enumerate(tokens) if len(t) == 1}
        # merges: "a b" -> rank
        self.ranks = {}
        self.pairs = []
        for i, m in enumerate(merges):
            a, b = m.split(" ", 1) if " " in m else (m, "")
            self.ranks[(a, b)] = i
            self.pairs.append((a, b))

    def encode(self, text: str, bos: bool = False, eos: bool = False,
               bos_id: int = 151646, eos_id: int = 151643) -> List[int]:
        b2u = byte_to_unicode()
        syms = [b2u[b] for b in text.encode("utf-8")]
        # BPE 合并（最小 rank）
        while len(syms) >= 2:
            best = None
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            _, i = best
            syms[i] = syms[i] + syms[i + 1]
            del syms[i + 1]
        idmap = {t: i for i, t in enumerate(self.tokens)}
        out = [idmap.get(s, self.symbol2id.get(s, 0)) for s in syms]
        if bos:
            out = [bos_id] + out
        if eos:
            out = out + [eos_id]
        return out

    def decode(self, ids: List[int]) -> str:
        u2b = unicode_to_byte()
        chars = []
        for i in ids:
            if 0 <= i < len(self.tokens):
                chars.append(self.tokens[i])
        s = "".join(chars)
        out = bytearray()
        for ch in s:
            out.append(u2b.get(ch, ord(ch) & 0xFF))
        return out.decode("utf-8", errors="replace")
