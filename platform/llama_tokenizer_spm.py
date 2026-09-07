# -*- coding: utf-8 -*-
"""llama / SentencePiece (SPM) BPE 分词器，对齐 llama.cpp llm_tokenizer_spm。
- 预处理：开头加空格 -> ▁，空格 -> ▁（U+2581）
- 按 UTF-8 字符切分，BPE 以“合并 token 的 score”排序做优先级合并
- 未成 token 的符号按字节回退（byte -> unicode -> id，与新字节映射无关，实际为 unicode_byte_to_utf8）
"""
from __future__ import annotations
from typing import List, Tuple

import json


def bytes_to_unicode() -> dict:
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\u00a1"), ord("\u00ac") + 1)) +
          list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def unicode_to_byte() -> dict:
    return {v: k for k, v in bytes_to_unicode().items()}


class LlamaSPM:
    def __init__(self, tokens: List[str], merges: List[str], scores: List[float]):
        self.tokens = tokens
        self.tok2id = {t: i for i, t in enumerate(tokens)}
        self.scores = scores if scores else [0.0] * len(tokens)
        # 词表里 byte 位置（byte_to_token）：单字符 token 用 bytes_to_unicode 映射
        self.char2id = {}
        b2u = bytes_to_unicode()
        for i, t in enumerate(tokens):
            if len(t) == 1:
                self.char2id[t] = i
        # 空格 symbol = ▁
        self.space = "\u2581"
        # merges "a b" -> score（用左 token 的 score 作优先级）
        self.merge_score = {}
        for m in merges:
            if " " in m:
                a, b = m.split(" ", 1)
                # 合并后文本 txt=a+b，其 score 若在词表则有对应 token
                self.merge_score[(a, b)] = self._score_of(a + b)

    def _score_of(self, s: str) -> float:
        i = self.tok2id.get(s, -1)
        return self.scores[i] if i >= 0 else -1e9

    # 把当前句子中的每个字节转成 ▁-空格后的 UTF-8 字符序列（保留 multi-byte 原字节）
    def _prep(self, text: str) -> List[str]:
        s = " " + text           # 前置空格 -> ▁
        s = s.replace(" ", self.space)
        # 按 UTF-8 字符切分
        return [s[i] for i in range(len(s))] if not self._is_mb(s) else self._utf8_chars(s)

    @staticmethod
    def _is_mb(s: str) -> bool:
        return any(ord(c) > 0x7f for c in s)

    @staticmethod
    def _utf8_chars(s: str) -> List[str]:
        return list(s)

    def encode(self, text: str, bos: bool = False, eos: bool = False,
               bos_id: int = 1, eos_id: int = 2):
        parts = self._prep(text)
        # 优先合并：反复找相邻 pair，其合并后是 token 且 score 最高
        while True:
            best = None
            for i in range(len(parts) - 1):
                merged = parts[i] + parts[i + 1]
                tid = self.tok2id.get(merged)
                if tid is None:
                    continue
                sc = self.scores[tid]
                if best is None or sc > best[0]:
                    best = (sc, i, tid)
            if best is None:
                break
            _, i, tid = best
            parts[i] = self.tokens[tid]
            del parts[i + 1]
        # 映射到 id；非 token 的按字节回退
        ids = []
        for p in parts:
            tid = self.tok2id.get(p)
            if tid is not None:
                ids.append(tid)
            else:
                # 逐字符按字节回退（char -> byte -> utf8 字串 -> token）
                for ch in p:
                    b = ch.encode("utf-8")
                    ids += self._fallback_byte(b)
        if bos:
            ids = [bos_id] + ids
        if eos:
            ids = ids + [eos_id]
        return ids

    def _fallback_byte(self, byte_utf8: bytes) -> List[int]:
        # 字节 -> 新字节映射后的 unicode 字符 -> token
        b2u = bytes_to_unicode()
        toks = []
        for b in byte_utf8:
            u = b2u[b]
            tid = self.char2id.get(u)
            toks.append(tid if tid is not None else 0)
        return toks

    def decode(self, ids: List[int]) -> str:
        u2b = unicode_to_byte()
        out = []
        for i in ids:
            if 0 <= i < len(self.tokens):
                out.append(self.tokens[i])
        s = "".join(out)
        # ▁ -> 空格；其余按字节回退 -> utf8
        res = bytearray()
        for ch in s:
            if ch == self.space:
                res.append(0x20)
            elif ch in u2b:
                res.append(u2b[ch])
            else:
                res += ch.encode("utf-8")
        return res.decode("utf-8", errors="replace")
