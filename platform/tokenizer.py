# -*- coding: utf-8 -*-
"""
ParlzAIPlatformPAP — byte-level BPE tokenizer.
与 C 端 (src/model/tokenizer.c) 完全一致的算法:
  - id 0..255  = 单个字节 bytes([b])
  - id 256+    = BPE 合并产生的 token (vocab[base+i])
  - merges[i] = (a, b), 合并后得到 id = 256 + i
编码采用“全局最小 rank 优先”的 BPE（与 C 端一致），用堆实现 O(n log n)。
"""
from __future__ import annotations

import heapq
from typing import List, Tuple


def train_bpe(text: str, vocab_size: int) -> Tuple[List[bytes], List[Tuple[int, int]]]:
    """在字节序列上训练 BPE，返回 (vocab, merges)。"""
    raw = text.encode("utf-8")
    ids: List[int] = list(raw)

    vocab: List[bytes] = [bytes([b]) for b in range(256)]

    def get_stats(ids: List[int]) -> dict:
        st: dict = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            st[pair] = st.get(pair, 0) + 1
        return st

    merges: List[Tuple[int, int]] = []
    while len(vocab) < vocab_size:
        stats = get_stats(ids)
        if not stats:
            break
        best = max(stats.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        (a, b) = best
        new_id = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        merges.append((a, b))
        new_ids: List[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids
    return vocab, merges


def build_ranks(merges: List[Tuple[int, int]]) -> dict:
    return {pair: i for i, pair in enumerate(merges)}


def encode(text: str, vocab_size: int, merges: List[Tuple[int, int]],
           bos: bool = False, eos: bool = False, bos_id: int = 0, eos_id: int = 0) -> List[int]:
    """全局最小 rank 的 BPE 编码；结果与 C 端 tokenizer_encode 一致。"""
    ranks = {pair: i for i, pair in enumerate(merges)}
    if not ranks:
        ids = list(text.encode("utf-8"))
    else:
        ids = _bpe_merge(list(text.encode("utf-8")), ranks)
    if bos:
        ids = [bos_id] + ids if bos_id != 0 else ids
    if eos:
        ids = ids + [eos_id]
    return ids


def _bpe_merge(ids: List[int], ranks: dict) -> List[int]:
    n = len(ids)
    if n < 2:
        return ids
    base = 256
    prev = [i - 1 for i in range(n)]
    nxt = [i + 1 for i in range(n)]
    nxt[n - 1] = -1
    alive = [True] * n

    heap: List[Tuple[int, int]] = []
    for i in range(n - 1):
        r = ranks.get((ids[i], ids[i + 1]))
        if r is not None:
            heapq.heappush(heap, (r, i))

    while heap:
        rank, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        j = nxt[i]
        if j < 0:
            continue
        cur = ranks.get((ids[i], ids[j]))
        if cur != rank:              # 过期条目：隔壁合并改变了该 pair
            if cur is not None:
                heapq.heappush(heap, (cur, i))
            continue
        # 合并 i,j -> i 保留，j 标记死亡
        ids[i] = base + rank
        alive[j] = False
        k = nxt[j]
        nxt[i] = k
        if k >= 0:
            prev[k] = i
        # 受影响的两个关节： (prev[i], i) 与 (i, k)
        pj = prev[i]
        if pj >= 0 and alive[pj]:
            r = ranks.get((ids[pj], ids[i]))
            if r is not None:
                heapq.heappush(heap, (r, pj))
        if k >= 0:
            r = ranks.get((ids[i], ids[k]))
            if r is not None:
                heapq.heappush(heap, (r, i))

    return [ids[i] for i in range(n) if alive[i]]


def decode(ids: List[int], vocab: List[bytes]) -> str:
    buf = b"".join(vocab[i] for i in ids if 0 <= i < len(vocab))
    return buf.decode("utf-8", errors="replace")
