# -*- coding: utf-8 -*-
"""生成一个随机小模型并导出 .pap，用于 C 与 torch 的对拍验证（不训练）。"""
import os

import model as model_mod
import pap_format
import tokenizer as tok


def main(out: str = "/mnt/f/ParlzMAI/models/_smoke.pap"):
    # 纯字节词表（num_merges=0），最干净
    vocab = [bytes([b]) for b in range(256)]
    merges = []
    cfg = {"vocab_size": 256, "n_layer": 2, "d_model": 64, "n_head": 4,
           "d_ff": 128, "moe_n_experts": 4, "moe_top_k": 2, "d_expert": 128,
           "max_seq_len": 32, "rmsnorm_eps": 1e-5, "moe_mask": [1, 0]}
    model = model_mod.ParlzGPTMoE(cfg)

    tens = model_mod.export_tensors(model)
    pcfg = pap_format.ModelConfig()
    pcfg.vocab_size = cfg["vocab_size"]
    pcfg.n_layer = cfg["n_layer"]
    pcfg.d_model = cfg["d_model"]
    pcfg.n_head = cfg["n_head"]
    pcfg.d_ff = cfg["d_ff"]
    pcfg.moe_n_experts = cfg["moe_n_experts"]
    pcfg.moe_top_k = cfg["moe_top_k"]
    pcfg.d_expert = cfg["d_expert"]
    pcfg.max_seq_len = cfg["max_seq_len"]
    pcfg.num_merges = len(merges)
    pcfg.tie_weights = 0
    pcfg.rmsnorm_eps = cfg["rmsnorm_eps"]
    pcfg.moe_mask = list(cfg["moe_mask"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pap_format.save_pap(out, pcfg, vocab, merges, tens)
    print(f"exported random model -> {out}")


if __name__ == "__main__":
    main()
