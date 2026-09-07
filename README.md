# ParlzMAI · ParlzAIPlatformPAP

> **ParlzMAI**（`pmai`）：纯 C 的 **MoE 推理服务** + 自研 AI 框架 **ParlzAIPlatformPAP**。
> 自研模型格式 **`.pap`**、兼容 **GGUF(llama.cpp)**，在 WSL(自带 GPU) 上**真实训练出 0.1B MoE 模型**并端到端生成。

## 项目性质
- 纯 C 推理引擎（`src/`）+ Python 训练框架（`platform/`）+ 统一 CLI 入口 **`pmai`**。
- 支持两种模型：自研 GPT+MoE 架构（`.pap`）与 **llama/Mixtral 架构**（GGUF）。
- 自定义 `.pap` 格式（`PAP1` 魔数 + 配置 + byte-level BPE 词表 + 命名张量，支持 f32/fp16/q8 量化）。
- 训练端可训 0.1B 级全层 MoE 模型，导出 `.pap` 或标准 `.gguf`。

## 架构
```
┌──────────────────────────────────────────────┐
│ pmai (统一 CLI)                               │
│   生成 · --interactive 对话 · --serve HTTP/SSE │
│   --inspect GGUF · --output 落盘              │
├──────────────────────────────────────────────┤
│ 推理引擎 (src/)  C                            │
│   GPT: RMSNorm · Attention+KV · MoE(top-k)   │
│   llama: RoPE · GQA · SwiGLU · Mixtral MoE   │
│   加载器: .pap(多dtype)/GGUF(多量化)          │
│   分词器: byte-level BPE / GPT-2 BPE         │
│   采样: top-k/temperature                    │
├──────────────────────────────────────────────┤
│ 训练框架 (platform/)  Python                  │
│   GPT+MoE/pt · BPE · .pap/GGUF 读写 · 量化   │
├──────────────────────────────────────────────┤
│ WSL2(Ubuntu 22.04) + GPU RTX5060/CUDA         │
└──────────────────────────────────────────────┘
```

## 目录结构
```
src/                            ParlzMAI C 引擎
  core/    tensor/matrix + transformer(GPT) + KVCache
  model/   config + tokenizer + model(.pap) + gguf(容器/反量化) + llama(架构)
  infer/   sample(top-k) + generate
  server/  HTTP 服务 (libmicrohttpd + SSE)
  utils/   math/mempool/logger
platform/                       ParlzAIPlatformPAP (Python)
  tokenizer.py pap_format.py gguf_format.py model.py llama_model.py llama_tokenizer.py
  train.py export_gguf.py quant_pap.py verify.py verify_llama.py make_random.py
data/        语料 (tinyshakespeare.txt)     models/  .pap/.gguf 模型
output/      构建产物 (pmai / pmai-llama / libmoe_*)     scripts/  脚本
```

## 构建与运行（WSL，Ubuntu-22.04）
```bash
cd /mnt/f/ParlzMAI
cmake -S . -B build && cmake --build build -j
ctest --test-dir build                 # 单元测试

# 构建产物统一在 output/ 目录：pmai(统一入口) / pmai-llama(llama 运行器)
./output/pmai --model models/moe-0.1b-fp16.pap --prompt "ROMEO:" --n_tokens 120 --temperature 0.8 --top_k 40
./output/pmai --model models/moe-0.1b-fp16.pap --interactive            # 对话式
./output/pmai --model models/moe-0.1b-fp16.pap --serve --port 8080      # HTTP/SSE
./output/pmai --model models/moe-0.1b-fp16.pap --prompt "QUEEN:" --output output/out.txt
./output/pmai --inspect models/moe-0.1b.gguf                           # 查看 GGUF
```

## 训练（GPU）
```bash
/opt/pap-venv/bin/python platform/train.py --data data/tinyshakespeare.txt \
  --out models/moe-0.1b.pap --vocab 2048 --layers 6 --d_model 512 --heads 8 \
  --experts 16 --top_k 2 --d_expert 1024 --max_seq 256 --steps 2500 --batch 8
# 约 109M 参数(0.1B)，全层 MoE
```

## 量化 / 格式转换
| 操作 | 命令 |
|---|---|
| `.pap` → fp16 (≈2×) | `python platform/quant_pap.py --src a.pap --dst a-fp16.pap --dtype fp16` |
| `.pap` → q8  (≈3.8×) | `python platform/quant_pap.py --src a.pap --dst a-q8.pap --dtype q8` |
| `.pap` → `.gguf`     | `python platform/export_gguf.py --src a.pap --dst a.gguf --dtype fp16` |

实测（自研 0.1B = 109,238,784 参数）：

| 文件 | 体积 | 相对 f32 | 贪心输出 |
|---|---|---|---|
| `moe-0.1b.pap` (f32) | 437.0 MB | 1.0× | 基准 |
| `moe-0.1b-fp16.pap` | 218.5 MB | 2.0× | 一致 |
| `moe-0.1b-q8.pap`   | 116.1 MB | 3.76× | 一致 |
| `moe-0.1b.gguf`(fp16)| 218.5 MB | 2.0× | 一致 |

## 一致性验证
```bash
# GPT-MoE：C vs torch 贪心
/opt/pap-venv/bin/python platform/verify.py  --pap models/_smoke.pap    # [MATCH] YES
# llama 架构：C vs torch 贪心（合成 MoE GGUF round-trip）
/opt/pap-venv/bin/python platform/verify_llama.py                       # [MATCH] YES
# llama BPE 分词器 vs llama-cpp-python（真实 Qwen2-MoE 词表）+ C 端对拍
/opt/pap-venv/bin/python -c "..."                    # 逐 token MATCH
./output/test_llmtok models/qwen25moe.gguf 'Hello'   # C 端分词器：与参考一致
```

## Git 仓库
- ✅ **GitHub**：`main = 95a3339`（`origin=https://github.com/JGZYES/ParlzMAI.git`，用 `gh auth token` + `credential.helper=store`，`http.version HTTP/1.1`）。
- ✅ **Gitee**：`gitee=https://gitee.com/JGZYES/pmai.git`，`main = master = 95a3339`（oauth2:token 认证）。
- ⚠️ **parlz**（`git.parlz.com`）：SSH(22) 本机网络不可达（443 可达）；需 **HTTPS 凭据**推送：`git push https://<TOKEN>@git.parlz.com/JGZ_YES/ParlzMAI.git main`

---

# 项目状态

## ✅ 已实现（已验证）

### 环境与基建
- WSL2 专用镜像 **Ubuntu 22.04** + 工具链（gcc/cmake/make/pip）。
- **GPU 直通**：RTX 5060 / 8GB，CUDA 可用；venv `/opt/pap-venv`（torch 2.14+cu130 / numpy）。
- **llama-cpp-python**（参考/逐 token 对照）+ **HF 镜像**（可下载真实模型）。

### 推理引擎（C，`src/`）
- ✅ tensor / matrix / math / mempool / logger；`-Wall -Wextra` 干净编译；ctest 全绿。
- ✅ **GPT-MoE 前向**：RMSNorm、带 KV Cache 注意力、MoE(router+Top-K+专家加权)、GFU FFN。
- ✅ **llama/Mixtral 架构**：RoPE(NeoX) + GQA + SwiGLU + Mixtral MoE(重归一化 Top-k) —— **C 与 PyTorch 贪心逐 token 一致（`[MATCH] YES`）**。
- ✅ **byte-level BPE**（GPT/自研两端一致）与 **GPT-2 byte-level BPE 分词器**（`platform/llama_tokenizer.py`，**与 llama-cpp-python 逐 token MATCH，含中文**）。
- ✅ **`.pap` 加载器**（多 dtype f32/fp16/q8 反量化）；**GGUF 加载器**（容器解析、元数据查询、F32/F16/BF16/Q4_0/Q5_0/Q5_1/Q8_0 反量化，**并写入 Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K 反量化，Q6_K 公式已与 llama.cpp 逐字参考一致**）。
- ✅ **`mmap` 模型加载**（`.pap` 用 `mmap` 按需分页，不再整块拷贝；实测生成正常）。
- ✅ **真实 GGUF 兼容**：**顺序探测**（元数据/张量在前）+ **版本自适应**（v1 用 u32、v2+ 用 u64）。
- ✅ 采样（top-k/temperature）+ 自回归生成。

### CLI（`pmai`，构建到 `output/`）
- ✅ `--prompt` 生成、**`--interactive` 对话**（输入一行→流式生成→exit 退出）、**`--serve` HTTP/SSE**（`POST /api/generate`、`GET /api/models`、`GET /health`）、**`--inspect` 查看任意 GGUF**、**`--output <file>` 落盘**（默认存 `output/`，交互自动 `output/chat-<ts>.txt`）。
- ✅ `pmai-llama`：llama 架构 GGUF 运行器（加载 + 生成 id）。

### 训练框架（Python，`platform/`）
- ✅ BPE 训练/编码（O(n log n)）；GPT+MoE(PyTorch)；`.pap`/`gguf` 读写；量化（fp16 ≈2×、q8 ≈3.76×）；导出 `.pap`/`.gguf`。
- ✅ **真实训练并验证**：GPU 训练 **0.1B（109M）MoE**（tiny_shakespeare，2500 步 ≈11 分钟，loss 7.79→3.73），产出 `.pap`(437MB)/fp16(218MB)/q8(116MB)/`.gguf`(218MB)。

## ⬜ 未实现 / 待办（如实）

| 类别 | 内容 |
|---|---|
| **qwen2moe 架构运行** | 真实 `Qwen2.5-MoE-2X1.5B`(Q4_K_M) 需：**K 系反量化**(Q4_K/Q5_K/Q6_K/Q8_K/Q2_K/Q3_K，已取到 llama.cpp 源码未移植) + **qwen2moe 分支**(共享专家 `ffn_*_shexp` / 3D 路由专家 `ffn_*_exps` / 双重 router / 注意力 bias)。**未发布未经验证实现**。 |
| **C 端 llama 分词器** | `llama_tokenizer.py` 已验证，但 C 端未接入 → `pmai`/`pmai-llama` 对 llama 系列暂输出 id、非文本。 |
| **llama 真实文本生成对拍** | 需上两条都完成后，用 `pmai-llama` 与 llama-cpp-python 逐 token 一致才算跑通。 |
| **推理增强** | 无 batch 推理、无 prefill 批量化、无 mmap 大模型加载、无专家预加载/热切换。 |
| **Phase 3 剩余** | HTTP/SSE/健康检查已做；缺 WebSocket、线程池/请求调度、并发限流。 |
| **Phase 4 运维** | 无 config.yaml、无 QPS/延迟监控、无优雅关闭/热重载、无模型下载工具、无 wrk/ab 基准、无 OpenAPI 文档。 |
| **精简** | GGUF v1 旧格式模型（如 `klosax` 的 v1 目录）未能完全解析（不影响现代 v2/v3）。 |
| **Tokenizer 特殊 token** | `n_special=0`，未内置 `<|im_start|>` 等。 |
| **交互上下文** | `--interactive` 为无状态（每轮仅用当前输入，不携带上文）。 |
| **训练质量** | 语料 1MB、2500 步，仅“莎士比亚腔”，要更连贯需更大语料 + 长训练。 |

## 下一步（按优先级）
1. **C 端 llama BPE 分词器** → `pmai`/`pmai-llama` 对任何 llama/GPT-2 系 GGUF 输出真实文本（边界清晰、可逐 token 验证）。
2. **K 系反量化移植**（Q4_K/Q5_K/Q6_K/Q8_K 等，从 llama.cpp 源码逐行核对）。
3. **qwen2moe 架构分支** + 与 llama-cpp-python 端到端对拍 → `pmai-llama` 真跑 `Qwen2.5-MoE-2X1.5B`。
4. 批推理 / mmap / QPS 监控 / config.yaml / 模型下载。

## 环境备注
- 构建/训练跑在 WSL `Ubuntu-22.04`：`wsl -d Ubuntu-22.04`。
- 项目 Windows `F:\ParlzMAI` ↔ WSL `/mnt/f/ParlzMAI`。
- 预处理缓存 `data/tinyshakespeare.txt.papcache`（BPE，避免重跑 ~4.6 分钟）。
