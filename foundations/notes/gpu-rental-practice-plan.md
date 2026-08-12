# GPU 租用实践方案（FDE 实战）

> 面向自费租服务器练习 vLLM/SGLang 部署的完整方案：分阶段配置、预算、实践清单。

---

## 推荐平台

| 平台 | 优势 | 注意事项 |
|------|------|---------|
| AutoDL (autodl.com) | 国内最便宜，按量计费，关机不收费（保留15天数据），学生认证更便宜 | RTX 4090 高峰需抢 |
| Vast.ai (vast.ai) | 全球最便宜，RTX 4090 约 $0.15-0.30/hr | 英文界面，部分机器需验证 |

---

## 第一阶段：单机基础部署（¥30-80）

**目标**：掌握 vLLM 和 SGLang 安装、单卡启动、API 调用、性能调优

**配置**：1× RTX 4090 (24GB) 或 1× RTX 3090 (24GB)

| 平台 | GPU | 单价 | 建议时长 | 费用 |
|------|-----|------|---------|------|
| AutoDL | RTX 4090 24GB | ¥1.6-2.0/hr | 10-20h | ¥30-40 |
| AutoDL | RTX 3090 24GB | ¥1.2-1.3/hr | 10-20h | ¥15-25 |

**可实践的模型（24GB 显存）：**

| 模型 | 参数量 | 能否装下 | 说明 |
|------|--------|---------|------|
| Qwen2.5-7B-Instruct | 7B (~14GB BF16) | ✅ | 推荐入手模型 |
| Llama-3-8B-Instruct | 8B (~16GB BF16) | ✅ | |
| Qwen2.5-3B-Instruct | 3B (~6GB BF16) | ✅ | 快速实验 |
| Qwen2.5-0.5B-Instruct | 0.5B | ✅ | 验证部署流程最快 |

**实践清单：**

```bash
# 1. 环境准备（30min）
python3 -m venv vllm-env && source vllm-env/bin/activate
pip install vllm
pip install flash-attn --no-build-isolation  # 试编译流程
python3 -c "import vllm; print(vllm.__version__)"

# 2. vLLM 单卡启动（30min）
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
# 观察: 模型加载6阶段、KV Cache 预分配、CUDA Graph 捕获

# 3. 调参数练手（1-2h）
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-chunked-prefill
# 改 utilization 0.85/0.95，看 num_gpu_blocks 变化

# 4. API 测试（1h）
# Chat Completions、流式、多轮对话、benchmark

# 5. 刻意制造故障（1h）
# utilization 1.0 → 看 OOM
# max-model-len 32768 → 看 KV Cache 分配和并发限制
# 跑 benchmark → 看 TTFT/TPOT/Throughput

# 6. 切换 SGLang 重复（2-3h）
pip install sglang[all]
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct --port 30000
```

---

## 第二阶段：多卡 TP（¥20-50，建议 3-6 小时集中练）

**目标**：实践 TP、Ray 集群、多机部署

**配置 A（推荐）**：2× RTX 4090 (24GB×2)，AutoDL 约 ¥3-4/hr，3-6h 约 ¥10-25
**配置 B**：2× A100 40GB，AutoDL 约 ¥6-8/hr，3-6h 约 ¥20-50

> **注意**：2× 24GB 只能装 7B 模型（TP=2），无法跑 70B。但足够学会 TP 配置和验证方法。

```bash
# 实践 TP=2
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90
# 验证: nvidia-smi 看两张 GPU 都有进程；日志确认 TP=2

# SGLang TP=2
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct --tp 2 --port 30000
```

---

## 第三阶段：分布式 / PD 分离（¥50-150，建议 2-4 小时）

**目标**：实践 Ray 分布式部署或 PD 分离

**配置（两个独立实例）**：2× (1× A100 40GB)，AutoDL 约 ¥6-8/hr/台

> **重要**：AutoDL 两台机器内网通常不通（除非同机房）。Ray 分布式需要两台机器能互通。
> **建议**：先验证内网连通性再开始；不通也没关系，可在一台机器上开两个终端模拟 Ray（都连 localhost），配置流程完全一样。

```bash
# 如果内网通，真正分布式:
# 机器A: ray start --head --port=6379
# 机器B: ray start --address='<机器A_IP>:6379'

# 如果内网不通，单机模拟:
# 终端1: ray start --head --port=6379
# 终端2: ray start --address='127.0.0.1:6379' --num-gpus=0
# ray status 会显示 2 个节点，配置流程完全一样
```

---

## 总预算估算

| 方案 | 内容 | 预计时长 | 预计费用 |
|------|------|---------|---------|
| **极致省钱** | 1× RTX 3090 完成单机+TP模拟 | 20h | **¥30-50** |
| **标准推荐** | 1× RTX 4090 (15h) + 2× RTX 4090 (4h) | 19h | **¥50-80** |
| **完整体验** | 1× RTX 4090 (10h) + 2× A100 (4h) + 双机 (2h) | 16h | **¥100-180** |

---

## 最小实践路径（约 ¥50）

```
第 1 天: 租 1× RTX 4090，4 小时（¥8）→ vLLM 单卡安装启动 + API + 调优
第 2 天: 同一台继续，4 小时（¥8）→ SGLang 单卡 + RadixAttention 缓存效果对比
第 3 天: 租 2× RTX 4090，3 小时（¥12）→ TP=2 + Ray 单机模拟 + benchmark
第 4 天: 租双机（2× A100），2 小时（¥15）→ Ray 分布式 + 日志排查全流程
总计: ~13 小时, ¥43
```

---

## 注意事项

1. **选支持 `--gpus all` 的实例**：vLLM 需要直接访问 GPU，确认容器有 nvidia-docker 支持
2. **确认 CUDA 版本**：租机器看 CUDA 12.1+，否则 flash-attn 无法编译
3. **PD 分离阶段选同机房实例**：AutoDL 同机房内网互通，KV Cache 传输需要低延迟
4. **用完关机/销毁**：AutoDL 关机不收费（数据保留 15 天），下次开机继续
5. **70B 模型至少 2× A100 80GB**：想体验完整 TP 和显存管理，短租 2 小时 A100-80GB×2（约 ¥20-30）

---

## 2026 年参考价格（AutoDL）

| GPU | 显存 | 时租 | 月租 |
|-----|------|------|------|
| RTX 3090 | 24GB | ¥1.2-1.3 | ~¥1,000 |
| RTX 4090 | 24GB | ¥1.6-2.0 | ~¥1,200-2,800 |
| A100 40GB | 40GB | ¥3.4 | ~¥1,600 |
| A100 80GB | 80GB | ¥5.5 | - |
| H800 | 80GB | ¥8.9 | - |
| H20 | 96GB | ¥7.6 | - |

> 价格浮动较快，以官网实时报价为准。来源：AutoDL 官网及 2026 年 Q2-Q3 多方实测。
