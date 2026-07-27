# Day 1: LLM 推理基础

> 学习目标：理解 LLM 推理的核心原理，掌握 PagedAttention、KV Cache、Continuous Batching 等关键技术

## 🎯 本日学习内容

- [ ] Transformer 推理的两个阶段：Prefill 与 Decode
- [ ] KV Cache 原理与显存开销分析
- [ ] PagedAttention（vLLM 核心技术）
- [ ] Continuous Batching（连续批处理）
- [ ] 推理性能指标：TTFT、TPOT、ITL、Throughput

## 📖 理论学习（4h）

### 1. Prefill 与 Decode（1h）

**Prefill 阶段**：处理完整的输入 prompt，并行计算所有 token 的 KV Cache
- 计算密集型（Compute-bound）：需要大量矩阵乘法
- 一次性处理整个序列，GPU 利用率高

**Decode 阶段**：逐 token 生成输出
- 内存密集型（Memory-bound）：主要瓶颈在 KV Cache 的读取
- 每步只生成 1 个新 token，GPU 利用率低

> 💡 关键洞见：Prefill 和 Decode 的资源需求不同，这是 PD 分离架构的根本动机

### 2. KV Cache 详解（1h）

**为什么需要 KV Cache？**
- Self-Attention 中每个 token 都需要与之前所有 token 做 attention
- 不缓存的话每一步都要重新计算，复杂度 O(n²)
- 用 KV Cache 后变成 O(n)

**显存占用公式**：
```
M = 2 × (num_layers) × (num_heads) × (d_head) × (seq_len) × 2 × precision_bytes
```
- `2` = K cache + V cache
- 对于 70B 模型，batch=1，seq_len=4096：约 **3.2GB** KV Cache
- 对于 1024 并发请求：超过 **3.2TB**！

> 💡 这就是为什么需要 PagedAttention：KV Cache 是推理服务最大的显存开销

### 3. PagedAttention（1h）

**核心思想**：借鉴操作系统虚拟内存的分页机制

| 对比项 | 传统实现 | PagedAttention |
|--------|---------|---------------|
| KV Cache 管理 | 连续显存分配 | 分页（block）分配 |
| 内部碎片 | 严重（预留最大长度） | 仅最后一块可能有碎片 |
| 内存共享 | 不支持 | 支持 beam search、并行采样共享 |
| 显存利用率 | 20%-40% | 90%+ |

**关键论文**：https://arxiv.org/abs/2309.06180

**中文解读资源**：
- [vLLM 原理深度解析 (CSDN)](https://blog.csdn.net/m0_38097087/article/details/149334128)
- [图解 vLLM 源码解析 1 - 整体架构](https://www.cvmart.net/community/detail/8596)
- [图解 vLLM 源码解析 2 - Scheduler](https://www.cvmart.net/community/detail/8617)

### 4. Continuous Batching（1h）

**传统 Static Batching 的问题**：
- 所有请求必须同时开始、同时结束
- 短的请求要等长的请求完成
- 显著降低吞吐

**Continuous Batching 的改进**：
- 请求粒度：可以在任意时刻加入/离开 batch
- Iteration-level scheduling：每步迭代决定哪些请求参与
- 效果：吞吐量可提升 10-20x

**配合 PagedAttention 的调度流程**：
```
Request → Scheduler (决定谁加入) → Block Manager (分配显存) → Model Runner (实际计算)
                                                        ↑
                                              Continuous Batching 迭代调度
```

### 📺 推荐视频资源
- [vLLM 源码解析 PagedAttention 原理详解 (B站)](https://www.bilibili.com/video/BV1YfW4eDE6V/)

## 🛠️ 实操练习（3h）

### 练习 1：手动计算 KV Cache 开销

```python
# kv_cache_calculator.py
def calc_kv_cache_size(
    num_layers: int = 80,   # Llama-3 70B 的层数
    num_heads: int = 64,    # 每层 heads
    d_head: int = 128,      # 每个 head 的维度
    seq_len: int = 4096,    # 序列长度
    num_requests: int = 1,  # 并发请求数
    dtype_bytes: int = 2,   # float16 = 2 bytes
    kv_ratio: float = 2.0,  # K + V = 2
) -> dict:
    """计算 KV Cache 占用显存"""
    
    kv_per_layer_per_token = num_heads * d_head * dtype_bytes * kv_ratio
    kv_per_layer = kv_per_layer_per_token * seq_len
    total_per_request = kv_per_layer * num_layers
    total = total_per_request * num_requests
    
    return {
        "per_layer_MB": kv_per_layer / (1024**2),
        "per_request_GB": total_per_request / (1024**3),
        "total_GB": total / (1024**3),
        "gpu_hours_80gb": total / (80 * 1024**3),  # 需要多少张 H100
    }

# 测试不同场景
scenarios = [
    ("70B 单请求", 80, 64, 128, 4096, 1),
    ("70B 1024并发", 80, 64, 128, 4096, 1024),
    ("7B 单请求", 32, 32, 128, 4096, 1),
    ("7B 1024并发", 32, 32, 128, 4096, 1024),
]

for name, *args in scenarios:
    result = calc_kv_cache_size(*args)
    print(f"{name}: {result['total_GB']:.2f} GB KV Cache "
          f"(≈{result['gpu_hours_80gb']:.1f} 张 H100)")
```

### 练习 2：模拟 Continuous Batching 调度

```python
# simulate_continuous_batching.py
"""
模拟 Continuous Batching 与传统 Static Batching 的吞吐差异
"""
import random
from dataclasses import dataclass
from typing import List
import time

@dataclass
class Request:
    id: int
    prompt_len: int
    output_len: int
    arrived_at: float

def static_batching(requests: List[Request]):
    """传统静态批处理：所有请求同时开始，最长的决定结束时间"""
    if not requests:
        return 0
    
    max_total = max(r.prompt_len + r.output_len for r in requests)
    total_tokens = sum(r.output_len for r in requests)
    # 假设每秒处理 100 个 token（单卡）
    elapsed = max_total / 100
    throughput = total_tokens / elapsed
    return throughput

def continuous_batching(requests: List[Request]):
    """连续批处理：request 粒度调度"""
    # 简化模拟：假设 request 可以随时加入/离开
    # 因为 Continuous Batching 允许不同长度的请求混合，
    # 短请求不会等长请求
    if not requests:
        return 0
    
    # 按长度排序，短的先完成
    sorted_reqs = sorted(requests, key=lambda r: r.prompt_len + r.output_len)
    total_tokens = sum(r.output_len for r in sorted_reqs)
    
    # 模拟并行处理：最长的请求决定总时间下限
    # 但其他请求的超额计算可以填充空闲
    max_single = max(r.prompt_len + r.output_len for r in sorted_reqs)
    medium_single = sorted([r.prompt_len + r.output_len for r in sorted_reqs])[len(sorted_reqs)//2]
    
    # Continuous Batching 的并行利用率更高
    # 简化：用 max 和 medium 的平均估计
    effective_time = (max_single + medium_single) / 2 / 100 * 0.7  # 0.7 为调度效率系数
    throughput = total_tokens / effective_time
    return throughput

# 实验：生成不同分布的请求
for scenario, num_reqs, output_dist in [
    ("均匀短输出", 100, lambda: random.randint(50, 150)),
    ("长短混合", 100, lambda: random.choice([50, 100, 500, 1000])),
    ("长尾分布", 100, lambda: int(random.paretovariate(2)) + 50),
]:
    reqs = [Request(i, 1024, output_dist(), i * 0.1) for i in range(num_reqs)]
    
    sb = static_batching(reqs)
    cb = continuous_batching(reqs)
    
    print(f"\n{scenario}:")
    print(f"  Static Batching throughput:    {sb:.0f} tok/s")
    print(f"  Continuous Batching throughput: {cb:.0f} tok/s")
    print(f"  Speedup: {cb/sb:.1f}x")
```

### 练习 3：用 HuggingFace 直接推理（对比 baseline）

```bash
# 安装依赖
pip install transformers torch

# 用 transformers 直接推理（无优化）
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'  # 小模型快速测试
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map='auto'
)

prompt = 'Hello, explain what is KV cache in LLM inference.'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# 推理计时
start = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)
elapsed = time.time() - start
generated = outputs[0][inputs['input_ids'].shape[1]:]
num_tokens = len(generated)

print(f'Generated {num_tokens} tokens in {elapsed:.2f}s')
print(f'Throughput: {num_tokens/elapsed:.1f} tok/s')
print(f'Latency per token: {elapsed/num_tokens*1000:.1f} ms/tok')
print(f'Output: {tokenizer.decode(generated)}')
"
```

## 📝 学习日志

在 `daily-logs/day-01.md` 中记录：
1. 上述三个练习的输出结果
2. 你理解的 PagedAttention 相比传统实现的核心优势
3. Continuous Batching 在哪些场景下提升最大？
4. 遇到的报错和解决方法

## 📚 延伸阅读

- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention (论文)](https://arxiv.org/abs/2309.06180)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (SOSP 2023)](https://doi.org/10.1145/3600006.3613165)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — Continuous Batching 的原始论文
- [从源码剖析 vLLM 显存管理 (腾讯云)](https://cloud.tencent.com/developer/article/2714056)

## ❓ FAQ

**Q: 为什么说 Decode 阶段是 memory-bound 的？**
A: Decode 每步只计算 1 个 token 的 attention，计算量很小。但需要读取所有之前 token 的 KV Cache，这部分访存量大。所以瓶颈在 HBM 带宽，不在计算。

**Q: PagedAttention 的 block size 如何选择？**
A: vLLM 默认 block_size=16。太小增加管理开销，太大增加内部碎片。16 是经验平衡值。
