# Day 4: SGLang 深入解析

> 学习目标：掌握 SGLang 的架构设计、RadixAttention、单节点部署、与 vLLM 差异对比

## 🎯 本日学习内容

- [ ] SGLang 架构与设计理念
- [ ] RadixAttention 原理
- [ ] SGLang 安装与基本使用
- [ ] SGLang vs vLLM 对比
- [ ] SGLang Router 介绍

## 📖 理论学习（3.5h）

### 1. SGLang 整体架构（1h）

```
┌──────────────────────────────────────────────┐
│          OpenAI API / SGLang Runtime         │
├──────────────────────────────────────────────┤
│           SGLang Engine (Python)              │
│  ┌───────────┐ ┌──────────┐ ┌─────────────┐ │
│  │TpModel     │ │Scheduler │ │RadixAttention│ │
│  │Worker      │ │(zero     │ │Manager      │ │
│  │            │ │overhead) │ │(前缀树缓存)  │ │
│  └───────────┘ └──────────┘ └─────────────┘ │
│  ┌──────────────────────────────────────┐    │
│  │   CUDA Kernels (FlashInfer, 自研)     │    │
│  └──────────────────────────────────────┘    │
│  ┌──────────────────────────────────────┐    │
│  │   NCCL 通信层 (TP/DP/PP/EP)          │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  sgl-router     │ ← Rust 实现的高性能负载均衡器
│  (可选组件)       │
└─────────────────┘
```

**设计哲学差异**：
| 维度 | vLLM | SGLang |
|------|------|--------|
| 核心创新 | PagedAttention | RadixAttention |
| 调度方式 | iteration-level | 零开销调度 |
| 前缀复用 | 精确匹配 | 前缀树 + 自动复用 |
| 结构化输出 | 需额外工具 | 原生支持 |
| Router | 外部负载均衡 | 自带 sgl-router |
| 性能优势 | 通用场景 | 共享前缀 + 多步场景 |

### 2. RadixAttention 原理（1.5h）

**传统 PagedAttention vs RadixAttention**：

```
PagedAttention: 每个请求独立分配 KV Cache block
请求1: [A][B][C][D]...
请求2: [A][B][E][F]...    → A,B 被存储了两次 ❌

RadixAttention: 用前缀树统一管理 KV Cache
      root
     /    \
    A      ...
   / \
  B   ...
 / \
C   E
     → 相同的 A,B 只存一次 ✅
```

**Radix Tree 结构**：
```
         ""
       /    \
  "What is"  "Explain"
     /           \
  "the"          "how"
   /    \           \
"capital" "meaning"  "to"
```

**优势场景**：
1. **共享系统 prompt**: 大量请求共用 `system prompt`
2. **Few-shot prompt**: 演示示例被多个请求使用
3. **Chat history**: 多轮对话中历史对话复用
4. **Beam search**: 多个候选路径共享前缀

**参考资源**：
- [SGLang Deep Dive: Inside SGLang](https://blog.sugiv.fyi/sglang-deep-dive-inside-sglang) — 最详细的架构分析
- [SGLang: Efficient Execution of Structured Language Model Programs (论文)](https://arxiv.org/abs/2312.07104)
- [mini-sglang (教学精简版)](https://github.com/sgl-project/mini-sglang) — 推荐阅读源码学习

### 3. SGLang 核心特性（1h）

**关键参数对比**：
```bash
# vLLM 启动
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --enable-prefix-caching

# SGLang 启动
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tp 2 \
  --enable-metrics
```

**SGLang 特有功能**：
- RadixAttention 自动前缀复用（无需手动开启）
- Structured Generation（约束解码）
- 原生支持 multimodal（图片、视频输入）
- DP（Data Parallelism）/ EP（Expert Parallelism）

## 🛠️ 实操练习（4h）

### 练习 1：安装 SGLang

```bash
# 方法 1: pip 安装
pip install sglang[all]

# 方法 2: Docker（推荐）
docker pull lmsysorg/sglang:latest

# 验证安装
python -c "import sglang; print(sglang.__version__)"
```

### 练习 2：启动 SGLang 服务

参考脚本 [scripts/05-sglang-basic.sh](../scripts/05-sglang-basic.sh)。

```bash
# 单卡启动
docker run --gpus all -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 30000

# 多卡 TP 启动
docker run --gpus all -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tp 2 \
  --host 0.0.0.0 \
  --port 30000
```

### 练习 3：API 调用测试

```bash
# 查看模型信息
curl http://localhost:30000/v1/models

# Chat Completion (OpenAI 兼容)
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Compare SGLang and vLLM."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### 练习 4：前缀共享效果演示

```python
# demo_prefix_sharing.py
"""
演示 RadixAttention 的前缀共享效果
需要先启动 SGLang 服务
"""
from openai import OpenAI
import time

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="not-needed"
)

# 共用的系统 prompt（长一些以体现效果）
system_prompt = """
You are an AI assistant specialized in distributed systems and high-performance computing.
You have deep knowledge of NCCL, RDMA, InfiniBand, GPU architecture, and LLM inference optimization.
Please provide detailed, accurate, and technically precise answers.
""".strip()

queries = [
    "What is NCCL and how does it work?",
    "What is RDMA and why is it important?",
    "What is the difference between RoCEv2 and InfiniBand?",
]

# 第一次请求（冷启动）
print("=== 冷启动请求 ===")
start = time.time()
resp = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": queries[0]},
    ],
    max_tokens=100,
)
cold_time = time.time() - start
print(f"冷启动时间: {cold_time:.2f}s")

# 第二次请求（相同 system prompt，不同 query）
print("\n=== 前缀复用请求 ===")
start = time.time()
resp = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": system_prompt},  # 完全相同的 system prompt 前缀
        {"role": "user", "content": queries[1]},
    ],
    max_tokens=100,
)
hot_time = time.time() - start
print(f"前缀复用时间: {hot_time:.2f}s")
print(f"加速比: {cold_time/hot_time:.1f}x")
```

### 练习 5：SGLang 自带 Benchmark

```bash
# SGLang 提供了 benchmark 目录
git clone https://github.com/sgl-project/sglang.git
cd sglang/benchmark

# 运行指定 benchmark
python3 bench_serving.py \
  --backend sglang \
  --base-url http://localhost:30000 \
  --num-prompts 200 \
  --request-rate 5 \
  --random-input-len 512 \
  --random-output-len 128
```

### 练习 6：mini-sglang 源码阅读

```bash
# 克隆 mini-sglang（教学版，代码量远小于完整版）
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 重点阅读：系统架构设计
cat docs/structures.md

# 核心代码结构
ls python/minisgl/
```

**重点阅读文件**：
- `docs/structures.md` — 系统架构概览
- `python/minisgl/scheduler.py` — 调度器
- `python/minisgl/radix_attention.py` — RadixAttention 实现
- `python/minisgl/model_runner.py` — 模型执行

## 📝 学习日志

在 `daily-logs/day-04.md` 中记录：
1. SGLang 与 vLLM 的核心差异总结
2. 前缀共享实验的结果对比
3. mini-sglang 源码阅读笔记
4. 你对 RadixAttention 的理解

## 🔗 参考资料

- [SGLang 官方文档](https://docs.sglang.ai/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [mini-sglang GitHub](https://github.com/sgl-project/mini-sglang)
- [SGLang Architecture Overview](https://sgl-project-sglang-93.mintlify.app/developer/architecture-overview)
- [SGLang Deep Dive Blog](https://blog.sugiv.fyi/sglang-deep-dive-inside-sglang)
- [SGLang Learning Materials (官方)](https://github.com/sgl-project/sgl-learning-materials)
- [Awesome ML-SYS Tutorial - SGLang 代码解析 (中文)](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md)
