# Day 2: vLLM 深度解析 — 单机部署

> 学习目标：掌握 vLLM 的安装、API 使用、单机多卡 TP/PP 配置、基础性能调优

## 🎯 本日学习内容

- [ ] vLLM 安装与架构概览
- [ ] OpenAI 兼容 API 的调用
- [ ] 单机多卡：Tensor Parallelism (TP)
- [ ] 单机多卡：Pipeline Parallelism (PP)
- [ ] 核心参数调优：max-model-len, gpu-memory-utilization, max-num-seqs
- [ ] 性能基准测试

## 📖 理论学习（3h）

### 1. vLLM 整体架构（1h）

```
┌─────────────────────────────────────────────┐
│              OpenAI API Server              │
│          (FastAPI + Uvicorn)                │
├─────────────────────────────────────────────┤
│              LLM Engine                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │Scheduler │  │Block     │  │Model     │ │
│  │          │  │Manager   │  │Runner    │ │
│  │决定谁加入│  │分页管理KV│  │执行前向 │ │
│  │batch     │  │Cache     │  │传播      │ │
│  └──────────┘  └──────────┘  └──────────┘ │
│  ┌──────────────────────────────────┐      │
│  │    PagedAttention (CUDA Kernel)  │      │
│  └──────────────────────────────────┘      │
│  ┌──────────────────────────────────┐      │
│  │ Worker (TP/PP) 通信层 (NCCL)    │      │
│  └──────────────────────────────────┘      │
└─────────────────────────────────────────────┘
```

**关键组件**：
- **Scheduler**：决定哪些请求进入本轮 batch
- **Block Manager**：管理 PagedAttention 的物理/逻辑 block
- **Model Runner**：加载模型并执行前向传播
- **Worker**：分布式时每个 GPU 一个 worker，通过 NCCL 通信

### 2. 并行策略（1h）

**Tensor Parallelism (TP)**：
- 将单个 transformer layer 切分到多个 GPU
- 每张卡持有模型的一部分，共同计算一层
- 每层需要 all-reduce 通信 → 对互联带宽要求高
- 推荐在同一节点内使用（NVLink 最佳）

**Pipeline Parallelism (PP)**：
- 按层切分：不同 GPU 负责不同 layer
- 每张卡持有完整的部分层
- 通信量小（只需传递中间激活）
- 适合跨节点场景（弥补跨节点带宽瓶颈）

**选择策略**：
```
模型显存需求 / 单卡显存 = 最小 GPU 数
TP 取单节点 GPU 数（尽量整除 head 数）
PP 取节点数
总 GPU = TP × PP

例如：70B (~140GB FP16)
- A100-80GB × 2: TP=2, PP=1
- 跨 2 节点各 4 卡: TP=4, PP=2
```

### 3. vLLM 关键参数解读（1h）

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|---------|
| `--tensor-parallel-size` | 1 | TP 并行度 | 等于单节点 GPU 数 |
| `--pipeline-parallel-size` | 1 | PP 并行度 | 等于节点数 |
| `--gpu-memory-utilization` | 0.9 | 预留 KV Cache 比例 | 高并发调大，长序列调小 |
| `--max-model-len` | 自动 | 模型最大序列长度 | 按实际需求设置，越小缓存越多 |
| `--max-num-seqs` | 256 | 一次调度最大请求数 | 高并发场景调大 |
| `--max-num-batched-tokens` | 自动 | 一次 batch 最大 token 数 | 限制显存爆炸 |
| `--enable-prefix-caching` | False | 前缀缓存（共享 prompt） | 共享前缀场景开启 |
| `--kv-cache-dtype` | auto | KV Cache 精度 | `fp8` 可省一半显存 |

## 🛠️ 实操练习（4h）

### 练习 1：安装 vLLM

```bash
# 推荐用 Docker（最稳定）
docker pull vllm/vllm-openai:latest

# 或者 pip 安装（确保 CUDA 环境）
pip install vllm

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

### 练习 2：启动 vLLM 服务

```bash
# 基本启动（单卡）
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 4096
```

参考脚本 [scripts/02-multi-gpu-vllm.sh](../scripts/02-multi-gpu-vllm.sh) 查看更多配置。

### 练习 3：OpenAI API 测试

```bash
# 测试 API 连通性
curl http://localhost:8000/v1/models

# Chat Completion 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "什么是 PagedAttention？用中文解释。"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### 练习 4：多卡 TP 部署

```bash
# 2 卡 TP 部署
docker run --gpus all -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 4096

# 或 4 卡
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 8192
```

### 练习 5：参数调优实验

```bash
# 对比不同 gpu-memory-utilization 的影响
for util in 0.7 0.8 0.9 0.95; do
  echo "=== Testing gpu_memory_utilization=$util ==="
  docker run --gpus all -d --name vllm-test-$util \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization $util \
    --max-model-len 4096
  sleep 30  # 等启动
  
  # 运行 benchmark（见 Day 3）
  
  docker stop vllm-test-$util && docker rm vllm-test-$util
done
```

### 练习 6：用 Python 客户端进行负载测试

```python
# test_client.py
from openai import OpenAI
import time
import concurrent.futures
from statistics import mean, median

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

def send_request(prompt: str) -> dict:
    """发送单个请求并统计时间"""
    start = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.7,
    )
    elapsed = time.time() - start
    
    usage = response.usage
    ttft = None  # 这里需要流式模式才能拿到 TTFT
    
    return {
        "latency": elapsed,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }

# 并发测试
prompts = [
    "Explain quantum computing in simple terms.",
    "Write a short poem about AI.",
    "What is the capital of France?",
    "How to make pizza?",
    "Explain TCP/IP protocol.",
] * 4  # 20 个请求

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(send_request, p) for p in prompts]
    for f in concurrent.futures.as_completed(futures):
        results.append(f.result())

# 统计
latencies = [r["latency"] for r in results]
print(f"请求数: {len(results)}")
print(f"平均延迟: {mean(latencies):.2f}s")
print(f"中位延迟: {median(latencies):.2f}s")
print(f"最大延迟: {max(latencies):.2f}s")
print(f"最小延迟: {min(latencies):.2f}s")
print(f"总输出 tokens: {sum(r['output_tokens'] for r in results)}")
print(f"总耗时: {sum(latencies):.1f}s")
```

## 📝 学习日志

在 `daily-logs/day-02.md` 中记录：
1. 你的硬件配置（GPU型号、显存、数量）
2. 各个练习的执行结果
3. 不同 tp_size 下观察到的显存占用差异
4. 你尝试了哪些参数组合？效果如何？

## 🔗 参考资料

- [vLLM 官方文档 - Serving](https://docs.vllm.ai/en/latest/serving/)
- [vLLM 分布式推理文档](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [vLLM GitHub Issue #12762 - 多机部署讨论](https://github.com/vllm-project/vllm/issues/12762)
- [图解 vLLM 源码解析 1 - 整体架构](https://www.cvmart.net/community/detail/8596)
