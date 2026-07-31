# Day 5: SGLang 进阶 — PD 分离 + 多机部署

> 学习目标：掌握 SGLang Prefill/Decode 分离架构、多机多卡部署、Router 负载均衡

## 🎯 本日学习内容

- [ ] PD (Prefill/Decode) 分离架构原理
- [ ] SGLang PD 分离部署
- [ ] SGLang 多机部署（Ray）
- [ ] sgl-router 缓存感知路由
- [ ] Mooncake 分离方案

## 📖 理论学习（3h）

### 1. PD 分离架构原理（1.5h）

**为什么需要 PD 分离？**

```
传统统一架构：
┌──────────────────────────┐
│   同一批 GPU 同时处理      │
│   Prefill + Decode       │
│                          │
│  ❌ Prefill 计算密集       │
│  ❌ Decode 内存密集        │
│  ❌ 长 prompt 阻塞短输出    │
└──────────────────────────┘

PD 分离架构：
┌──────────┐    ┌──────────┐
│ Prefill  │    │ Decode   │
│ Cluster  │───▶│ Cluster  │
│ (计算优化) │    │ (内存优化) │
│          │    │          │
│ TP=8    │    │ TP=8    │
│ Batch=64│    │ Batch=256│
│ GPU: H100│   │ GPU: H200│
└──────────┘    └──────────┘
```

**核心流程**：
```
1. Prefill 节点接收请求
2. 计算 KV Cache（计算密集型）
3. 将 KV Cache + last token 传给 Decode 节点
4. Decode 节点逐 token 生成（内存密集型）
5. 完成后返回
```

**优势**：
- 资源利用率更高（各自针对优化）
- 允许 non-uniform scaling（Prefill 少但强，Decode 多但广）
- 降低 TPOT（Decode 节点不被 Prefill 打断）
- 支持跨地域部署（Prefill 近用户，Decode 集中部署）

**参考**：
- [SGLang PD 分离流程细节 (CSDN)](https://blog.csdn.net/u013701860/article/details/151795035)
- [部署 SGLang PD 分离推理服务 (阿里云)](https://help.aliyun.com/zh/ack/cloud-native-ai-suite/user-guide/deploy-sglang-pd-separated-inference-service)

### 2. sgl-router 架构（0.5h）

```
          Client Requests
                │
                ▼
        ┌──────────────┐
        │  sgl-router  │ ← Rust, 高性能, 缓存感知
        │  (负载均衡)    │
        └──────┬───────┘
         ┌─────┼─────────┐
         ▼     ▼         ▼
      ┌────┐ ┌────┐  ┌────┐
      │W1  │ │W2  │  │W3  │  ← SGLang Workers
      └────┘ └────┘  └────┘
```

**sgl-router 与普通 LB 的区别**：
- 普通 LB：RR / 最少连接 → 忽略 KV Cache 状态
- sgl-router：追踪每个 worker 的 RadixAttention 缓存状态 → 路由到缓存命中率最高的 worker

### 3. PD 分离 + Mooncake（0.5h）

Mooncake 是面向 PD 分离的 KV Cache 传输方案，通过 RDMA 直接传输 GPU 显存数据。

```
Prefill Node              Decode Node
┌──────────┐   RDMA     ┌──────────┐
│ GPU Mem  │──────────▶ │ GPU Mem  │
│ (KV Cache)│  GPUDirect│ (KV Cache)│
└──────────┘            └──────────┘
```

## 🛠️ 实操练习（4.5h）

### 练习 1：SGLang 单机 PD 分离

参考脚本 [scripts/07-sglang-pd-separation.sh](../scripts/07-sglang-pd-separation.sh)。

```bash
# Step 1: 启动 Prefill 节点（2 卡）
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tp 2 \
  --disaggregation-mode prefill \
  --port 30001 \
  --disaggregation-bootstrap-port 34001 \
  --enable-metrics

# Step 2: 启动 Decode 节点（另外 2 卡）
CUDA_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tp 2 \
  --disaggregation-mode decode \
  --port 30002 \
  --enable-metrics

# Step 3: 启动 Scheduler（负载均衡）
python3 -m sglang.srt.disaggregation.mini_lb \
  --prefill http://localhost:30001 \
  --decode http://localhost:30002 \
  --host 0.0.0.0 \
  --port 30000
```

### 练习 2：SGLang 多节点 Docker 部署

参考脚本 [scripts/06-sglang-multi-node.sh](../scripts/06-sglang-multi-node.sh)。

```bash
# ===== 机器 A (192.168.1.10, 8 卡) =====

# 启动 Prefill
docker run --gpus all --network host --shm-size=32g \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path /models/Qwen2.5-72B-Instruct \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30001 \
  --enable-metrics

# ===== 机器 B (192.168.1.11, 8 卡) =====

# 启动 Decode
docker run --gpus all --network host --shm-size=32g \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path /models/Qwen2.5-72B-Instruct \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30002 \
  --enable-metrics

# ===== 机器 C (调度器) =====

# 启动 scheduler
docker run --network host \
  lmsysorg/sglang:latest \
  python3 -m sglang.srt.disaggregation.mini_lb \
  --prefill http://192.168.1.10:30001 \
  --decode http://192.168.1.11:30002 \
  --host 0.0.0.0 \
  --port 30000
```

### 练习 3：sgl-router 使用

```bash
# 启动 sgl-router（需要单独构建）
git clone https://github.com/sgl-project/sglang.git
cd sglang/sgl-router

# 构建（Rust）
cargo build --release

# 启动 router，后端挂 2 个 SGLang worker
./target/release/sgl-router \
  --port 30000 \
  --worker http://worker1:30001,http://worker2:30002 \
  --cache-threshold 0.7  # 缓存命中阈值
```

### 练习 4：SGLang 多机统一 Benchmark

```bash
# 对比统一部署 vs PD 分离的吞吐差异

# 测试 1：统一部署（4 卡 TP）
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tp 4 --host 0.0.0.0 --port 30000 --enable-metrics

# Benchmark 统一部署
python3 bench_serving.py \
  --backend sglang \
  --base-url http://localhost:30000 \
  --num-prompts 200 --request-rate 5 \
  --random-input-len 1024 --random-output-len 256 \
  --result-dir ./results/standard/

# 测试 2：PD 分离（2+2 卡）
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct --tp 2 \
  --disaggregation-mode prefill --port 30001 \
  --disaggregation-bootstrap-port 34001 &

CUDA_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct --tp 2 \
  --disaggregation-mode decode --port 30002 &

python3 -m sglang.srt.disaggregation.mini_lb \
  --prefill http://localhost:30001 \
  --decode http://localhost:30002 \
  --host 0.0.0.0 --port 30000

# Benchmark PD 分离
python3 bench_serving.py \
  --backend sglang \
  --base-url http://localhost:30000 \
  --num-prompts 200 --request-rate 5 \
  --random-input-len 1024 --random-output-len 256 \
  --result-dir ./results/pd-separated/
```

### 练习 5：性能对比分析

```python
# analyze_results.py
import json

def load_result(path):
    with open(f"{path}/result.json") as f:
        return json.load(f)

standard = load_result("./results/standard")
pd_sep = load_result("./results/pd-separated")

print(f"{'指标':<30} {'统一部署':<20} {'PD分离':<20} {'提升':<10}")
print("-"*80)

metrics = [
    ("request_throughput", "吞吐(req/s)"),
    ("output_throughput", "输出吞吐(tok/s)"),
    ("mean_ttft_ms", "平均TTFT(ms)"),
    ("mean_tpot_ms", "平均TPOT(ms)"),
    ("median_ttft_ms", "中位TTFT(ms)"),
]

for key, name in metrics:
    s = standard.get(key, 0)
    p = pd_sep.get(key, 0)
    change = ((p - s) / s * 100) if s else 0
    direction = "↑" if change > 0 else "↓"
    print(f"{name:<30} {s:<20.2f} {p:<20.2f} {direction}{abs(change):>7.1f}%")
```

## 📝 学习日志

在 `daily-logs/day-05.md` 中记录：
1. PD 分离架构的配置过程
2. 统一部署 vs PD 分离的 Benchmark 对比
3. sgl-router 的使用体验
4. 你对 PD 分离适用场景的分析

## 🔗 参考资料

- [阿里云 — 部署 SGLang PD 分离推理服务](https://help.aliyun.com/zh/ack/cloud-native-ai-suite/user-guide/deploy-sglang-pd-separated-inference-service)
- [华为云 — 使用 SGLang + Docker 多机多卡部署 DeepSeek](https://support.huaweicloud.com/bestpractice-ecs/ecs_bp_6018.html)
- [知乎 — SGLang 多机 3090 测试](https://zhuanlan.zhihu.com/p/32893164108)
- [CSDN — SGLang PD 分离流程详解](https://blog.csdn.net/u013701860/article/details/151795035)
- [Mooncake + SGLang PD 分离方案](https://hangzhou2025.gosim.org/zh/schedule/sglang-prefilldecode-disaggregation-with-mooncake/)
- [SGLang Router 架构改进提案 (#7532)](https://github.com/sgl-project/sglang/issues/7532)
