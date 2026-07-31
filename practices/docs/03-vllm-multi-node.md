# Day 3: vLLM 多机多卡 + 性能 Benchmark

> 学习目标：掌握 vLLM 多节点 Ray 集群部署、性能 Benchmark 方法论、参数调优

## 🎯 本日学习内容

- [ ] Ray 集群搭建基础
- [ ] vLLM 多节点部署（Docker + Ray）
- [ ] Benchmark 核心指标详解
- [ ] vLLM 原生 Benchmark CLI 使用
- [ ] 性能调优实践

## 📖 理论学习（3h）

### 1. Ray 集群架构（1h）

```
┌─────────────────────┐     ┌─────────────────────┐
│   Head Node         │     │   Worker Node 1      │
│  ┌───────────────┐  │     │  ┌───────────────┐   │
│  │ Ray Head      │  │     │  │ Ray Worker    │   │
│  │ (GCS Server)  │──┼─────┼─▶│               │   │
│  └───────────────┘  │     │  └───────────────┘   │
│  ┌───────────────┐  │     │  ┌───────────────┐   │
│  │ vLLM Engine   │  │     │  │ vLLM Engine   │   │
│  │ (GPU 0-7)     │  │     │  │ (GPU 0-7)     │   │
│  └───────────────┘  │     │  └───────────────┘   │
└─────────────────────┘     └─────────────────────┘
         │                           │
         └──────── NCCL ────────────┘
           (InfiniBand / RoCEv2)
```

**Ray 核心概念**：
- **Head Node**：集群控制节点，运行 GCS（Global Control Store）
- **Worker Node**：计算节点，注册到 Head
- **Object Store**：分布式共享内存
- **Task/Actor**：Ray 的任务执行模型

### 2. 网络通信栈（1h）

```
LLM 推理(TP/PP 通信)
        ↓
    NCCL (集合通信库)
        ↓
┌───────┴───────┐
│  IB / RoCEv2  │  ← 物理网络
│ (GPUDirect)   │
└───────┬───────┘
        ↓
   GPU Memory (KV Cache、Weights)
```

**关键网络环境变量**：
```bash
export NCCL_IB_DISABLE=0                    # 启用 InfiniBand
export NCCL_IB_GID_INDEX=3                  # RoCEv2 需要
export NCCL_SOCKET_IFNAME=eth0,ib0          # 指定网络接口
export NCCL_DEBUG=INFO                       # 调试 NCCL 通信
export NCCL_DEBUG_SUBSYS=ALL                 # 详细日志
export NCCL_IB_TIMEOUT=22                    # 超时设置
export NCCL_IB_QPS_PER_CONNECTION=8          # QP 数
```

### 3. Benchmark 指标体系（1h）

| 指标 | 全称 | 说明 | 关注场景 |
|------|------|------|---------|
| **TTFT** | Time to First Token | 首 token 延迟 | 交互式应用（聊天） |
| **TPOT** | Time Per Output Token | 每个输出 token 时间 | 长文本生成 |
| **ITL** | Inter-Token Latency | token 间延迟 | 流式体验 |
| **Throughput** | — | tok/s，req/s | 离线批处理 |
| **E2E Latency** | End-to-End Latency | 完整请求耗时 | API 服务 SLA |

**理解指标间关系**：
```
E2E Latency = TTFT + (output_tokens - 1) × TPOT
Throughput = total_output_tokens / total_time
```

## 🛠️ 实操练习（4.5h）

### 练习 1：用 vLLM 自带 Benchmark CLI

参考脚本 [scripts/03-vllm-benchmark.sh](../scripts/03-vllm-benchmark.sh)。

```bash
# 启动服务
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096

# 新终端运行 benchmark
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-prompts 200 \
  --request-rate 5 \
  --random-input-len 512 \
  --random-output-len 128 \
  --result-dir ./bench-results/
```

**输出解读**：
```
============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  42.3
Total input tokens:                      102400
Total generated tokens:                  25123
Request throughput (req/s):              4.73
Output token throughput (tok/s):         593.9
---------------Time to First Token----------------
Mean TTFT (ms):                          285.3
Median TTFT (ms):                        234.1
P99 TTFT (ms):                           892.4
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          55.2
Median TPOT (ms):                        49.8
P99 TPOT (ms):                           156.3
---------------Inter-token Latency----------------
Mean ITL (ms):                           45.8
Median ITL (ms):                         42.1
P99 ITL (ms):                            134.7
---------------------------------------------------
```

### 练习 2：多节点部署（Docker + Ray）

参考脚本 [scripts/04-multi-node-ray-setup.sh](../scripts/04-multi-node-ray-setup.sh)。

```bash
# ===== 在 Head Node (192.168.1.10) 上 =====

# Step 1: 启动 Ray Head 容器
docker run --gpus all --network host --shm-size=32g \
  --name ray-head \
  -v /mnt/models:/models:ro \
  vllm/vllm-openai:latest \
  bash -c "
    ray start --head --node-ip-address=192.168.1.10 --port=6379 \
    --object-manager-port=8076 --num-gpus=8
    sleep infinity
  "

# 检查 Ray 状态
docker exec ray-head ray status

# Step 2: 在 Head 上启动 vLLM
docker exec ray-head bash -c "
  ray list nodes
  
  vllm serve /models/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
"


# ===== 在 Worker Node (192.168.1.11) 上 =====

docker run --gpus all --network host --shm-size=32g \
  --name ray-worker \
  -v /mnt/models:/models:ro \
  vllm/vllm-openai:latest \
  bash -c "
    ray start --address=192.168.1.10:6379 \
    --object-manager-port=8076 --num-gpus=8
    sleep infinity
  "
```

### 练习 3：参数调优实验矩阵

```bash
# 对比不同 tp_size / pp_size 组合
# 需要 2 台机器，每台 4 卡

export MODEL=Qwen/Qwen2.5-72B-Instruct
export RESULTS_DIR=./bench-results

# Test 1: TP=4, PP=1 (单节点 4 卡)
vllm serve $MODEL --tensor-parallel-size 4 --pipeline-parallel-size 1 &
sleep 60
vllm bench serve --model $MODEL --num-prompts 200 \
  --request-rate 4 --result-dir $RESULTS_DIR/tp4-pp1

# Test 2: TP=4, PP=2 (两节点，各 4 卡)
vllm serve $MODEL --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --distributed-executor-backend ray &
sleep 60
vllm bench serve --model $MODEL --num-prompts 200 \
  --request-rate 4 --result-dir $RESULTS_DIR/tp4-pp2

# 对比结果
python -c "
import json, os

for config in ['tp4-pp1', 'tp4-pp2']:
    with open(f'$RESULTS_DIR/{config}/results.json') as f:
        r = json.load(f)
    print(f'{config}:')
    print(f'  Throughput: {r[\"request_throughput\"]:.1f} req/s')
    print(f'  Output tok/s: {r[\"output_throughput\"]:.1f}')
    print(f'  Mean TTFT: {r[\"mean_ttft_ms\"]:.0f} ms')
    print(f'  Mean TPOT: {r[\"mean_tpot_ms\"]:.1f} ms')
"
```

### 练习 4：GuideLLM 高级 Benchmark

[GuideLLM](https://github.com/vllm-project/guidellm) 是 vLLM 官方推荐的更灵活的 Benchmark 工具。

```bash
pip install guidellm

# 基本用法
guidellm --model Qwen/Qwen2.5-7B-Instruct \
  --backend vllm \
  --base-url http://localhost:8000 \
  --num-users 1 2 4 8 16 \
  --prompt-lengths 512 1024 2048 \
  --output-lens 128 256 \
  --output-dir ./guidellm-results/
```

## 📝 学习日志

在 `daily-logs/day-03.md` 中记录：
1. 你部署的多节点架构图
2. 不同 tp/pp 组合的 Benchmark 对比
3. NCCL 通信日志关键输出
4. 遇到的网络配置问题及解决

## 🔗 参考资料

- [vLLM 分布式推理官方文档](https://docs.vllm.ai/en/stable/serving/distributed_serving.html)
- [vLLM Benchmark CLI 文档](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM GitHub Discussion #7181 - Benchmark 教程](https://github.com/vllm-project/vllm/discussions/7181)
- [GuideLLM - 生产级 Benchmark](https://github.com/vllm-project/guidellm)
- [vLLM 多节点部署脚本](https://github.com/vllm-project/vllm/blob/main/examples/run_cluster.sh)
- [使用 Ray + Docker + vLLM 手动部署 DeepSeek (华为云)](https://support.huaweicloud.com/bestpractice-ecs/ecs_bp_6015.html)
