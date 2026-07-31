# vLLM 完整部署指南（逐行命令）

> 涵盖单机部署、Ray 分布式部署、PD 分离部署。每步含验证命令和期望日志。

---

## 环境假设

- OS: Ubuntu 22.04 / Rocky Linux 9
- GPU: H100 × 8 (单机), 每节点 NVLink 全互联
- CUDA: 12.4+, Driver: 550+
- Python: 3.10-3.12
- 网络: InfiniBand NDR400 (多机场景)

---

## 一、单机部署

### 1.1 环境准备

```bash
# 1. 确认 GPU 可见
nvidia-smi
# 期望: 列出 8 张 H100，Driver Version: 550.x, CUDA Version: 12.4

# 2. 确认 CUDA 工具链
nvcc --version
# 期望: Cuda compilation tools, release 12.4

# 3. 创建虚拟环境
python3 -m venv vllm-env
source vllm-env/bin/activate

# 4. 安装 vLLM
pip install vllm
# 期望: Successfully installed vllm-0.x.x

# 5. 安装 flash-attention（强烈推荐但可选）
pip install flash-attn --no-build-isolation
# 期望: Successfully installed flash-attn-2.x.x
# 编译失败不用管，vLLM 自动回退 xformers

# 6. 验证安装
python3 -c "import vllm; print(vllm.__version__)"
```

### 1.2 启动服务（单卡）

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

**验证服务启动成功：**

```bash
# 等待约 30s-2min，直到日志出现：
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete.

# 健康检查
curl http://localhost:8000/health
# 期望: 返回空或 OK

# 模型列表
curl http://localhost:8000/v1/models
# 期望: {"object":"list","data":[{"id":"Qwen/Qwen2.5-7B-Instruct",...}]}
```

### 1.3 启动服务（多卡 TP）

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

**验证 TP 配置：**

```bash
# 看日志确认 TP Worker 启动
grep -i "tensor_parallel\|# GPU blocks" vllm.log
# 期望: tensor_parallel_size: 4
#       # GPU blocks: 12345

# 确认进程数
ps aux | grep vllm | grep -v grep | wc -l
# 期望: 约 6-8 个（API Server + Engine Core + 4 GPU Workers + ZMQ）
```

### 1.4 测试 API

```bash
# Chat Completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-72B-Instruct",
       "messages":[{"role":"user","content":"用一句话介绍北京"}],
       "max_tokens":50,"temperature":0.7}'
# 期望: {"choices":[{"message":{"content":"北京是..."},"finish_reason":"stop"}],...}

# 流式输出
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-72B-Instruct",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":20,"stream":true}'
# 期望: 多个 data: 行，最后 data: [DONE]

# Prometheus Metrics
curl http://localhost:8000/metrics | grep -E "vllm_num_requests|vllm_num_blocks"
# 期望: vllm_num_requests_running 0.0
#       vllm_num_requests_waiting 0.0
#       vllm_num_blocks_available ...
```

---

## 二、Ray 分布式部署

### 2.1 环境准备（所有节点）

```bash
# 每个节点确认 GPU
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# 每个节点安装相同版本（版本必须一致！）
python3 -m venv vllm-env && source vllm-env/bin/activate
pip install vllm==0.6.3 ray==2.9.0

# 验证
python3 -c "import ray; print(ray.__version__)"
python3 -c "import vllm; print(vllm.__version__)"
```

### 2.2 启动 Ray 集群

```bash
# ===== 头节点 (192.168.1.10) =====
ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --num-gpus=8

# 期望日志:
#  Local node IP: 192.168.1.10
#  Next steps: ray start --address='192.168.1.10:6379'

# ===== 每个 Worker 节点 =====
ray start --address='192.168.1.10:6379' --num-gpus=8

# 期望日志:
#  Connected to Ray cluster.

# ===== 头节点验证集群 =====
ray status
# 期望:
#  Node count: 3
#  Total GPU: 24    ← 3 节点 × 8 GPU
```

### 2.3 启动 vLLM

```bash
# 在头节点执行
source vllm-env/bin/activate
echo $RAY_ADDRESS  # 期望: 192.168.1.10:6379

vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

**验证分布式部署：**

```bash
# 1. 确认 Ray 集群 GPU 数
grep "Ray cluster" vllm.log
# 期望: INFO: Connected to Ray cluster. Total GPUs: 24

# 2. 确认 Worker 分布
grep "worker.*rank" vllm.log | head -20
# 期望: rank 0-15 (TP=8 × PP=2 = 16 Workers)

# 3. 各节点确认 GPU 使用
nvidia-smi  # 头节点
ssh worker1 "nvidia-smi"  # Worker 节点

# 4. API 测试
curl http://192.168.1.10:8000/health
curl http://192.168.1.10:8000/v1/models
```

### 2.4 Ray 常见问题

```bash
# 问题: Worker 节点 GPU 不显示 → ray status --verbose 看每节点 GPU 数

# 问题: NCCL 跨节点超时
NCCL_DEBUG=INFO vllm serve ... 2>&1 | grep "NCCL.*transport"
# NET/Plugin 或 NET/IB = ✅ | NET/Socket = ❌（降级 TCP）
# 解决: export NCCL_SOCKET_IFNAME=ib0 NCCL_IB_DISABLE=0

# 问题: Ray 版本不一致 → pip list | grep ray（所有节点对比）
# 问题: 头节点重启后 Worker 掉线 → 每节点 ray stop && ray start --address=...
```

---

## 三、PD 分离部署

### 3.1 NIXL 从 0 到 1

```bash
# ===== 所有节点 =====

# 1. 硬件检查
lspci | grep -i mel   # 确认 Mellanox 网卡
ibstat                 # State: Active, Rate: 400 Gbps
nvidia-smi nvlink --capabilities 2>/dev/null | grep -i gpudirect
# 期望: GPUDirect RDMA: Supported

# 2. 安装 RDMA 依赖
sudo apt install -y rdma-core libibverbs1 ibverbs-utils librdmacm1

# 3. 安装 NIXL
pip install nixl
python3 -c "import nixl; print(nixl.__version__)"

# 4. 验证 RDMA 连通性
ibaddr  # 查看 IB 地址
# 在 Prefill 节点测试到 Decode 节点的带宽:
ib_write_bw -d mlx5_0 --report_gbits <decode_ib_ip>
# 期望: 接近 400 Gbps
# 异常: < 50 Gbps → 可能走非 RDMA 路径

# 5. 验证 NCCL + NIXL
NCCL_NET_PLUGIN=nixl NCCL_DEBUG=INFO python3 -c "
import torch; import torch.distributed as dist
dist.init_process_group(backend='nccl', init_method='tcp://localhost:29500', rank=0, world_size=1)
" 2>&1 | grep -i nixl
# 期望: NCCL INFO NET/Plugin: Using network plugin nixl
```

### 3.2 MoonCake 从 0 到 1

```bash
# ===== 所有节点 =====

# 1. 安装 MoonCake
pip install mooncake-transfer-engine
# 或从源码: git clone https://github.com/kvcache-ai/Mooncake && cd Mooncake && pip install -e .

# 2. 验证安装
python3 -c "import mooncake; print(mooncake.__version__)"

# 3. 启动 MoonCake Transfer Engine（后台服务）
mooncake-transfer-engine \
  --host 0.0.0.0 --port 5123 \
  --ib-device mlx5_0 &

# 期望日志:
# INFO: MoonCake Transfer Engine started on 0.0.0.0:5123
# INFO: Using InfiniBand device: mlx5_0

# 4. 验证 Engine 运行
curl http://localhost:5123/health
# 期望: {"status": "ok"}

# 5. 测试 GPU 到 GPU 传输
python3 -c "
import torch, mooncake
tensor = torch.zeros(1024, 1024, device='cuda')
print('GPU tensor ready')
"
```

### 3.3 启动 PD 分离

```bash
# ===== 环境变量（两个节点都要设）=====
export NCCL_IB_DISABLE=0
export NCCL_NET_PLUGIN=nixl       # 或用 MoonCake: export VLLM_USE_MOONCAKE=1
export NCCL_SOCKET_IFNAME=ib0
export NIXL_DEBUG=1               # MoonCake: export MOONCAKE_DEBUG=1

# ===== Prefill 节点 =====
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill
# ⚠️ vLLM 没有 --disaggregation-mode。NIXL 插件自动协调角色。

# ===== Decode 节点 =====
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 --port 8002 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 256 \
  --enforce-eager
# chunked-prefill 和 enforce-eager 是性能优化，不是角色声明。
```

### 3.4 验证 PD 分离

```bash
# ===== Prefill 节点日志 =====
grep "NIXL" vllm-prefill.log
# 期望: NIXL INFO: Connected to peer at 192.168.100.11

grep -i "gpudirect\|GDR" vllm-prefill.log
# 期望: GPUDirect RDMA is enabled
# 异常: GPUDirect RDMA disabled → 降级了

grep -i "transfer\|kv.*block" vllm-prefill.log | tail -5
# 期望: Transferred 42 KV blocks (672 MB) in 3.2 ms

# ===== Decode 节点日志 =====
grep "NIXL" vllm-decode.log
# 期望: NIXL INFO: Accepted connection from 192.168.100.10

grep -i "receive\|kv.*block" vllm-decode.log | tail -5
# 期望: Received 42 KV blocks from prefill node

# ===== 系统级验证 =====
# Prefill 节点看 PCIe Tx（传输时应有峰值）
nvidia-smi dmon -s pucv -d 1 -c 10
# 期望: Tx 列有几百 MB/s 峰值

# Decode 节点看显存增量
nvidia-smi --query-gpu=index,memory.used --format=csv
# 期望: 接收 KV Cache 后显存上升

# ===== API 测试 =====
curl http://192.168.100.10:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-72B-Instruct",
       "messages":[{"role":"user","content":"请分析以下文档..."}],
       "max_tokens":200}'
# 期望: 正常返回。同时 Prefill 日志有 transfer，Decode 日志有 decode step
```

### 3.5 PD 分离问题速查

| 现象 | 日志 | 排查 |
|------|------|------|
| 传输不工作 | `NCCL INFO NET/Socket` | NCCL 降级 TCP → `NCCL_IB_DISABLE=0` |
| 传输极慢 | `GPUDirect RDMA disabled` | `nvidia-smi nvlink --capabilities` |
| 连接超时 | `NIXL ERROR: Failed to connect` | `ibping <decode_ip>` |
| 中途断连 | `Connection closed` | `ibstat` 看 link down 次数 |
| Decode OOM | CUDA OOM | 降 `gpu-memory-utilization 0.85` |

---

## 四、一键部署脚本

```bash
#!/bin/bash
# vllm-deploy.sh [single|distributed|pd-prefill] [model] [port]
set -e
MODE=${1:-single}; MODEL=${2:-Qwen/Qwen2.5-7B-Instruct}; PORT=${3:-8000}

check() {
    nvidia-smi --query-gpu=index,name --format=csv | head -3
    python3 -c "import vllm; print('vLLM:', vllm.__version__)"
    echo "✅ OK"
}

case $MODE in
    single)
        check
        vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" &
        sleep 30
        curl -s http://localhost:"$PORT"/health && echo "✅"
        ;;
    distributed)
        check
        ray status || { echo "❌ Start Ray first"; exit 1; }
        vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
          --tensor-parallel-size 8 --pipeline-parallel-size 2 &
        ;;
    pd-prefill)
        check
        export NCCL_NET_PLUGIN=nixl NIXL_DEBUG=1
        vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
          --tensor-parallel-size 8 --max-model-len 32768 \
          --gpu-memory-utilization 0.85 --enable-chunked-prefill &
        ;;
    *) echo "Usage: $0 [single|distributed|pd-prefill]" ;;
esac
```

---

## 五、MoonCake vs NIXL 选型指南

### 一句话总结

**NIXL 是 NVIDIA 官方方案，一站式但绑定 NVIDIA 生态。MoonCake 是字节开源的 KV Cache 专用方案，异步流水线是核心竞争力。**

### 对比矩阵

| 维度 | NIXL | MoonCake |
|------|------|----------|
| **开发者** | NVIDIA 官方 | 字节跳动 KVCache.AI 开源 |
| **定位** | 通用 GPU 到 GPU RDMA 传输层 | **专为 LLM KV Cache 传输优化** |
| **协议层** | NCCL 插件 | 独立 Transfer Engine |
| **硬件依赖** | NVIDIA GPU + Mellanox IB | NVIDIA GPU + 支持 GPUDirect 的网卡（IB/RoCE 均可） |
| **异步流水线** | 基础支持 | **核心设计**：传输和计算可重叠 |
| **非 NVIDIA GPU** | 不支持 | 理论上可扩展（开源） |
| **安装复杂度** | 低：pip + NCCL 环境变量 | 中：需部署 Transfer Engine 后台进程 |
| **调试工具** | NCCL 生态（NCCL_DEBUG） | 独立日志 + Metrics |
| **vLLM 集成** | 原生支持（V1 默认） | 需 `VLLM_USE_MOONCAKE=1` 切换 |

### MoonCake 的核心优势

#### 1. 异步流水线：传输和计算重叠

```
NIXL（同步传输）:  Prefill 算完 → 传完 → 下一 chunk 才开始算（串行）
MoonCake（异步）:  Prefill 每算完一个 chunk 立即异步传输，不阻塞后续计算（并行）

对于长 prompt（> 8000 tokens），MoonCake 可降低端到端延迟 20-30%。
```

#### 2. 专用 KV Cache 优化

- Block 级别增量传输（只传新算的 blocks）
- Radix Tree 感知（跳过已在对端缓存的 blocks）
- 稀疏性压缩

#### 3. 硬件生态更灵活

支持 RoCE（以太网 RDMA），不严格绑定 Mellanox IB。

### NIXL 的核心优势

1. **NVIDIA 官方**，稳定性保证，新特性第一时间支持
2. **零额外进程**（NCCL 插件，无独立后台服务）
3. **部署最简单**（3 行环境变量）

### 选型决策树

```
① GPU 类型？
   ├─ NVIDIA H100/B100 → 往下
   └─ AMD/昇腾/其他 → MoonCake（NIXL 不支持）

② prompt 平均长度？
   ├─ < 2000 tokens → NIXL（传输量小，异步优势不明显）
   ├─ 2000-8000 → 两个都行，MoonCake 略优
   └─ > 8000 tokens → MoonCake（异步流水线优势显著）

③ 团队运维能力？
   ├─ 小团队 (<5人) → NIXL（部署简单，少一个故障点）
   └─ 大团队有专职 SRE → MoonCake（值得投入运维成本）

④ 延迟敏感度？
   ├─ 极致 TTFT 要求 → MoonCake
   └─ 吞吐优先 → 两者差距不大
```

### 生产环境建议

```
保守策略（大多数场景）: NIXL（官方支持 + 部署简单 + 覆盖广）
激进策略（长文本场景）: MoonCake（异步流水线 + KV 专用优化）

最佳实践: 先上 NIXL 快速落地 → 若 TTFT 偏高且定位到"传输慢" → 切换 MoonCake 对比 benchmark
```
