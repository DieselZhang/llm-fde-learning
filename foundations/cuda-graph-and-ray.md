# CUDA Graph 与 vLLM+Ray 集成笔记

---

## 第一部分：CUDA Graph

### 问题：GPU Kernel Launch Overhead

每次 GPU 计算前，CPU 需要逐个下发 kernel 指令，每个 launch 耗时 5-20 微秒：

```
一次 forward pass:
  80 层 × 每层 ~15 个 kernel × 10μs = 12000 μs = 12 ms

Decode 时每次只算 1 个 token:
  实际计算: ~5ms
  CPU launch 开销: ~12ms  ← CPU 开销比计算还长
```

### CUDA Graph 原理

把整个 forward pass 的 kernel 调用序列"录制"成一张图，之后一次 launch 跑完全部：

```
录制阶段 (启动时):
  对不同 batch_size 各录一个 graph:
    batch_size=1  → graph_1
    batch_size=2  → graph_2
    ...
    batch_size=256 → graph_256

执行阶段 (运行时):
  CPU: launch(graph_N) → GPU: 全自动执行完成
  原来需要 15ms 的 CPU 开销 → ~0.02ms
```

### 收益

```
Llama-3-70B, TP=2, batch_size=8:

无 CUDA Graph: ~23ms per token → ~43 tokens/s
有 CUDA Graph: ~8ms per token  → ~125 tokens/s
吞吐提升约 3x
```

### 为什么 Prefill 用不上

Prefill 每次处理的 token 数和 batch 组合都不同，无法"录一个模板重复用"。Decode 每次固定处理"每个请求的最后 1 个 token"，模式可预测。

### 代价

- 启动时间增加 10-60s（录制 graph）
- 额外显存占用 ~0.5-3 GB（存 graph）
- `max-num-batched-tokens` 越大 → 需要录制的 graph 越多 → 启动越慢

### 相关参数

- `--enforce-eager`: 禁用 CUDA Graph（调试时用，无加速）
- graph 录制失败时自动回退到 eager 模式，不影响正常运行

---

## 第二部分：vLLM + Ray 集成

### vLLM 如何发现 Ray？

**不需要**在命令行参数中指定 Ray 地址。vLLM 通过环境感知自动判断：

```
vllm serve ... 启动时内部逻辑:

if 检测到 RAY_ADDRESS 环境变量 or 本地已有 Ray 进程:
    → ray.init(address="auto") 自动连接集群
    → GPU Workers 通过 Ray Actor 跨节点分布
else:
    → 使用本地 Python multiprocessing
    → GPU Workers 作为本地子进程
```

`ray start --head` 会自动设置 `RAY_ADDRESS` 环境变量。

### 完整多机部署步骤

```bash
# 头节点
ray start --head --port=6379

# 每台 Worker 节点
ray start --address='192.168.1.10:6379'

# 头节点验证
ray status

# 启动 vLLM（不传 Ray 地址！自动连接）
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2
```

### Ray 在 vLLM 中的角色：分布式进程管理器

```
没有 Ray (单机):
  vLLM → Python multiprocessing → N 个本地子进程，每个绑一张 GPU

有 Ray (多机):
  vLLM → ray.init(address="auto") → 创建 Ray Actors
  每个 Actor 用 @ray.remote(num_gpus=1) 声明需要 1 张 GPU
  Ray 自动决定 rank=0~7 放机器A，rank=8~15 放机器B
  → 不需要手动指定每个 Worker 的位置
```

### 常见踩坑

| 现象 | 原因 | 解决 |
|------|------|------|
| Worker 节点不显示 GPU | 忘了 `ray start` | `ray status` 看 Node count |
| vLLM 只用头节点 GPU | Worker 未连接 | 每节点 `nvidia-smi` 验证 |
| Ray 版本不一致报错 | 节点版本不同 | `pip install ray==2.9.0` 统一 |
| 头节点重启后 Worker 掉线 | 不会自动重连 | 每节点 `ray stop && ray start --address=...` |
| NCCL 跨节点通信超时 | 网卡不对 | `NCCL_SOCKET_IFNAME=ib0` |

### 排查口诀

`ray status` → 每节点 `nvidia-smi` → `NCCL_DEBUG=INFO` → vLLM 启动日志
