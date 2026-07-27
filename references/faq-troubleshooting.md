# FDE 常见问题与故障排查手册

> 收集整理了推理部署中最常见的 15 个问题和对应解决方案

## 目录
1. [显存相关](#1-显存相关)
2. [NCCL 通信](#2-nccl-通信)
3. [部署启动](#3-部署启动)
4. [性能相关](#4-性能相关)
5. [网络相关](#5-网络相关)

---

## 1. 显存相关

### Q1: CUDA Out of Memory 怎么办？

**现象**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```

**排查步骤**：
```bash
# 1. 查看显存分配
nvidia-smi

# 2. 检查 vLLM 的 KV Cache block 数
# 启动日志中搜索: "# GPU blocks:"
vllm serve ... 2>&1 | grep "GPU blocks"

# 如果 GPU blocks 为 0 或很少，说明显存不足
```

**解决方案**（按优先级）：
1. 减小 `--max-model-len`（最有效）
2. 调低 `--gpu-memory-utilization`（如 0.85 → 0.7）
3. 增加 TP 并行度（如 TP=1 → TP=2）
4. 使用量化（`--quantization fp8` 或 `awq`）
5. 减少 `--max-num-seqs`

### Q2: vLLM 启动时提示 "The model's max seq len is too large"

**原因**：模型配置中的 `max_position_embeddings` 过大，KV Cache 放不下

**解决**：
```bash
# 指定 max-model-len（不要超过模型实际需要的长度）
vllm serve ... --max-model-len 4096
```

### Q3: PagedAttention block 数为什么这么少？

```bash
# 打印 block 数
vllm serve ... 2>&1 | grep "# GPU blocks"

# 如果 < 500 说明显存紧张
# block 数 = 可用显存 / (block_size × num_layers × num_heads × d_head × 2)
```

---

## 2. NCCL 通信

### Q4: NCCL 连接超时

**错误**：
```
NCCL WARN NET/Socket : Connect to 192.168.1.11:12345 failed : Connection refused
NCCL WARN timeout 30s
```

**排查**：
```bash
# 1. 测试网络连通性
ping -c 3 <worker_ip>

# 2. 检查端口是否开放（需要所有节点双向通信）
nc -zv <worker_ip> 6379  # Ray 端口
nc -zv <worker_ip> 8076  # Object Manager

# 3. 如果跨节点通信失败，检查防火墙和安全组
sudo ufw status
# iptables -L

# 4. 增加 NCCL 超时
export NCCL_IB_TIMEOUT=24
export NCCL_CONNECT_TIMEOUT=30
```

### Q5: NCCL 报 "handshake failed" 或 "unhandled cuda error"

**原因**：GPU 驱动问题或 GPU 掉卡

**解决**：
```bash
# 1. 检查 GPU 可用性
nvidia-smi
# 如果看不到所有 GPU → 重启机器或重新安装驱动

# 2. 检查 GPU 健康状况
dcgmi diag -r 2

# 3. 重置 GPU
nvidia-smi -r
```

### Q6: 如何确认 NCCL 走了 InfiniBand？

```bash
# 设置 DEBUG 日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# 启动推理，在日志中搜索：
# "[send] via NET/IB" → 走 InfiniBand (✅ 好)
# "[send] via NET/Socket" → 走 TCP (❌ 慢)
# "NCCL_INFO NET/IB : Using IB device mlx5_0" → IB 设备已识别
```

### Q7: 多机 TP 性能远低于预期

**排查**：
```bash
# 1. 检查 NVLink（机内）
nvidia-smi topo -m
# 应该看到 NV 连接，如果全是 PIX/PHB 说明走 PCIe

# 2. 检查跨机带宽
ib_write_bw -a -d mlx5_0  # 服务端
ib_write_bw -a -d mlx5_0 <server_ip>  # 客户端
# 预期 > 350 Gb/s (HDR)

# 3. 检查 NCCL 通道数
export NCCL_MIN_NCHANNELS=32  # H100 推荐
export NCCL_ALGO=Ring          # 有些场景 Tree 更快
```

---

## 3. 部署启动

### Q8: Docker 容器内看不到 GPU

**原因**：未正确安装 NVIDIA Container Toolkit

**解决**：
```bash
# 安装
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 重启 Docker
sudo systemctl restart docker

# 测试
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Q9: vLLM 启动报 "ImportError"

**原因**：Python 环境不兼容或 CUDA 版本不匹配

**解决**：
```bash
# 最好的方式是用 Docker（自带完整环境）
# 如果必须 pip 安装，确保 CUDA 版本匹配
python -c "import torch; print(torch.version.cuda)"
# 应 >= 11.8

# 或在 Docker 中指定版本标签
docker pull vllm/vllm-openai:v0.6.0  # 指定版本
```

### Q10: SGLang 启动报端口占用

```bash
# 检查端口
lsof -i :30000

# 杀掉占用进程
kill -9 <PID>
# 或换个端口
python -m sglang.launch_server ... --port 30001
```

---

## 4. 性能相关

### Q11: TTFT（首 token 延迟）过高

**原因**：
1. Prefill 阶段处理长 prompt
2. 前一个请求还在 Decode，阻塞了 Prefill
3. GPU 利用率不足

**解决**：
```bash
# 1. 启用前缀缓存（共享 prompt 时效果显著）
--enable-prefix-caching

# 2. 减小 max-num-batched-tokens
--max-num-batched-tokens 2048  # 防止单次 batch 太大

# 3. 使用 PD 分离（彻底解决 Prefill/Decode 资源竞争）
```

### Q12: Throughput 上不去

**排查**：
```bash
# 1. 检查 GPU 利用率
nvidia-smi
# 如果 GPU-Util < 90% → 可能是调度瓶颈

# 2. 检查 max-num-seqs 是否限制
--max-num-seqs 512  # 调大

# 3. 检查是否被 kv cache 限制（block 数太少）
# 增加 gpu-memory-utilization
--gpu-memory-utilization 0.95

# 4. 检查模型是否太大了，batch 无法做大
# 考虑量化或增加 TP
```

### Q13: FP8 模型部署时性能不如预期

```bash
# 确保 GPU 支持 FP8（H100/H200/B200）
# 检查 vLLM 日志中的 dtype
# --dtype auto 会自动选择

# 如果 FP8 性能还不如 FP16 → 检查是否有 FP8 兼容性问题
# 尝试 kv-cache-dtype=fp8 单独对 KV Cache 使用 FP8
vllm serve ... --kv-cache-dtype fp8
```

---

## 5. 网络相关

### Q14: 跨节点 GPU 间通信慢

**诊断工具**：
```bash
# 1. NCCL 带宽测试
# 机内：NVLink 应 > 400 GB/s (H100)
# 跨机：IB 应 > 45 GB/s (HDR400)

# all_reduce_perf 用法
./build/all_reduce_perf -b 128M -e 128M -g 2

# 2. IB 带宽测试
ib_write_bw -a -d mlx5_0  # > 45 GB/s 正常
```

**常见原因**：
- InfiniBand 降速（检查 cable 和 port status）
- RoCEv2 未配置 PFC（丢包导致重传）
- NCCL 走了 TCP 而非 RDMA（检查 NCCL_IB_DISABLE）

### Q15: Ray 集群节点无法互相发现

**排查**：
```bash
# 1. 确保所有节点能通过 hostname 通信
# 在 /etc/hosts 中添加：
192.168.1.10 node1
192.168.1.11 node2

# 2. 检查 Ray 状态
docker exec ray-head ray status

# 3. 确保端口开放（6379 和 8076）
# 4. 尝试用 --node-ip-address 显式指定 IP
ray start --head --node-ip-address=192.168.1.10
```

---

> 💡 **FDE 黄金法则**：遇到问题先看日志。NCCL_DEBUG=INFO + vLLM/SGLang 的 `--enable-metrics` 能解决 80% 的问题。
