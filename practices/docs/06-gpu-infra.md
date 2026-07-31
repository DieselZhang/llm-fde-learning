# Day 6: GPU 基础设施与运维

> 学习目标：掌握 GPU 集群运维的核心技能 — NCCL 通信调优、RDMA 网络、GPU 监控、故障排查

## 🎯 本日学习内容

- [ ] NCCL 集合通信库原理与配置
- [ ] RDMA / InfiniBand / RoCEv2 网络基础
- [ ] GPU 监控：nvidia-smi, nvtop, DCGM
- [ ] 常见故障排查
- [ ] GPU 性能基准测试

## 📖 理论学习（3.5h）

### 1. NCCL 深度解析（1.5h）

**NCCL 是什么？**
NVIDIA Collective Communications Library，GPU 间高效通信库。
用于 TP（Tensor Parallel）中的 all-reduce、all-gather 等操作。

**NCCL 通信路径优先级**：
```
NVLink (机内, 最快, ~900 GB/s H100)
  → PCIe (机内, ~64 GB/s)
    → InfiniBand (跨机, ~400 Gb/s = ~50 GB/s)
      → RoCEv2 (跨机, ~200 Gb/s = ~25 GB/s)
        → TCP/Ethernet (跨机, ~25 Gb/s = ~3 GB/s, 最慢)
```

**关键环境变量速查表**：
```bash
# 调试级别
export NCCL_DEBUG=INFO              # 基本通信信息
export NCCL_DEBUG=TRACE             # 详细日志（生产慎用）
export NCCL_DEBUG_SUBSYS=INIT,GRAPH # 只看初始化和拓扑

# 网络选择
export NCCL_IB_DISABLE=0            # 启用 InfiniBand
export NCCL_NET_GDR_LEVEL=5         # GPUDirect RDMA 级别 (0-5)
export NCCL_SOCKET_IFNAME=eth0,ib0  # 使用的网络接口

# 性能调优
export NCCL_IB_TIMEOUT=22           # 通信超时 (默认 22)
export NCCL_IB_QPS_PER_CONNECTION=8 # 每连接 QP 数
export NCCL_IB_SPLIT_DATA_ON_QPS=1  # 数据分拆到多个 QP
export NCCL_MIN_NCHANNELS=32        # 最小通信通道数

# 拓扑检测
export NCCL_TOPO_DUMP_FILE=/tmp/topo.xml  # 导出拓扑
export NCCL_GRAPH_DUMP_FILE=/tmp/graph.dot # 导出通信图
```

**NCCL test 工具**：
```bash
# 安装
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make

# 基本带宽测试
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
# -b: 起始字节, -e: 结束字节, -f: 倍数, -g: GPU 数

# 跨节点测试（多机）
mpirun -host node1:8,node2:8 \
  ./build/all_reduce_perf -b 1M -e 1G -f 2 -g 16
```

### 2. RDMA / InfiniBand 网络（1h）

**三种 RDMA 方案对比**：

| 方案 | 带宽 | 延迟 | 成本 | 适用场景 |
|------|------|------|------|---------|
| InfiniBand | 400-800 Gb/s | <1μs | 高 | 大型训练集群 |
| RoCEv2 | 100-400 Gb/s | 1-3μs | 中 | 推理集群 |
| iWARP | 100-200 Gb/s | 3-5μs | 低 | 存量 TCP 网络 |

**检查 InfiniBand 设备**：
```bash
# 查看 IB 设备
ibstat
ibv_devinfo

# 查看 IPoIB 地址
ip addr show ib0

# 测试 IB 带宽
ib_write_bw -a -d mlx5_0    # 服务端
ib_write_bw -a -d mlx5_0 192.168.1.10  # 客户端
```

**RoCEv2 配置**：
```bash
# 确认网卡支持
cma_avail -d mlx5_0

# 检查 RoCE 状态
rdma link show

# 配置 PFC（优先级流控，RoCEv2 需要无损网络）
mlnx_qos -i eth0 --pfc 0,0,0,1,0,0,0,0
```

### 3. GPU 监控体系（1h）

**三层监控架构**：
```
应用层 → DCGM Exporter → Prometheus → Grafana
                ↓
          nvidia-smi / nvtop (交互式调试)
```

**关键监控指标**：
| 指标 | 命令 | 正常范围 | 告警阈值 |
|------|------|---------|---------|
| GPU 利用率 | `nvidia-smi --query-gpu=utilization.gpu` | 80-100% | < 30%（利用率不足） |
| 显存占用 | `nvidia-smi --query-gpu=memory.used` | < 90% | > 95% |
| GPU 温度 | `nvidia-smi --query-gpu=temperature.gpu` | < 80°C | > 85°C |
| 显存带宽 | `nvidia-smi --query-gpu=memory.used` | 利用率高 | — |
| PCIe 带宽 | `nvidia-smi pcie --gen` | Gen4/5 | Gen 降级 |
| 功率 | `nvidia-smi --query-gpu=power.draw` | < 额定 | > 110% |
| ECC 错误 | `nvidia-smi --query-gpu=ecc.errors` | 0 | > 0 |

## 🛠️ 实操练习（4h）

### 练习 1：NCCL 环境检查与调试

```bash
# 1. 检查 NCCL 版本
python -c "import torch; print(torch.cuda.nccl.version())"

# 2. 导出拓扑信息
export NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml
python -c "
import torch
import torch.distributed as dist
dist.init_process_group('nccl', world_size=1, rank=0)
"
cat /tmp/nccl_topo.xml

# 3. NCCL 带宽测试
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make -j

echo "=== 单卡内带宽测试 ==="
./build/all_reduce_perf -b 128M -e 128M -g 1

echo "=== 多卡带宽测试 ==="
./build/all_reduce_perf -b 128M -e 128M -g $(nvidia-smi -L | wc -l)
```

### 练习 2：GPU 监控脚本

参考脚本 [scripts/08-gpu-monitoring.sh](../scripts/08-gpu-monitoring.sh)。

```bash
# 实时 GPU 监控
watch -n 1 nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv

# 详细的 GPU 信息
nvidia-smi -q

# 安装 nvtop（交互式监控）
# macOS: brew install nvtop
# Linux: apt install nvtop
nvtop
```

### 练习 3：DCGM 部署与使用

```bash
# 安装 DCGM
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y datacenter-gpu-manager

# 启动 DCGM
sudo systemctl start nvidia-dcgm
sudo systemctl enable nvidia-dcgm

# 健康检查
dcgmi health -g 0

# 诊断
dcgmi diag -r 1  # 基本诊断
dcgmi diag -r 3  # 全面诊断（耗时较长）

# 实时监控
dcgmi mon -g 0 -e 100,101,102,103,104,200,201,203  # 常用指标
```

### 练习 4：GPU 集群网络诊断

```bash
# 1. 检查跨节点 GPU 通信
# 在节点 A 上
python -c "
import torch
import torch.distributed as dist

# 模拟 TP 通信
dist.init_process_group('nccl', init_method='tcp://192.168.1.10:23456',
                       world_size=2, rank=0)
tensor = torch.randn(1024, 1024).cuda()
dist.all_reduce(tensor)
print(f'Rank 0 - All-reduce 成功, tensor shape: {tensor.shape}')
dist.destroy_process_group()
"

# 在节点 B 上
python -c "
import torch
import torch.distributed as dist

dist.init_process_group('nccl', init_method='tcp://192.168.1.10:23456',
                       world_size=2, rank=1)
tensor = torch.randn(1024, 1024).cuda()
dist.all_reduce(tensor)
print(f'Rank 1 - All-reduce 成功')
dist.destroy_process_group()
"
```

### 练习 5：故障排查模拟

**常见问题 1：CUDA Out of Memory**
```bash
# 排查步骤
nvidia-smi  # 查看当前显存占用

# 查看 KV Cache block 数 — 如果太少说明显存不足
vllm serve ... | grep "GPU blocks"

# 调优
# 方案 1: 减小 max-model-len
# 方案 2: 减小 gpu-memory-utilization
# 方案 3: 增加 TP 并行度
```

**常见问题 2：NCCL 通信超时**
```bash
# 错误: NCCL WARN Cuda failure 'driver reset' / 'unhandled cuda error'

# 排查步骤
# 1. 检查 NCCL 日志
export NCCL_DEBUG=INFO
# 2. 检查网络连通性
ping -c 3 worker-node-ip
ibping -c 3 worker-node-ip  # InfiniBand 连通性
# 3. 检查网卡配置
ibstat | grep state  # "ACTIVE" 才是正常的
# 4. 增加超时
export NCCL_IB_TIMEOUT=24
export NCCL_IB_RETRY_CNT=7
```

**常见问题 3：模型加载失败**
```bash
# 错误: ValueError: The model's max seq len is too large

# 排查
# 1. 检查模型配置
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print(f'max_position_embeddings: {config.max_position_embeddings}')
"
# 2. 设置合适的 max-model-len
vllm serve ... --max-model-len 4096
```

**常见问题 4：GPU 间通信性能瓶颈**
```bash
# 症状：TP 速度远低于预期
# 排查：检查 NVLink/PCIe 拓扑
nvidia-smi topo -m

# 期望输出示例（NVLink 连接为良好）：
#        GPU0   GPU1   ...
# GPU0   X      NV2    ← NVLink Gen2
# GPU1   NV2    X
# GPU2   NV2    NV2

# 如果看到 PIX / PHB 说明走 PCIe，性能会差很多
```

## 📝 学习日志

在 `daily-logs/day-06.md` 中记录：
1. NCCL 带宽测试结果
2. DCGM 诊断输出
3. 你的 GPU 拓扑结构
4. 排查了哪些故障、如何解决的
5. 一张监控 Dashboard 截图（如有）

## 🔗 参考资料

- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL 环境变量参考](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA DCGM](https://developer.nvidia.cn/dcgm)
- [AI 集群 InfiniBand 详解 (掘金)](https://juejin.cn/post/7314941294873362495)
- [NCCL + RDMA 配置经验 (知乎)](https://zhuanlan.zhihu.com/p/9376124792)
- [NVLink、RDMA、NCCL 配置指南](https://www.dong-blog.fun/post/1687)
- [GPU 监控最佳实践 (InfoQ)](https://xie.infoq.cn/article/1143654b5b7a4bddd2aef0313)
- [使用 DCGM 进行 GPU 性能分析 (阿里云)](https://help.aliyun.com/zh/ack/cloud-native-ai-suite/use-cases/performance-analysis-of-gpu-using-dcgm)
