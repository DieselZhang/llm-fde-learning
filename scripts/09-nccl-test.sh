#!/bin/bash
# ============================================================
# Script: NCCL 测试与诊断
# 用途：快速检测 NCCL 通信性能
# ============================================================
set -e

MODE=${1:-"local"}   # local 或 cross-node
PEER_IP=${2:-""}

echo "=== NCCL 测试 ==="
echo "Mode: $MODE"

# 1. 检查 NCCL 版本
echo ""
echo ">>> NCCL 版本"
python3 -c "import torch; print(f'NCCL: {\".\".join(map(str, torch.cuda.nccl.version()))}')"

# 2. 安装 nccl-tests（如果未安装）
if [ ! -f "./nccl-tests/build/all_reduce_perf" ]; then
    echo ""
    echo ">>> 安装 nccl-tests..."
    git clone https://github.com/NVIDIA/nccl-tests.git 2>/dev/null || true
    cd nccl-tests && make -j4 2>/dev/null || echo "跳过编译（可能缺少 MPI）"
    cd ..
fi

# 3. 机内测试
echo ""
echo ">>> 机内 All-reduce 带宽测试"
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPU 数量: $GPU_COUNT"

if [ -f "./nccl-tests/build/all_reduce_perf" ]; then
    ./nccl-tests/build/all_reduce_perf -b 128M -e 128M -g $GPU_COUNT -w 10 -n 5
else
    echo "nccl-tests 不可用，使用 PyTorch 简单测试"
    python3 -c "
import torch
import time

# 简单带宽测试
tensor = torch.randn(1024, 1024, 1024).cuda()  # 4GB tensor
torch.cuda.synchronize()

start = time.time()
for _ in range(10):
    tensor.mul_(1.0)
torch.cuda.synchronize()
elapsed = time.time() - start

bandwidth = (4 * 10) / elapsed  # GB/s
print(f'简单带宽测试: {bandwidth:.1f} GB/s (HBM 带宽)')
"
fi

# 4. 跨节点测试
if [ "$MODE" = "cross-node" ] && [ -n "$PEER_IP" ]; then
    echo ""
    echo ">>> 跨节点通信测试 (节点: $PEER_IP)"
    
    # 简单 PyTorch 跨节点通信测试
    python3 -c "
import torch
import torch.distributed as dist
import os

# 用环境变量传递 rank
rank = int(os.environ.get('RANK', 0))
world_size = 2

if rank == 0:
    init_method = f'tcp://{PEER_IP}:23456'
else:
    init_method = f'tcp://{PEER_IP}:23456'

print(f'初始化分布式... rank={rank}, world={world_size}')
dist.init_process_group('nccl', init_method=init_method,
                       world_size=world_size, rank=rank)

# 创建测试 tensor（1GB）
tensor = torch.randn(256, 1024, 1024).cuda()
torch.cuda.synchronize()

import time
start = time.time()
dist.all_reduce(tensor)
torch.cuda.synchronize()
elapsed = time.time() - start

# all_reduce 实际传输 2× 数据量（reduce + broadcast）
data_gb = tensor.numel() * tensor.element_size() / (1024**3) * 2
bandwidth = data_gb / elapsed

print(f'All-reduce: {data_gb:.1f} GB in {elapsed*1000:.0f} ms')
print(f'带宽: {bandwidth:.1f} GB/s')
dist.destroy_process_group()
"
fi

echo ""
echo "=== NCCL 测试完成 ==="
echo "检查 NCCL_DEBUG=INFO 设置以获取详细日志"
