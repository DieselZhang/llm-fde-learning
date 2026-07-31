#!/bin/bash
# ============================================================
# Script: GPU 监控与诊断脚本
# 用途：收集 GPU 状态、NCCL 信息、生成监控报告
# ============================================================
set -e

OUTPUT_DIR="./gpu-reports/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "=== GPU 监控诊断报告 ==="
echo "时间: $(date)"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 1. 基本信息
echo ">>> 1. GPU 基本信息" | tee $OUTPUT_DIR/01-gpu-info.txt
nvidia-smi --query-gpu=index,name,driver_version,pcie.link.gen.current,pcie.link.width.current,memory.total,temperature.gpu,power.limit \
  --format=csv >> $OUTPUT_DIR/01-gpu-info.txt
echo "" | tee -a $OUTPUT_DIR/01-gpu-info.txt

# 2. GPU 拓扑
echo ">>> 2. GPU 拓扑 (NVLink / PCIe)" | tee $OUTPUT_DIR/02-topology.txt
nvidia-smi topo -m >> $OUTPUT_DIR/02-topology.txt

# 3. 当前进程占用
echo ">>> 3. GPU 进程占用" | tee $OUTPUT_DIR/03-processes.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> $OUTPUT_DIR/03-processes.txt

# 4. NCCL 环境变量
echo ">>> 4. NCCL 环境变量" | tee $OUTPUT_DIR/04-nccl-env.txt
env | grep -i nccl >> $OUTPUT_DIR/04-nccl-env.txt 2>/dev/null || echo "（无 NCCL 环境变量设置）" >> $OUTPUT_DIR/04-nccl-env.txt

# 5. DCGM 健康检查（如果安装了）
echo ">>> 5. DCGM 健康检查" | tee $OUTPUT_DIR/05-dcgm.txt
if command -v dcgmi &> /dev/null; then
    dcgmi health -g 0 >> $OUTPUT_DIR/05-dcgm.txt 2>&1
else
    echo "DCGM 未安装，跳过" >> $OUTPUT_DIR/05-dcgm.txt
fi

# 6. 网络接口信息
echo ">>> 6. 网络接口信息" | tee $OUTPUT_DIR/06-network.txt
ip addr show | grep -E "^(eth|ib|eno|enp)" -A2 >> $OUTPUT_DIR/06-network.txt 2>/dev/null || echo "未找到网络接口" >> $OUTPUT_DIR/06-network.txt

# 7. InfiniBand 状态
echo ">>> 7. InfiniBand 设备状态" | tee $OUTPUT_DIR/07-ib.txt
if command -v ibstat &> /dev/null; then
    ibstat >> $OUTPUT_DIR/07-ib.txt 2>&1
else
    echo "InfiniBand 未安装，跳过" >> $OUTPUT_DIR/07-ib.txt
fi

# 8. 汇总报表
echo ""
echo "=== 汇总 ===" | tee $OUTPUT_DIR/summary.txt
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPU 数量: $GPU_COUNT" | tee -a $OUTPUT_DIR/summary.txt

# 显存总量
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "单卡显存: $TOTAL_MEM" | tee -a $OUTPUT_DIR/summary.txt

# 是否存在 NVLink
if nvidia-smi topo -m | grep -q "NV"; then
    echo "NVLink: ✅ 可用" | tee -a $OUTPUT_DIR/summary.txt
else
    echo "NVLink: ❌ 不可用（仅 PCIe）" | tee -a $OUTPUT_DIR/summary.txt
fi

# 是否存在 InfiniBand
if command -v ibstat &> /dev/null && ibstat &>/dev/null; then
    echo "InfiniBand: ✅ 可用" | tee -a $OUTPUT_DIR/summary.txt
else
    echo "InfiniBand: ❌ 不可用" | tee -a $OUTPUT_DIR/summary.txt
fi

# CUDA 版本
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "未知")
echo "CUDA 版本: $CUDA_VER" | tee -a $OUTPUT_DIR/summary.txt

echo ""
echo "报告已生成: $OUTPUT_DIR"
