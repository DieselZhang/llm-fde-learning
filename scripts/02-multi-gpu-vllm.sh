#!/bin/bash
# ============================================================
# Script: vLLM 多GPU部署脚本
# 用途：演示 vLLM 在不同 TP/PP 配置下的启动
# ============================================================
set -e

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
TP_SIZE=${2:-1}
PP_SIZE=${3:-1}
PORT=${4:-8000}

echo "=== vLLM 多GPU部署 ==="
echo "Model: $MODEL"
echo "TP: $TP_SIZE, PP: $PP_SIZE"
echo "Port: $PORT"

# 检查 GPU 可用数
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $GPU_COUNT"

REQUIRED_GPUS=$((TP_SIZE * PP_SIZE))
if [ "$REQUIRED_GPUS" -gt "$GPU_COUNT" ]; then
    echo "ERROR: Need $REQUIRED_GPUS GPUs but only $GPU_COUNT available"
    exit 1
fi

# 启动 vLLM 服务
docker run --gpus all -d \
  --name "vllm-tp${TP_SIZE}-pp${PP_SIZE}" \
  -p ${PORT}:8000 \
  --shm-size=32g \
  vllm/vllm-openai:latest \
  --model $MODEL \
  --tensor-parallel-size $TP_SIZE \
  --pipeline-parallel-size $PP_SIZE \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching

echo "vLLM 服务已启动在 http://localhost:${PORT}"
echo "运行以下命令测试："
echo "  curl http://localhost:${PORT}/v1/models"
echo ""
echo "停止服务："
echo "  docker stop vllm-tp${TP_SIZE}-pp${PP_SIZE} && docker rm vllm-tp${TP_SIZE}-pp${PP_SIZE}"
