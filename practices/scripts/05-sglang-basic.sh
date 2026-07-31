#!/bin/bash
# ============================================================
# Script: SGLang 基本启动脚本
# 用途：演示 SGLang 在不同配置下的启动
# ============================================================
set -e

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
TP=${2:-1}
PORT=${3:-30000}
HOST=${4:-"0.0.0.0"}

echo "=== SGLang 服务启动 ==="
echo "Model: $MODEL"
echo "TP: $TP"
echo "Port: $PORT"

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Available GPUs: $GPU_COUNT"

if [ "$TP" -gt "$GPU_COUNT" ]; then
    echo "ERROR: TP=$TP 需要 $TP 张卡，但只有 $GPU_COUNT 张"
    exit 1
fi

# Docker 方式启动
docker run --gpus all \
  --name "sglang-tp${TP}" \
  -p ${PORT}:${PORT} \
  --shm-size=32g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path $MODEL \
    --tp $TP \
    --host $HOST \
    --port $PORT \
    --enable-metrics

echo "SGLang 服务已启动在 http://${HOST}:${PORT}"
echo ""
echo "测试命令:"
echo "  curl http://localhost:${PORT}/v1/models"
echo ""
echo "Chat Completion:"
cat << EOF
  curl http://localhost:${PORT}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
EOF
