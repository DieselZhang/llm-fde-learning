#!/bin/bash
# ============================================================
# Script: SGLang 多机多节点部署
# 用途：跨两台机器部署 SGLang + PD 分离
# ============================================================
set -e

MODE=${1:-"prefill"}
HEAD_IP=${2:-"192.168.1.10"}
MODEL=${3:-"/models/Qwen2.5-72B-Instruct"}
TP=${4:-8}

echo "=== SGLang 多机部署 ==="
echo "Mode: $MODE"
echo "Head IP: $HEAD_IP"

if [ "$MODE" = "prefill" ]; then
    echo ">>> 启动 Prefill 节点..."
    docker run --gpus all --network host --shm-size=32g \
      --name sglang-prefill \
      -v $MODEL:$MODEL:ro \
      lmsysorg/sglang:latest \
      python3 -m sglang.launch_server \
        --model-path $MODEL \
        --tp $TP \
        --host 0.0.0.0 \
        --port 30001 \
        --enable-metrics

elif [ "$MODE" = "decode" ]; then
    echo ">>> 启动 Decode 节点..."
    docker run --gpus all --network host --shm-size=32g \
      --name sglang-decode \
      -v $MODEL:$MODEL:ro \
      lmsysorg/sglang:latest \
      python3 -m sglang.launch_server \
        --model-path $MODEL \
        --tp $TP \
        --host 0.0.0.0 \
        --port 30002 \
        --enable-metrics

elif [ "$MODE" = "scheduler" ]; then
    echo ">>> 启动 Scheduler（负载均衡）..."
    docker run --network host \
      --name sglang-scheduler \
      lmsysorg/sglang:latest \
      python3 -m sglang.srt.disaggregation.mini_lb \
        --prefill http://${HEAD_IP}:30001 \
        --decode http://${HEAD_IP}:30002 \
        --host 0.0.0.0 \
        --port 30000

else
    echo "使用方法: $0 {prefill|decode|scheduler} [head_ip] [model_path] [tp]"
    exit 1
fi
