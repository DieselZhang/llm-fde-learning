#!/bin/bash
# ============================================================
# Script: Ray + vLLM 多机多节点部署
# 用途：跨两台机器部署分布式推理
# 使用方法：
#   在 Head Node:  ./04-multi-node-ray-setup.sh head <head_ip>
#   在 Worker Node: ./04-multi-node-ray-setup.sh worker <head_ip>
# ============================================================
set -e

MODE=${1:-"head"}           # head 或 worker
HEAD_IP=${2:-"192.168.1.10"}
MODEL_PATH=${3:-"/models/Qwen2.5-72B-Instruct"}
TP_SIZE=${4:-8}
PP_SIZE=${5:-2}

RAY_PORT=6379
OBJECT_MANAGER_PORT=8076

echo "=== Ray + vLLM 多机部署 ==="
echo "Mode: $MODE"
echo "Head IP: $HEAD_IP"

if [ "$MODE" = "head" ]; then
    echo ">>> 启动 Head Node..."
    
    # Step 1: 启动 Ray Head
    docker run --gpus all --network host --shm-size=32g \
      --name ray-head \
      -v ${MODEL_PATH}:${MODEL_PATH}:ro \
      vllm/vllm-openai:latest \
      bash -c "
        ray start --head \
          --node-ip-address=${HEAD_IP} \
          --port=${RAY_PORT} \
          --object-manager-port=${OBJECT_MANAGER_PORT} \
          --num-gpus=$(nvidia-smi -L | wc -l)
        
        echo '=== Ray 集群状态 ==='
        ray status
        echo '=== Ray 集群节点 ==='
        ray list nodes
        echo '=== 等待 Worker 连接... ==='
        sleep infinity
      "
      
    echo "Head 启动完成！在 Worker 运行:"
    echo "  ./04-multi-node-ray-setup.sh worker ${HEAD_IP}"
    
elif [ "$MODE" = "worker" ]; then
    echo ">>> 启动 Worker Node, 连接到 ${HEAD_IP}..."
    
    docker run --gpus all --network host --shm-size=32g \
      --name ray-worker \
      -v ${MODEL_PATH}:${MODEL_PATH}:ro \
      vllm/vllm-openai:latest \
      bash -c "
        ray start --address=${HEAD_IP}:${RAY_PORT} \
          --object-manager-port=${OBJECT_MANAGER_PORT} \
          --num-gpus=$(nvidia-smi -L | wc -l)
        
        echo '=== 连接成功！==='
        sleep infinity
      "
      
    echo "Worker 已连接到 ${HEAD_IP}"
    
else
    echo "使用方法: $0 {head|worker} [head_ip] [model_path] [tp] [pp]"
    exit 1
fi
