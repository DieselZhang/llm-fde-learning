#!/bin/bash
# ============================================================
# Script: SGLang PD 分离启动示例
# 用途：单机演示 Prefill/Decode 分离（需要 4+ 卡 GPU）
# ============================================================
set -e

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
TP=${2:-2}
SCHEDULER_PORT=${3:-30000}

echo "=== SGLang PD 分离部署（单机）==="
echo "Model: $MODEL"
echo "TP: $TP"
echo "Scheduler Port: $SCHEDULER_PORT"

# 检查 GPU
GPU_COUNT=$(nvidia-smi -L | wc -l)
NEED_GPU=$((TP * 2))  # 需要 TP×2 卡（Prefill TP + Decode TP）
echo "Available GPUs: $GPU_COUNT, Need: $NEED_GPU"

if [ "$NEED_GPU" -gt "$GPU_COUNT" ]; then
    echo "WARNING: GPU 数量不足，降低 TP 或使用更少的卡"
    exit 1
fi

# Step 1: 启动 Prefill
PREFILL_PORT=30001
BOOTSTRAP_PORT=34001

echo ""
echo ">>> Step 1: 启动 Prefill (GPU 0-$(($TP-1)), Port $PREFILL_PORT)"
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP-1))) python3 -m sglang.launch_server \
  --model-path $MODEL \
  --tp $TP \
  --disaggregation-mode prefill \
  --port $PREFILL_PORT \
  --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
  --host 0.0.0.0 \
  --enable-metrics &
PREFILL_PID=$!
echo "  PID: $PREFILL_PID"

sleep 10

# Step 2: 启动 Decode
DECODE_PORT=30002

echo ""
echo ">>> Step 2: 启动 Decode (GPU $TP-$(($TP*2-1)), Port $DECODE_PORT)"
CUDA_VISIBLE_DEVICES=$(seq -s, $TP $((TP*2-1))) python3 -m sglang.launch_server \
  --model-path $MODEL \
  --tp $TP \
  --disaggregation-mode decode \
  --port $DECODE_PORT \
  --host 0.0.0.0 \
  --enable-metrics &
DECODE_PID=$!
echo "  PID: $DECODE_PID"

sleep 10

# Step 3: 启动 Scheduler
echo ""
echo ">>> Step 3: 启动 Scheduler (Port $SCHEDULER_PORT)"
python3 -m sglang.srt.disaggregation.mini_lb \
  --prefill http://localhost:${PREFILL_PORT} \
  --decode http://localhost:${DECODE_PORT} \
  --host 0.0.0.0 \
  --port $SCHEDULER_PORT &
SCHEDULER_PID=$!
echo "  PID: $SCHEDULER_PID"

sleep 5

echo ""
echo "=== PD 分离部署完成 ==="
echo "API: http://localhost:${SCHEDULER_PORT}/v1/chat/completions"
echo ""
echo "测试命令:"
cat << EOF
curl http://localhost:${SCHEDULER_PORT}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
EOF
echo ""
echo "停止所有进程: kill $PREFILL_PID $DECODE_PID $SCHEDULER_PID"
