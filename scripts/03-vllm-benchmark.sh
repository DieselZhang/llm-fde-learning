#!/bin/bash
# ============================================================
# Script: vLLM Benchmark 脚本
# 用途：对比不同配置下的推理性能
# 结果保存到 ./bench-results/
# ============================================================
set -e

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
BASE_URL=${2:-"http://localhost:8000"}
OUTPUT_DIR=${3:-"./bench-results"}

mkdir -p $OUTPUT_DIR

echo "=== vLLM Benchmark ==="
echo "Model: $MODEL"
echo "API: $BASE_URL"
echo "Results: $OUTPUT_DIR"

# 定义测试参数
declare -a REQUEST_RATES=(1 2 4 8 16)
INPUT_LEN=512
OUTPUT_LEN=128
NUM_PROMPTS=100

for rate in "${REQUEST_RATES[@]}"; do
    echo ""
    echo "--- 测试请求率: $rate req/s ---"
    
    RESULT_FILE="$OUTPUT_DIR/rate-${rate}.json"
    
    vllm bench serve \
      --backend vllm \
      --model $MODEL \
      --base-url $BASE_URL \
      --num-prompts $NUM_PROMPTS \
      --request-rate $rate \
      --random-input-len $INPUT_LEN \
      --random-output-len $OUTPUT_LEN \
      --result-dir $OUTPUT_DIR \
      --result-filename "rate-${rate}.json" \
      --disable-tqdm
    
    # 输出关键指标
    if [ -f "$RESULT_FILE" ]; then
        python3 -c "
import json
with open('$RESULT_FILE') as f:
    r = json.load(f)
print(f'  Throughput: {r.get(\"request_throughput\", 0):.2f} req/s')
print(f'  Output tok/s: {r.get(\"output_throughput\", 0):.1f}')
print(f'  Mean TTFT: {r.get(\"mean_ttft_ms\", 0):.0f} ms')
print(f'  Mean TPOT: {r.get(\"mean_tpot_ms\", 0):.1f} ms')
print(f'  Median E2E latency: {r.get(\"median_e2e_ms\", 0):.0f} ms')
"
    fi
done

echo ""
echo "=== Benchmark 完成 ==="
echo "结果保存在: $OUTPUT_DIR"
echo "汇总命令: python3 scripts/bench-summary.py"
