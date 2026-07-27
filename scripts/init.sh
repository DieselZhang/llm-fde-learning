#!/bin/bash
set -e

echo "=== 初始化脚本（首次运行前执行）==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

chmod +x "$SCRIPT_DIR"/*.sh
echo "✅ 所有脚本已设为可执行"
echo ""
echo "可用脚本："
ls -la "$SCRIPT_DIR"/*.sh
echo ""
echo "使用方法（按学习顺序）："
echo "  1) 安装所需工具"
echo "  2) 运行 02-multi-gpu-vllm.sh 启动 vLLM 服务"
echo "  3) 运行 03-vllm-benchmark.sh 进行性能测试"
echo "  ..."
