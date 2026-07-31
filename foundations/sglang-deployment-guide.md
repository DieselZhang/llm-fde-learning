# SGLang 完整部署指南（逐行命令）

> 涵盖单机部署、sgl-router 多机部署、PD 分离部署。每步含验证命令和期望日志。

---

## 环境假设

- OS: Ubuntu 22.04 / Rocky Linux 9
- GPU: H100 × 8 (单机), 每节点 NVLink 全互联
- CUDA: 12.4+, Driver: 550+
- Python: 3.10-3.12
- 网络: InfiniBand NDR400 (多机/PD 分离场景)

---

## 一、单机部署

### 1.1 环境准备

```bash
# 1. 确认 GPU
nvidia-smi
# 期望: 列出所有 GPU，Driver 550+

# 2. 虚拟环境
python3 -m venv sglang-env
source sglang-env/bin/activate

# 3. 安装 SGLang
pip install sglang[all]
# 期望: Successfully installed sglang-0.x.x

# 4. 验证安装
python3 -c "import sglang; print(sglang.__version__)"
```

### 1.2 启动服务（单卡）

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

**验证启动成功：**

```bash
# 等待约 30s-2min，直到日志出现：
# INFO: Started server process [pid]
# INFO: Uvicorn running on http://0.0.0.0:30000
# The server is fired up and ready to roll!

# 健康检查
curl http://localhost:30000/health
# 期望: 返回 OK 或空

# 模型列表（SGLang 用 "default" 作为默认 model ID）
curl http://localhost:30000/v1/models
# 期望: {"object":"list","data":[{"id":"default",...}]}

# 服务信息
curl http://localhost:30000/get_server_info
# 期望: {"model_path":"Qwen/Qwen2.5-7B-Instruct","tp_size":1,...}
```

### 1.3 启动服务（多卡 TP）

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --tp 4 \
  --mem-fraction-static 0.85 \
  --enable-metrics
```

**验证 TP 配置：**

```bash
# 看日志确认 TP 大小
grep -i "tp_size\|tensor_parallel\|# GPU" sglang.log
# 期望: tp_size: 4

# 确认进程数
ps aux | grep sglang | grep -v grep | wc -l
# 期望: 约 TP+2 个（Scheduler + Tokenizer + TP 个 Worker）

# 确认 GPU 全部使用
nvidia-smi
# 期望: 4 张 GPU 有 sglang 进程
```

### 1.4 测试 API

```bash
# Chat Completions（注意 model 用 "default"）
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default",
       "messages":[{"role":"user","content":"用一句话介绍北京"}],
       "max_tokens":50,"temperature":0.7}'
# 期望: {"choices":[{"message":{"content":"北京是..."}}],...}

# 流式输出
curl -N http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":20,"stream":true}'
# 期望: 多个 data: 行，最后 data: [DONE]

# Prometheus Metrics
curl http://localhost:30000/metrics | grep -E "sglang_num_requests|sglang_time"
# 期望: sglang_num_requests_running 0.0
#       sglang_time_to_first_token ...
```

---

## 二、sgl-router 多机部署

### 2.1 启动多个 SGLang Worker

```bash
# ===== 每台 Worker 节点各自启动 =====
# Worker 1 (192.168.1.10):
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 --port 30000 --tp 8 --enable-metrics &

# Worker 2 (192.168.1.11):
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 --port 30000 --tp 8 --enable-metrics &

# Worker 3 (192.168.1.12):
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 --port 30000 --tp 8 --enable-metrics &

# 每个 Worker 独立验证
curl http://192.168.1.10:30000/health
curl http://192.168.1.11:30000/health
curl http://192.168.1.12:30000/health
# 期望: 均返回 200 OK
```

### 2.2 启动 sgl-router

```bash
# sgl-router 可部署在任意节点（推荐独立节点，节省 Worker GPU）
pip install sglang[router]

python3 -m sglang.launch_router \
  --worker-urls http://192.168.1.10:30000,http://192.168.1.11:30000,http://192.168.1.12:30000 \
  --host 0.0.0.0 --port 8080 \
  --policy cache_aware
```

**验证 Router：**

```bash
# 1. 确认 Router 启动和 Worker 注册
grep -i "router.*started\|Registered worker" sgl-router.log
# 期望:
#   INFO: sgl-router started on 0.0.0.0:8080
#   INFO: Registered worker: http://192.168.1.10:30000

# 2. 确认 Router 健康状态
curl http://localhost:8080/health
# 期望: 200 OK

# 3. 确认 Worker 列表
curl http://localhost:8080/list_workers
# 期望: ["http://192.168.1.10:30000", "http://192.168.1.11:30000", ...]

# 4. 通过 Router 发送请求
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
# 期望: 正常返回

# 5. 验证缓存感知路由生效
# 连续发两次相同前缀的请求，检查是否路由到同一 Worker:
for i in 1 2; do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default",
         "messages":[{"role":"system","content":"你是一个专业助手"},
                      {"role":"user","content":"Hello"}],
         "max_tokens":10}' &
done
wait
# 在每个 Worker 日志中 grep "Hello"，第二次应命中同一个 Worker
```

### 2.3 Router 问题排查

| 现象 | 排查 |
|------|------|
| Worker 未注册 | `curl http://worker_ip:30000/health` 确认可达 |
| 缓存感知不生效 | 检查 `--policy cache_aware` 是否传入 |
| Router 成为瓶颈 | sgl-router(Rust) 单核数万 QPS，一般不是瓶颈；如需要可启多实例 |

---

## 三、PD 分离部署

### 3.1 SGLang vs vLLM 的关键差异

SGLang 有**显式的** `--disaggregation-mode prefill/decode`。vLLM 没有。

### 3.2 网络要求验证

```bash
# ===== 两个节点都要验证 =====
ibstat               # State: Active, Rate: 400 Gbps
ibaddr               # 确认 IB IP
ib_write_bw -d mlx5_0 --report_gbits <peer_ib_ip>
# 期望: 接近 400 Gbps

nvidia-smi nvlink --capabilities 2>/dev/null | grep -i gpudirect
# 期望: GPUDirect RDMA: Supported
```

### 3.3 启动 PD 分离

```bash
# ===== Prefill 节点 (192.168.100.10) =====
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 --port 30001 \
  --tp 8 \
  --disaggregation-mode prefill \
  --enable-metrics
# 期望日志:
#   INFO: Disaggregation mode: prefill
#   INFO: Server started on 0.0.0.0:30001

# ===== Decode 节点 (192.168.100.11) =====
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 --port 30002 \
  --tp 4 \
  --disaggregation-mode decode \
  --prefill-url http://192.168.100.10:30001 \
  --enable-metrics
# 期望日志:
#   INFO: Disaggregation mode: decode
#   INFO: Connected to prefill node at http://192.168.100.10:30001
#   INFO: Server started on 0.0.0.0:30002
```

### 3.4 验证 PD 分离

```bash
# ===== Prefill 节点验证 =====
grep "disaggregation.*prefill" sglang-prefill.log
# 期望: INFO: Disaggregation mode: prefill

grep -i "transfer\|migrate\|send.*kv" sglang-prefill.log | tail -5
# 期望: 传输日志，含 block 数

# ===== Decode 节点验证 =====
grep "prefill\|connect" sglang-decode.log | head -5
# 期望: INFO: Connected to prefill node at http://192.168.100.10:30001

grep -i "receive\|recv.*kv" sglang-decode.log | tail -5
# 期望: 接收日志

# ===== API 测试 =====
curl http://192.168.100.10:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default",
       "messages":[{"role":"user","content":"请分析以下文档：..."}],
       "max_tokens":100}'
# 期望: 正常返回

# ===== 系统级验证 =====
nvidia-smi dmon -s pucv -d 1 -c 10  # 传输时有 PCIe Tx 峰值
nvidia-smi --query-gpu=index,memory.used --format=csv  # 显存增量
```

### 3.5 PD 分离问题排查

| 现象 | 排查 |
|------|------|
| Decode 连不上 Prefill | `curl http://prefill_ip:30001/health` |
| 传输失败 | `ib_write_bw` 验证 RDMA 连通性 |
| Prefill OOM | 降 `--mem-fraction-static 0.80` |
| Decode OOM | 降 `--max-running-requests` |

---

## 四、sgl-router + PD 分离联合部署（生产推荐）

```bash
# ===== 后端 PD 分离 =====
# Prefill 节点:
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --disaggregation-mode prefill --tp 8 --port 30001 &

# Decode 节点:
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --disaggregation-mode decode --tp 4 \
  --prefill-url http://192.168.100.10:30001 --port 30002 &

# ===== sgl-router 统一入口（指向 Decode 节点） =====
python3 -m sglang.launch_router \
  --worker-urls http://192.168.100.11:30002 \
  --host 0.0.0.0 --port 8080

# 验证: 通过 Router 发请求，自动走 PD 分离流程
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
```

---

## 五、一键部署脚本

```bash
#!/bin/bash
# sglang-deploy.sh [single|multi-worker|pd-prefill|pd-decode] [model] [port]
set -e
MODE=${1:-single}; MODEL=${2:-Qwen/Qwen2.5-7B-Instruct}; PORT=${3:-30000}

check() {
    nvidia-smi --query-gpu=index,name --format=csv | head -3
    python3 -c "import sglang; print('SGLang:', sglang.__version__)"
    echo "✅ OK"
}

case $MODE in
    single)
        check
        python3 -m sglang.launch_server --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" &
        sleep 30 && curl -s http://localhost:"$PORT"/health && echo "✅"
        ;;
    multi-worker)
        check
        python3 -m sglang.launch_server --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" --tp 8 --enable-metrics &
        echo "Worker on :$PORT. Point sgl-router to http://<this_ip>:$PORT"
        ;;
    pd-prefill)
        check
        python3 -m sglang.launch_server --model-path "$MODEL" --disaggregation-mode prefill --tp 8 --port "$PORT" &
        sleep 60 && curl -s http://localhost:"$PORT"/health && echo "✅ Prefill ready"
        ;;
    pd-decode)
        check
        PREFILL=${4:-http://localhost:30001}
        python3 -m sglang.launch_server --model-path "$MODEL" --disaggregation-mode decode --tp 4 --prefill-url "$PREFILL" --port "$PORT" &
        sleep 60 && curl -s http://localhost:"$PORT"/health && echo "✅ Decode ready"
        ;;
    *) echo "Usage: $0 [single|multi-worker|pd-prefill|pd-decode] [model] [port]" ;;
esac
```
