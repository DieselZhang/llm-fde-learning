# 📊 性能报告模板

> 每次 Benchmark 后请填写这份报告，形成系统化的性能基线

## 1. 实验信息

| 项目 | 内容 |
|------|------|
| 日期 | YYYY-MM-DD |
| 实验目的 | 简要说明本次实验的目标 |
| 实验编号 | e.g., BENCH-001 |

## 2. 环境信息

### 硬件
| 项目 | 规格 |
|------|------|
| GPU 型号 | e.g., NVIDIA A100-80GB |
| GPU 数量 | e.g., 8 |
| NVLink | 是/否 |
| CPU | e.g., Intel Xeon Platinum 8480C |
| 内存 | e.g., 512 GB |
| 机内互联 | NVLink |
| 跨机互联 | InfiniBand 400Gb/s / RoCEv2 200Gb/s |

### 软件
| 项目 | 版本 |
|------|------|
| 推理框架 | vLLM v0.8.0 / SGLang v0.4.10 |
| CUDA | 12.4 |
| PyTorch | 2.5.0 |
| NCCL | 2.22.3 |
| 驱动版本 | 550.54.15 |

### 网络拓扑
```
（可简图描述节点布局和网络连接）
```

## 3. 模型信息

| 项目 | 值 |
|------|-----|
| 模型名 | e.g., Qwen/Qwen2.5-72B-Instruct |
| 参数量 | 72B |
| 精度 | FP16 / FP8 / INT4 |
| 估算显存 | ~144 GB (FP16) / ~72 GB (FP8) |

## 4. 部署配置

| 参数 | 值 |
|------|-----|
| TP Size | 8 |
| PP Size | 1 |
| max-model-len | 8192 |
| gpu-memory-utilization | 0.90 |
| max-num-seqs | 256 |
| enable-prefix-caching | True/False |

## 5. Benchmark 配置

| 参数 | 值 |
|------|-----|
| Num Prompts | 500 |
| Input Length | 1024 (random) |
| Output Length | 256 |
| Request Rate | 1, 2, 4, 8, 16 |
| Dataset | random / sharegpt |

## 6. 实验结果

### 6.1 吞吐量
| Request Rate | Req/s Throughput | Output Tok/s | Input Tok/s |
|:---:|:---:|:---:|:---:|
| 1 | | | |
| 2 | | | |
| 4 | | | |
| 8 | | | |
| 16 | | | |

### 6.2 延迟
| Request Rate | Mean TTFT(ms) | P99 TTFT(ms) | Mean TPOT(ms) | P99 TPOT(ms) | Mean E2E(ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 8 | | | | | |
| 16 | | | | | |

## 7. 显存监控

| 指标 | 值 |
|------|-----|
| KV Cache 块数 | e.g., 790 块 |
| 峰值显存占用 | e.g., 78.5 GB / 80 GB |
| 显存利用率 | e.g., 98.1% |

## 8. 分析与结论

### 8.1 瓶颈分析
- 当前配置的瓶颈在哪里？（计算/显存/网络/调度）
- 数据支撑：

### 8.2 优化建议
1. 
2. 
3. 

### 8.3 下一步实验
- 

## 9. 附加信息

- NCCL 日志关键输出：
- 异常情况记录：
- 参考链接：
