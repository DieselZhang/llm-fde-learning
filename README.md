# 🚀 LLM FDE 学习项目 — 大模型推理部署运维实战

> **From Foundation to Deployment Engineer**
> 从大模型基础知识出发，系统掌握 vLLM / SGLang 多机多卡部署运维

## 📌 项目定位

本项目是一份**7天密集型 FDE（Frontier Deployment Engineer）学习路线**，每天 ≈ 8 小时。目标是帮助有大模型基础的同学，快速掌握：

- **vLLM** 的 PagedAttention、Continuous Batching、分布式推理、性能调优
- **SGLang** 的 RadixAttention、PD 分离、多节点部署、Router 架构
- **GPU 基础设施**：NCCL / RDMA / InfiniBand / DCGM 监控 / 故障排查
- **生产级运维**：Benchmark 方法论、日志分析、性能报告

每天的学习成果（笔记、脚本、性能报告、故障排查记录）都会归档在本仓库。

## 🗺️ 学习路线概览

| 天数 | 主题 | 核心产出 |
|------|------|---------|
| Day 1 | **LLM 推理基础** — PagedAttention, KV Cache, Continuous Batching | 原理笔记 + 推理流程图解 |
| Day 2 | **vLLM 单机部署** — 安装、API、单机多卡 TP/PP | 部署脚本 + API 调用 Demo |
| Day 3 | **vLLM 多机多卡** — Ray 集群、分布式推理、Benchmark | 集群部署脚本 + 性能报告 |
| Day 4 | **SGLang 入门** — 架构、RadixAttention、单节点部署 | 部署脚本 + 架构分析笔记 |
| Day 5 | **SGLang 进阶** — PD 分离、多机部署、sgl-router | PD 分离配置 + 压测报告 |
| Day 6 | **GPU 基础设施运维** — NCCL、RDMA、DCGM、故障排查 | 监控脚本 + 运维手册 |
| Day 7 | **综合实战 + 项目封装** — 完整 Pipeline 演练 | 端到端部署方案 + 总结文档 |

## 📂 仓库结构

```
llm-fde-learning/
├── README.md                    # ← 你现在在这里
├── docs/                        # 每日学习文档（原理 + 实操记录）
├── scripts/                     # 可直接运行的脚本集合
├── reports/                     # Benchmark 性能报告模板和示例
├── daily-logs/                  # 每日学习日志
├── references/                  # 参考手册（架构图、FAQ、速查表）
└── .github/workflows/           # CI 自动同步
```

## 🛠️ 前置要求

- 有 LLM 基础概念（Transformer、Attention 机制、训练/推理区别）
- 了解 Python、Linux 基本操作
- 至少有一张 NVIDIA GPU（本机或云上均可），多卡更佳
- 安装 Docker（多机部署需要）

## 🔗 核心参考资料

| 资源 | 链接 |
|------|------|
| vLLM 官方文档 | https://docs.vllm.ai/ |
| vLLM GitHub | https://github.com/vllm-project/vllm |
| SGLang 官方文档 | https://docs.sglang.ai/ |
| SGLang GitHub | https://github.com/sgl-project/sglang |
| SGLang 学习资料 | https://github.com/sgl-project/sgl-learning-materials |
| mini-sglang（教学版） | https://github.com/sgl-project/mini-sglang |
| Ray 官方文档 | https://docs.ray.io/ |
| NVIDIA DCGM | https://developer.nvidia.cn/dcgm |
| NCCL 文档 | https://docs.nvidia.com/deeplearning/nccl/ |
| 阿里云 vLLM+SGLang 多机部署 | https://help.aliyun.com/zh/ack/cloud-native-ai-suite/user-guide/deploy-multi-machine-distributed-inference-services |

## 🤝 如何使用本仓库

1. **初学者**：按 Day 1 → Day 7 顺序学习，每天阅读 `docs/` 对应文档，运行 `scripts/` 对应脚本
2. **有经验者**：重点看 Day 3/5/6 的多机部署和运维内容
3. **面试准备**：配合 `references/faq-troubleshooting.md` 复习

## 📊 每日进度打卡

完成每日学习后请在 `daily-logs/` 对应文件记录：
- ✅ 今日完成的内容
- 🤔 遇到的难点和解决思路
- 🔗 发现的优质资源
- 📈 性能数据（如有 Benchmark）

---

> **维护者寄语**：FDE 的核心能力不是"会用某个工具"，而是**理解系统原理后能快速定位并解决问题**。这份计划侧重让你既懂原理、又能动手。
