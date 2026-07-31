# 🚀 LLM FDE Learning — From Ops to GPU Infra Engineer

> **从传统运维转向大模型 GPU 运维的学习与实践仓库**
> **A learning & practice repo for moving from traditional ops to large-model GPU operations**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 📌 项目定位 / What This Is

本仓库是一份面向 **FDE（Frontier Deployment Engineer / 大模型部署工程师）** 的基础学习资料与实践示例集，帮助有**传统运维或基础开发背景**的工程师，系统掌握大模型 GPU 推理部署运维所需的完整知识体系。

This repo is a foundational learning resource and practical example collection for **FDE (Frontier Deployment Engineer) roles**. It helps engineers with **traditional ops or basic development backgrounds** systematically master the complete knowledge stack required for large-model GPU inference deployment & operations.

**你将掌握的核心能力 / Core skills you will build:**

- 🖥️ **vLLM** — PagedAttention、Continuous Batching、分布式推理、性能调优
- ⚡ **SGLang** — RadixAttention、PD 分离、多节点部署、Router 架构
- 🛠️ **GPU 基础设施** — NCCL / RDMA / InfiniBand / DCGM 监控 / 故障排查
- 📊 **生产级运维** — Benchmark 方法论、日志分析、性能报告

## 🗺️ 学习路线 / Learning Path

```
传统运维/基础开发
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Foundations（基础理论）                                      │
│  交互式学习页面 + 深度笔记                                     │
│  → KV Cache / PagedAttention / Continuous Batching          │
│  → TP/PP/DP / CUDA Graph / 缓存命中机制                      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Practices（动手实践）                                        │
│  7 天学习文档 + 9 个可直接运行的部署脚本                        │
│  → 单机部署 → 多卡 TP → 多机 Ray → SGLang → PD 分离 → 监控    │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Reports（性能基线）                                          │
│  Benchmark 报告模板 → 形成系统化性能基线                        │
└─────────────────────────────────────────────────────────────┘
```

## 📂 仓库结构 / Repository Structure

```
llm-fde-learning/
├── README.md                        # ← 你现在在这里 / You are here
├── LICENSE                          # CC BY 4.0
├── CONTRIBUTING.md                  # 贡献指南 / Contribution guide
│
├── foundations/                     # 📚 基础学习资料（理论 + 交互式页面）
│   ├── vllm-interactive-learning.html    # vLLM 交互式学习页面
│   ├── sglang-interactive-learning.html  # SGLang 交互式学习页面
│   ├── kv-cache-deep-dive.md             # KV Cache 深度解析
│   ├── vllm-deployment-guide.md          # vLLM 逐行部署指南
│   ├── sglang-deployment-guide.md        # SGLang 逐行部署指南
│   ├── continuous-batching-implementation.md
│   ├── cuda-graph-and-ray.md
│   ├── tp-vs-pp-why-tp-first.md
│   └── gpu-rental-practice-plan.md       # 自费 GPU 租用实践方案
│
├── practices/                       # 🛠️ 动手实践（学习文档 + 脚本）
│   ├── docs/                        # 7 天学习文档
│   │   ├── 01-inference-fundamentals.md   # Day1: LLM 推理基础
│   │   ├── 02-vllm-deep-dive.md           # Day2: vLLM 深入
│   │   ├── 03-vllm-multi-node.md          # Day3: vLLM 多机
│   │   ├── 04-sglang-deep-dive.md         # Day4: SGLang 深入
│   │   ├── 05-sglang-advanced.md          # Day5: SGLang 进阶
│   │   ├── 06-gpu-infra.md                # Day6: GPU 基础设施
│   │   └── 07-summary-cheatsheet.md       # Day7: 总结速查
│   ├── scripts/                     # 可直接运行的部署脚本
│   │   ├── init.sh                  # 初始化：设置脚本权限
│   │   ├── 02-multi-gpu-vllm.sh     # vLLM 多卡 TP/PP 部署
│   │   ├── 03-vllm-benchmark.sh     # vLLM 性能 Benchmark
│   │   ├── 04-multi-node-ray-setup.sh # Ray 多机部署
│   │   ├── 05-sglang-basic.sh       # SGLang 单机部署
│   │   ├── 06-sglang-multi-node.sh  # SGLang 多机部署
│   │   ├── 07-sglang-pd-separation.sh # SGLang PD 分离
│   │   ├── 08-gpu-monitoring.sh     # GPU 监控与诊断
│   │   ├── 09-nccl-test.sh          # NCCL 通信测试
│   │   └── day1-full-pipeline.py    # Day1 完整推理管线
│   └── references/                  # 参考手册
│       ├── faq-troubleshooting.md   # 常见问题排查手册
│       └── README.md                # 目录说明
│
├── reports/                         # 📊 性能报告
│   └── template.md                  # Benchmark 报告模板
```

## 🚀 快速开始 / Quick Start

### 环境要求 / Prerequisites

- 有 LLM 基础概念（Transformer、Attention、训练/推理区别）
- 熟悉 Python、Linux 基本操作
- 至少 1 张 NVIDIA GPU（本机或云上均可，多卡更佳）
- 安装 Docker（多机部署需要）

### 方式一：先学理论（无 GPU 也能开始）

直接在浏览器打开 `foundations/` 下的交互式学习页面：

```bash
# 打开 vLLM 交互式学习
open foundations/vllm-interactive-learning.html

# 打开 SGLang 交互式学习
open foundations/sglang-interactive-learning.html
```

### 方式二：动手实践（有 GPU）

```bash
# 1. 进入实践目录
cd practices/scripts

# 2. 初始化（设置脚本可执行权限）
bash init.sh

# 3. 启动 vLLM 单卡服务（默认 Qwen2.5-7B）
bash 02-multi-gpu-vllm.sh Qwen/Qwen2.5-7B-Instruct 1 1 8000

# 4. 测试 API
curl http://localhost:8000/v1/models

# 5. 运行性能 Benchmark
bash 03-vllm-benchmark.sh
```

> 💡 没有 GPU？参考 `foundations/gpu-rental-practice-plan.md`，有完整的自费租用实践方案（约 ¥50 可完成全流程）。

## 📚 各阶段学习资源 / Resources by Stage

### Stage 1: 基础理论 Foundations

| 资源 | 说明 |
|------|------|
| [vllm-interactive-learning.html](foundations/vllm-interactive-learning.html) | vLLM 交互式学习：10 模块 + 自测题 |
| [sglang-interactive-learning.html](foundations/sglang-interactive-learning.html) | SGLang 交互式学习：10 模块 + 自测题 |
| [kv-cache-deep-dive.md](foundations/kv-cache-deep-dive.md) | KV Cache 从底层原理到生产策略 |
| [tp-vs-pp-why-tp-first.md](foundations/tp-vs-pp-why-tp-first.md) | 为什么推荐 TP 优先于 PP |

### Stage 2: 动手实践 Practices

| 资源 | 说明 |
|------|------|
| [practices/docs/01-inference-fundamentals.md](practices/docs/01-inference-fundamentals.md) | Day 1: 推理基础（Prefill/Decode/KV Cache） |
| [practices/docs/02-vllm-deep-dive.md](practices/docs/02-vllm-deep-dive.md) | Day 2: vLLM 单机部署 |
| [practices/docs/03-vllm-multi-node.md](practices/docs/03-vllm-multi-node.md) | Day 3: vLLM 多机多卡 |
| [practices/docs/04-sglang-deep-dive.md](practices/docs/04-sglang-deep-dive.md) | Day 4: SGLang 入门 |
| [practices/docs/05-sglang-advanced.md](practices/docs/05-sglang-advanced.md) | Day 5: SGLang 进阶 + PD 分离 |
| [practices/docs/06-gpu-infra.md](practices/docs/06-gpu-infra.md) | Day 6: GPU 基础设施运维 |
| [practices/docs/07-summary-cheatsheet.md](practices/docs/07-summary-cheatsheet.md) | Day 7: 总结速查 |

### Stage 3: 部署手册与报告

| 资源 | 说明 |
|------|------|
| [vllm-deployment-guide.md](foundations/vllm-deployment-guide.md) | vLLM 单机/分布式/PD分离逐行命令 + NIXL/MoonCake |
| [sglang-deployment-guide.md](foundations/sglang-deployment-guide.md) | SGLang 单机/Router/PD分离逐行命令 |
| [practices/references/faq-troubleshooting.md](practices/references/faq-troubleshooting.md) | 15 个高频故障排查 |
| [reports/template.md](reports/template.md) | Benchmark 性能报告模板 |

## 🎯 适合谁 / Who It's For

| 角色 | 收益 |
|------|------|
| **传统运维工程师** | 从 Linux/网络基础平滑迁移到 GPU 基础设施运维 |
| **后端/平台开发** | 理解推理引擎底层原理，做架构决策 |
| **FDE 求职者** | 系统化学习路线 + 面试高频考点（PagedAttention/Continuous Batching/PD分离） |
| **AI Infra 初学者** | 交互式页面降低学习门槛，无需 GPU 即可开始 |

## 🤝 贡献 / Contributing

我们欢迎一切形式的贡献——无论是修正笔误、补充案例、完善文档，还是提交新脚本。

We welcome all forms of contributions — typo fixes, case studies, doc improvements, or new scripts.

- **报告 Bug / Report bugs**：[提交 Issue](https://github.com/DieselZhang/llm-fde-learning/issues/new)
- **提交改进 / Submit improvements**：Fork 后提 PR，详见 [CONTRIBUTING.md](CONTRIBUTING.md)
- **提问 / Questions**：直接开 Issue，我们会在 48 小时内回复

## 📄 许可证 / License

本仓库采用 [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) 许可证。

This repository is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).

你可以自由地共享、演绎本内容，只需标注来源。You are free to share and adapt this content, provided you give appropriate attribution.

---

> **维护者寄语 / Maintainer's note**: FDE 的核心能力不是"会用某个工具"，而是**理解系统原理后能快速定位并解决问题**。这份资料侧重让你既懂原理、又能动手。欢迎共同维护，让它成为一份持续生长的支持库。
>
> The core competency of an FDE is not "knowing how to use a tool," but **being able to quickly locate and solve problems by understanding system principles**. This repo focuses on both theory and hands-on practice. Welcome to co-maintain and make it a continuously growing knowledge base.
