# Day 7: 综合实战与项目封装

> 学习目标：整合一周所学，完成一个端到端的部署方案，封装 GitHub 项目

## 🎯 本日学习内容

- [ ] 端到端部署演练：从 0 搭建推理集群
- [ ] 性能报告撰写
- [ ] GitHub 项目完善
- [ ] FDE 面试/工作准备

## 📖 理论学习（1.5h）

### 1. FDE 核心能力图谱

```
┌─────────────────────────────────────┐
│         FDE 能力模型                  │
├─────────────────────────────────────┤
│  1. 部署能力                         │
│     ├── 框架部署 (vLLM / SGLang)    │
│     ├── 集群管理 (Ray / Slurm)      │
│     └── 容器化 (Docker / k8s)       │
├─────────────────────────────────────┤
│  2. 性能调优                         │
│     ├── Benchmark 方法论             │
│     ├── 参数优化                     │
│     └── 瓶颈分析                     │
├─────────────────────────────────────┤
│  3. 运维监控                         │
│     ├── GPU 监控 (DCGM)             │
│     ├── 日志分析                     │
│     └── 告警体系                     │
├─────────────────────────────────────┤
│  4. 故障排查                         │
│     ├── NCCL 通信问题               │
│     ├── 显存不足                     │
│     └── 网络瓶颈                     │
├─────────────────────────────────────┤
│  5. 工程素养                         │
│     ├── 文档化                       │
│     ├── 可复现                       │
│     └── 知识沉淀                     │
└─────────────────────────────────────┘
```

### 2. 性能报告的撰写规范

一份好的性能报告应包含：

1. **实验目的**：为什么要做这个 Benchmark
2. **环境信息**：硬件、软件版本、网络拓扑
3. **实验方法**：数据集、并发数、指标定义
4. **结果数据**：以表格/图表展示
5. **分析结论**：数据背后的原因
6. **优化建议**：下一步可以做什么

## 🛠️ 实操练习（6h）

### 练习 1：端到端部署大挑战

从零搭建一个可用的推理服务，记录每一步的时间和环境。

**方案 A：单机多卡（推荐，硬件要求低）**
```bash
# 场景：部署 Qwen2.5-7B，支持 10 并发请求，P99 延迟 < 2s

# Step 1: 环境准备
# Step 2: 启动 vLLM / SGLang
# Step 3: 压测
# Step 4: 参数调优
# Step 5: 性能报告
```

**方案 B：多机部署（如有 2+ 台服务器）**
```bash
# 场景：部署 72B 模型，跨 2 节点 16 卡
# Step 1: 配置 Ray 集群
# Step 2: 启动 PD 分离
# Step 3: 压测对比
# Step 4: 网络调优
# Step 5: 性能报告
```

### 练习 2：撰写性能报告

在 `reports/` 目录下创建你的第一份性能报告。

使用模板 [reports/template.md](../reports/template.md)。

### 练习 3：GitHub 项目封装

```bash
# 1. 创建 GitHub 仓库
cd llm-fde-learning

# 初始化 git
git init
git add .
git commit -m "🎉 init: LLM FDE learning project"

# 2. 推送到 GitHub
# 先在 GitHub 创建仓库（不要勾选 README），然后：
git remote add origin git@github.com:你的用户名/llm-fde-learning.git
git branch -M main
git push -u origin main

# 3. 完善 README 的徽章
# 在 README 中添加：
# - 项目的 Stars/Forks 徽章
# - Python 版本徽章
# - License 徽章
```

### 练习 4：知识自检

通过自问自答检验学习效果：

**基础篇**：
1. PagedAttention 如何解决 KV Cache 显存碎片问题？
2. Continuous Batching 相比 Static Batching 提升了什么？
3. Tensor Parallelism 和 Pipeline Parallelism 的区别？

**部署篇**：
4. vLLM 多机部署的完整步骤是什么？
5. SGLang 的 PD 分离解决了什么核心问题？
6. RadixAttention 相比 PagedAttention 的优势在哪里？

**运维篇**：
7. NCCL 的通信路径优先级是怎样的？
8. 如何诊断 GPU 跨节点通信瓶颈？
9. 推理服务显存不足时有哪些调优手段？

**综合篇**：
10. 给一个 70B 模型做在线服务，8 卡 A100-80GB 节点 2 台，你的部署方案是什么？TP/PP 如何设置？为什么？

## 📝 学习日志与总结

在 `daily-logs/day-07.md` 中完成：

### 一周学习回顾

```
完成度：☐ 100% / ☐ 80% / ☐ 60% / ☐ 40%
```

| 天数 | 主题 | 自我评分(1-5) | 难点 |
|------|------|:---:|------|
| Day 1 | 推理基础 | | |
| Day 2 | vLLM 单机 | | |
| Day 3 | vLLM 多机 | | |
| Day 4 | SGLang 入门 | | |
| Day 5 | SGLang 进阶 | | |
| Day 6 | GPU 运维 | | |
| Day 7 | 综合实战 | | |

### 核心收获（列出 3-5 个最重要的知识点）

### 仍然困惑的问题

### 下一步学习方向建议
- [ ] 深入学习 k8s 部署（vLLM Operator, KServe）
- [ ] 学习训练框架（Megatron-LM, DeepSpeed）
- [ ] 学习推理优化（Speculative Decoding, Quantization）
- [ ] 学习 GPU 编程（CUDA, Triton）
- [ ] 其他：________

## 🏆 项目完成 Checklist

- [ ] 7 天的 daily-logs 都已完成
- [ ] docs/ 下有 7 篇学习文档
- [ ] scripts/ 下可运行的脚本
- [ ] 至少 1 份性能报告 (reports/)
- [ ] references/ 下有 FAQ / 速查表
- [ ] 代码已推送到 GitHub
- [ ] 项目对其他人有参考价值

## 🎉 祝贺你完成一周的 FDE 学习之旅！

> FDE 的核心不是"会用某个工具"，而是**当所有工具都失效时，你还能从系统原理出发定位问题**。保持好奇，持续动手。
