# MISSION — 为什么要掌握 mem0

## 一句话使命

> 深度讲透 mem0 的架构、数据流与设计权衡，并产出一份**完整、自包含、可直接分享的课程**（COVERAGE.md + HTML）——作为 repo-mastery 能力的展示产物，让读者通过它同时看懂 mem0 和 repo-mastery 的工作方式。

## 价值定位（Value positioning）

- **mem0 能教什么**：Agent 记忆层的完整设计——写路径 LLM 事实抽取、读路径多信号融合召回、ADD-only 哲学、Provider 插件架构，以及"何时自研 vs 集成"的可迁移决策链。
- **它在生态中站哪里**：站在"裸向量库"（只做文本相似）与"完整 Agent 框架"（Letta/LangMem 之类）之间，是**事实抽取 + 召回**的记忆层。选它的判据是"要不要在写路径做 LLM 蒸馏"。
- **vs 同类方案**：Zep 主打时间知识图谱（时效/矛盾消解强，但重度托管）；Letta 主打 Agent 自编辑分层记忆（自治但框架重）；LangMem 绑死 LangGraph 生态；mem0 是**生态最大、上手最快、自托管最自由**的事实抽取记忆层。

## 学习边界（本课程覆盖 / 不覆盖）

- **覆盖**：`mem0/` Python SDK 的定位、构建、架构、add/search 数据流、设计哲学、动手实验与决策链。
- **不覆盖**：`server/`（FastAPI 自托管平台）、`mem0-ts/`（TypeScript SDK）、各 `integrations/` 的完整实现——它们作为"平台 vs OSS"分界的证据被引用，但不逐行展开。

## 为什么用 repo-mastery 学它

- mem0 是**中型仓库**（`mem0/` 32k 行），结构规整、设计决策反直觉且可迁移——正是 repo-mastery 的"key implementations + design 权衡"专长场景。
- 产出物（COVERAGE.md + HTML）本身就是 repo-mastery 能力的可分享样本：读者看到的不只是 mem0，还有"一个 skill 如何把源码变成深度课程"。
