# Mem0 深度源码课程 — COVERAGE

> 本课程由 **repo-mastery**（一个把任意开源仓库变成深度开发课程的 skill）生成。
> 目标：从源码讲透 mem0 的架构、数据流与设计权衡。覆盖 7 模块 / 23 知识点，Markdown + HTML 双格式。
> 生成日期: 2026-08-12 | 仓库: `mem0ai/mem0`（main 分支）

---

## 价值定位

mem0 是给 AI Agent 用的**开源记忆层**：写路径用 LLM 把对话蒸馏成自包含持久事实（ADD-only），读路径用语义 + BM25 + 实体多信号融合打分召回，按 `user_id/agent_id/run_id` 隔离，附带完整生命周期。
**核心一句话**：向量库回答"哪段文本最像"；mem0 回答"哪些持久事实对当前用户最相关"。

选它 vs 自研的核心判据：**要不要在写路径做 LLM 蒸馏** + 场景要不要时间推理。详见 [positioning.md](positioning.md)。

---

## 课程地图（7 模块 / 23 知识点）

| 模块 | 学什么 | 知识点数 | 状态 |
|---|---|---|---|
| **m01** 定位与生态 | mem0 是什么、vs 谁、何时选 | 3 | ✅ 已覆盖 · 待复习验证 |
| **m02** 构建与环境 | 装起来、配置、API 用法 | 3 | ✅ 已覆盖 · 待复习验证 |
| **m03** 总体架构心智模型 | 分层/存储/三入口 | 3 | ✅ 已覆盖 · 待复习验证 |
| **m04** 写入管线 add() | 8 阶段、抽取、去重、实体 | 4 | ✅ 已覆盖 · 待复习验证 |
| **m05** 检索管线 search() | 数据流、融合、RRF、实体加分 | 4 | ✅ 已覆盖 · 待复习验证 |
| **m06** 设计哲学与权衡 | ADD-only、代价、平台分界 | 4 | ✅ 已覆盖 · 待复习验证 |
| **m07** 动手实验与决策链 | demo 实跑、自研 vs 集成 | 2 | ✅ 已覆盖 · 待复习验证 |

> 状态说明：模块已走教材式闸门（章节全部生成），真掌握待 spaced review 验证——**covered ≠ mastered**，不伪造掌握度。

---

## 模块详解（每模块完整章节见 `.learning/chapters/`）

### m01 定位与生态

- **kp01-01** mem0 是什么与生态位：白盒记忆——写路径蒸馏事实、读路径多信号召回、内置隔离与生命周期。`main.py:930-959`（抽取）、`prompts.py:468`（契约）。
- **kp01-02** 与裸向量库/RAG/Claude Code 的本质差异：记忆形态决定架构——稀疏精确笔记 vs 海量模糊事实库；RAG 面向外部知识，mem0 面向用户事实。
- **kp01-03** 与 Zep/Letta/LangMem 的选型判据：**要不要写路径 LLM 蒸馏** + 要不要时间推理。Zep 赢时效、Letta 赢自治、LangMem 赢 LangGraph 生态 `[web]`。

### m02 构建与环境

- **kp02-01** 源码构建与运行：`pip install mem0ai[nlp]` + spaCy；`Memory()` 零参数可用。
- **kp02-02** MemoryConfig 与 Provider 工厂：`configs/base.py:29` 五要素；`factory.py:42` 注册表（LLM 18 / VS 25 / Emb 11 / Rerank 5）。
- **kp02-03** 核心 API 用法：add/search/get/get_all/update/delete/delete_all/history——`update`/`delete` 是手动按 ID，非 LLM 驱动。

### m03 总体架构心智模型

- **kp03-01** 分层架构与依赖：入口→配置→Provider→存储，依赖自上而下；`AsyncMemory` 是 `Memory` 的 `to_thread` 镜像。
- **kp03-02** 三层存储：向量库（事实+实体，可检索）/ SQLite messages（滚动 10 条，抽取上下文）/ SQLite history（审计）。
- **kp03-03** 三入口关系：同步/异步 × OSS/平台 的笛卡尔积，接口镜像 → 一行切换。

### m04 写入管线 add()

- **kp04-01** V3 八阶段：作用域→上下文→已有记忆→**单次抽取**→embed→MD5 去重→批量写→实体链接。全程仅一次 LLM 调用。
- **kp04-02** 抽取 + UUID 重映射：真实 UUID 重映射成 `"0".."9"` 防 LLM 幻觉 id，`main.py:924-928`。
- **kp04-03** MD5 去重而非语义：prompt 软去重（免费）→ MD5 硬去重（确定性）→ 语义留检索期；阈值不可靠 + 不可逆合并违背 ADD-only。
- **kp04-04** 实体链接替代图 DB：spaCy 抽实体 → 批量 embed → 精确/语义(≥0.95)匹配 → 并集更新 `linked_memory_ids`；图 memory 平台独有（#6808）。

### m05 检索管线 search()

- **kp05-01** 检索数据流：预处理→embed→over-fetch(`max(limit×4,60)`)→关键词→BM25 归一→实体加分→候选→融合。
- **kp05-02** 加性融合公式：`combined = min((sem+bm25+entity)/max_possible, 1.0)`；threshold **先滤语义分**再融合。
- **kp05-03** 加性融合而非 RRF：mem0 分数量纲可比 → 保留分数信息、可解释（explain）、可调参；RRF 适合异构不可比检索器。
- **kp05-04** 实体加分 + explain：`sim × 0.5 × 1/(1+0.001×(n_linked-1)²)`，封顶 [0,0.5]，max 去重；`explain=True` 逐项可归因。

### m06 设计哲学与权衡

- **kp06-01** 为什么 ADD-only：写路径在信息不充分时做不可逆裁决会误删找不回；V3 只单调累积，裁决移给读路径——"memory is retrieval, not storage"。
- **kp06-02** ADD-only 代价：无冲突消解 / 无时间衰减（`reference_date` 抛错 `main.py:1422`）/ 库只增不减 / 召回过时版本。
- **kp06-03** 平台 vs OSS：时间衰减/temporal scoring/latest_only/Dream/图 memory 平台独有；OSS 保持轻量可自托管。
- **kp06-04** 隐私/成本/合规：数据发 LLM 是常态；OSS+本地 LLM=数据不出域；写路径烧 token、读路径重延迟；`infer=False` 跳过 LLM。

### m07 动手实验与决策链

- **kp07-01** OSS demo：`Memory()` + `add` + `search(explain=True)`；验证 MD5 去重、threshold 过滤、新旧并存。
- **kp07-02** 自研 vs 集成决策链：要不要写路径蒸馏 → 裸向量库 / 集成 / fork / 自研；fork 比自研划算（改 `prompts.py`/`scoring.py`/`factory.py`）。

---

## 复习计划（spaced review）

23 个知识点均已初始化复习（concept/design 走 Feynman 复述判定，procedure 走量化闸门）。设计类间隔 14 天，概念/程序类 3 天起。用 `/repo-mastery review` 按到期队列复习。

## 后续学习入口

- 读完整教材：`.learning/chapters/m01.md` … `m07.md`
- 看架构叙事：`.learning/notes/overview.md`
- 看生态定位：`.learning/positioning.md`
- 交互式复习：`/repo-mastery review`
