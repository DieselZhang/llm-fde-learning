# 全局概览 — mem0 架构叙事

> 一句话：mem0 是给 AI Agent 用的**开源记忆层**——写路径把对话蒸馏成持久事实（ADD-only），读路径多信号融合召回。本页是进入任何知识点前的地图。

## 一、整体架构叙事（入口 → 核心数据流）

```
                     ┌──────────────────────────────────────────────┐
                     │             Memory / AsyncMemory              │
                     │              (mem0/memory/main.py)            │
                     │   add()  V3 8阶段管线  ·  search() 多信号召回  │
                     └───────┬──────────────┬──────────────┬─────────┘
                             │              │              │
                    ┌────────▼───────┐ ┌────▼─────┐ ┌───────▼──────┐
                    │   LLM 工厂     │ │ Embedder │ │ VectorStore  │
                    │  18 providers  │ │ 11 provs │ │ 25 providers │
                    └────────┬───────┘ └──────────┘ └───────┬──────┘
                             │  mem0/utils/factory.py        │
                    ┌────────▼───────────────────────────────▼──────┐
                    │            SQLite（本地辅助存储）              │
                    │   messages: 原始消息滚动10条/会话（仅供抽取上下文）│
                    │   history:  ADD/UPDATE/DELETE 审计日志          │
                    └───────────────────────────────────────────────┘
```

**核心数据流一句话**：`add()` 用一次 LLM 调用把对话蒸馏成自包含事实 → 精确 MD5 去重 → 批量写向量库（主集合 + `_entities` 实体集合）→ 实体链接；`search()` 用 query 语义 over-fetch 候选 → BM25 关键词 + 实体加分 → 加性融合 → threshold 先过滤语义 → top_k。

## 二、模块地图（7 模块 / 23 知识点）

| 模块 | 学什么 | 为什么这个顺序 |
|---|---|---|
| **m01** 定位与生态 | mem0 是什么、vs 谁、何时选 | 先建立"它站在生态哪个位置"的框架 |
| **m02** 构建与环境 | 装起来、配置、API 用法 | 信任建立：看到它跑起来 |
| **m03** 总体架构 | 分层/存储/三入口 | 心智模型：能预测行为 |
| **m04** 写入管线 add() | 8 阶段、抽取、去重、实体 | 核心实现①：记忆怎么进来 |
| **m05** 检索管线 search() | 数据流、融合公式、RRF、实体加分 | 核心实现②：记忆怎么被想起 |
| **m06** 设计哲学 | ADD-only、代价、平台分界 | 深水区：为什么这么设计 |
| **m07** 动手 + 决策链 | demo 实跑、自研 vs 集成 | 落地：把理解变成判断力 |

## 三、关键实现亮点（进任何模块前先知道这三个）

1. **单次 LLM 抽取（ADD-only）**：整个 `add()` 只有一次 LLM 调用（`main.py:930-959`），强制 JSON 输出 `{"memory":[{id,text,attributed_to,linked_memory_ids}]}`，无 UPDATE/DELETE 决策——裁决权移到读路径。
2. **加性融合打分**：`combined = (semantic + bm25 + entity_boost) / max_possible`（`scoring.py:60-139`），threshold **先作用于 semantic 再融合**——不是 RRF 的排名盲融合，是量纲归一、可解释、可调参的分数融合。
3. **实体链接替代图数据库**：spaCy 抽实体 → `_entities` 集合精确/语义（≥0.95）匹配 → 并集更新 `linked_memory_ids`（`main.py:1077-1180`）——轻量覆盖"实体→记忆"关联，图 memory 留给 Platform。

## 四、差异化总结（vs 生态）

mem0 站在**裸向量库**（只做文本相似）与**完整 Agent 框架**（Letta/LangMem）之间，是"事实抽取 + 召回"的记忆层。选它的核心判据：**要不要在写路径做 LLM 蒸馏**。详见 `positioning.md`。

## 五、学习入口

- 从 m01 开始 → 顺序推进；每模块先读模块 overview，再走教材式章节。
- 源码锚点：`mem0/memory/main.py`（Memory/AsyncMemory 双实现）、`mem0/utils/scoring.py`（融合公式）、`mem0/utils/factory.py`（Provider 注册）、`mem0/configs/prompts.py`（抽取 prompt）。
