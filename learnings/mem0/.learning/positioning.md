# Ecosystem Positioning — mem0 vs 同类方案

> 定位事实分两类，永不混写：**repo 事实 `[src]`**（只来自源码/README，仓库是自己的权威）与 **peer/生态事实 `[web]`**（来自检索，带 URL + 访问日期）。未标注来源的 tutor 记忆为 `[unv]`（需验证），只做检索种子，绝不进入参考答案。

## 一句话定位（one-liner）

mem0 是给 AI Agent 用的**开源记忆层**：写路径用 LLM 把对话蒸馏成自包含持久事实（ADD-only），读路径用语义 + BM25 + 实体多信号融合打分召回，按 `user_id/agent_id/run_id` 隔离，附带完整生命周期。
**与裸向量库的本质区别**：向量库回答"哪段文本最像"（文本相似度）；mem0 回答"哪些持久事实对当前用户最相关"（事实抽取 + 实体关联 + 作用域隔离的召回）。

## 七列对比矩阵（where it fits）

| 维度 | **mem0** `[src]`+`[web]` | 裸向量库/自建 RAG `[src]` | Claude Code auto memory `[unv]` | Zep `[web]` | Letta `[web]` | LangMem `[web]` |
|---|---|---|---|---|---|---|
| **写路径** | LLM 抽取持久事实（ADD-only，`main.py:906-1196`） | 存原始文本 chunk，无抽取 | Claude 写自然语言 Markdown 笔记 | 时间知识图谱（bi-temporal） | Agent 自编辑分层记忆（core/recall/archival） | LangGraph 原生 store，含 procedural memory |
| **存储** | 向量库 + `_entities` 实体集合 + SQLite（messages/history） | 向量库（可选 + 文档元数据） | `~/.claude/projects/<p>/memory/` 文件 + MEMORY.md 索引 | Graphiti 时间图 | 分层内存块 + 检索索引 | Postgres / 内存 |
| **检索** | 语义 + BM25 + 实体加分融合（`utils/scoring.py:60-139`） | 纯语义相似度 | MEMORY.md 索引 + 模型上下文判断（无向量检索） | 时间感知图查询 | 分层检索（recall/archival） | LangGraph state + store 查询 |
| **作用域隔离** | `user_id/agent_id/run_id` 内置（`_strip_identity_keys` `main.py:143`） | 自己写 | 按项目 git 根 | tenant 概念（托管） | 无内置多租户 | LangGraph checkpoint |
| **生命周期** | add/search/get/get_all/update/delete/delete_all/history 完整 API | 自己写 | /memory 编辑 + Auto Dream | 图增删改查 + 时间查询 | Agent 工具自编辑 | store CRUD + 更新策略 |
| **时间/矛盾处理** | OSS 无时间衰减（`reference_date` 抛错 `main.py:1422`）；平台有 temporal scoring/Dream | 无 | 文件覆盖（最新胜） | **最强**：LongMemEval temporal 63.8% vs mem0 49.0% `[web]` | 矛盾策略交给 agent 自定 `[web]` | 弱（依赖 LangGraph） |
| **生态/形态** | 最大社区 ~51.8k stars `[web]`；OSS SDK + 托管云；22+ 向量库/15+ LLM/11+ 嵌入 | 无 | Claude Code 内置 | 托管服务（CE 2025 已弃）`[web]` | 框架（Apache-2.0，~21k stars）`[web]` | LangChain 生态内（MIT）`[web]` |
| **基准（directional）** | LoCoMo 66.9% / graph 68.4% `[web]` | — | — | LoCoMo 66.0% / LongMemEval 71.2% `[web]` | LoCoMo **74.0%** `[web]` | LoCoMo 58.1% `[web]` |

> 基准来源不同协议、仅 directional，不可直接横比（`[web]`：atlan.com/know/best-ai-agent-memory-frameworks-2026/，2026-08-12 访问）。

## When to pick it（可迁移判据）

1. **要事实抽取、不要时间推理** → mem0：标准事实记忆场景，OSS 自托管自由 + 生态最大。
2. **只要文本相似、不要抽取** → 裸向量库 + 自写 RAG 足够，别引 mem0 的写路径成本。
3. **要时间/矛盾消解（"什么时候变的"）** → Zep / Graphiti。
4. **Agent 要自治管理记忆、长期自主运行** → Letta（接受框架重）。
5. **已扎根 LangChain/LangGraph** → LangMem（零新依赖）。
6. **数据不能出境、要自托管 + 本地 LLM** → mem0 OSS（18 LLM / 25 向量库，`factory.py`），隐私是 OSS 相对托管的卖点。

## 关键 design tradeoffs 摘要

- **ADD-only 反直觉**：错误不写在写路径（不可逆），移到读路径反复打分裁决——换取消歧但引入矛盾事实并存、候选膨胀（详见 m06）。
- **实体链接 vs 图数据库**：图 DB 运维贵、OSS 用户用不上；轻量 `_entities` 集合 + `linked_memory_ids` 覆盖"实体→记忆"关联（详见 m04/m05）。图 memory 是 Platform-only（#6808）。
- **MD5 精确去重 vs 语义去重**：语义阈值不可靠 + 写路径不可逆合并违背 ADD-only；prompt 软去重 + MD5 硬去重 + 语义留检索期（详见 m04）。

## 引用来源

- `[src]`：`/Users/zhangyongshun/work/Trea_code/mem0`（main 分支）
- `[web]` Best AI Agent Memory Frameworks in 2026 — atlan.com/know/best-ai-agent-memory-frameworks-2026/（2026-08-12）
- `[web]` Best Mem0 Alternatives in 2026 — atlan.com/know/mem0-alternatives/（2026-08-12）
- `[web]` Mem0 vs Zep vs Letta vs Cognee vs LangMem — dev.to/izgorodin/...（2026-08-12）
- `[unv]` Claude Code auto memory 行（`~/.claude/projects/<p>/memory/` 结构）：本机实测，跨环境存在差异，需读者自行验证
