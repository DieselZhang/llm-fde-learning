# Pi Agent 源码精读笔记

> **来源**：dg-ai-notes（冬瓜）《Pi Agent 源码精读》10 章教程
> **网站**：https://dg-ai-notes.pages.dev/
> **项目**：Pi Agent（pi.dev / github.com/earendil-works/pi）—— 极简、可扩展的终端编码 Agent SDK
> **笔记类型**：Agent SDK 源码精读（与 LLM 推理引擎学习形成互补：推理引擎管"算"，Agent 管"怎么用"）

---

## 第 1 章：开篇 —— 为什么 Pi-Agent 值得花时间

### 一句话定义
Pi 是"极简、可扩展的终端编码 Agent 外壳（coding agent harness）"，由 libGDX 作者 Mario Zechner 创建，全 TypeScript，MIT 开源。

### 关键数字（v0.80.2）
- GitHub Stars 64,000+
- 内置工具：4 核心（read/write/edit/bash）+ 3 辅助（grep/find/ls）
- 系统提示词：静态模板约 90 词（运行时 200-400 词），对比 Claude Code 数万字
- TUI 代码约 12,000 行
- 支持 35 个 KnownProvider（去重后约 27 个独立品牌）
- 4 个核心包：pi-ai / pi-agent-core / pi-tui / pi-coding-agent
- 4 种运行模式：交互 / print-JSON / RPC / SDK

### 四层架构
```
pi-coding-agent  ← 完整 CLI 产品 + SDK（系统提示词·内置工具·会话管理·扩展）
pi-tui（正交 UI 库）│ pi-agent-core（AgentLoop·工具系统·事件流）
pi-ai  ← 多供应商 LLM 抽象（统一 API·流式·Token 追踪）
```

### 三种身份
1. **编码 Agent**：定位是"积木，而非整车"，提供**五根定制杠杆**——扩展（最强）、技能、提示词模板、主题、Pi 包
2. **学习教材**：核心循环只有几百行，可"真正读完一个高质量 Agent 的全部核心代码"
3. **SDK**：三层可独立使用（只用 pi-ai 调模型 / pi-ai+agent-core 跑自定义 Agent / 全三层完整产品）

### 减法哲学
Pi 不做的：MCP（会话灌 13,700+ token 工具描述 → 用带 README 的 CLI）、子 Agent（→ tmux 多实例）、权限弹窗（→ 容器化）、计划模式（→ plan.md）、后台 bash（→ tmux）、内置待办（→ TODO.md）。

**核心结论**："做减法是一种有竞争力的产品立场"——"我不需要的，就不会被构建"本身就是一项真正的功能。

---

## 第 2 章：三层架构 —— 项目的骨骼

### 五包结构（npm workspaces monorepo）
- `pi-ai`：统一 LLM API，抹平 30+ 提供商差异（统一类型 + `streamSimple` 流式）
- `pi-agent-core`：通用 Agent 循环/状态/事件/压缩（不含业务知识）
- `pi-coding-agent`：编程业务（7 工具、扩展系统、CLI、持久化、认证）
- `pi-tui`：只管终端显示（运行时仅依赖 marked + get-east-asian-width）
- `pi-orchestrator`：实验性多 Agent 编排层

### 分层的真正规则
**不是"限制引用层级"，而是"控制依赖方向必须单向向上"**。coding-agent 直接依赖 pi-ai 是必然（基础类型统一定义在 pi-ai），关键在底层永远不反向引用上层。

```
pi-ai（底层）
  ↑         ↑
  │    pi-agent-core（中间层）
  │         ↑
  └─── pi-coding-agent（顶层）
```

### 类型递进扩展
```
Tool（原子：长什么样）→ AgentTool（分子：怎么执行）→ ToolDefinition（材料：怎么显示）
```
通过 `extends` / 联合类型扩展，**从不修改底层类型**。

### 三个可复用方法
1. **依赖漏斗分层法**：底层"不知道外面世界"，验证靠"去掉上层这层还能跑吗"
2. **类型递进扩展**：每层只加自己关心的事，不改底层
3. **可独立使用测试**：移除上层依赖看编译测试是否通过

---

## 第 3 章：Agent Loop —— 让模型转动起来的引擎

### 大模型的三种用法
| 维度 | 直接调用 | Workflow | Agent Loop |
|---|---|---|---|
| 决策者 | 用户 | 你的代码 | 模型 |
| 模型调用次数 | 1 次 | N 次（代码控制） | 不确定（模型控制） |
| 模型角色 | 执行者 | 流水线环节 | 自主决策者 |

### 两个关键概念
- **Trace**：一次完整运行（agent_start → agent_end），包含多个 Turn
- **Turn**：一次模型调用 + 这次调用触发的所有工具执行（turn_start → turn_end）

### 循环驱动机制：stopReason
**"模型不会说'我要停了'。"** 模型只是 token 预测器。

| stopReason | 来源 | 含义 |
|---|---|---|
| toolUse / stop / length | 模型 API 返回 | 有工具调用 / 自然终止 / 截断 |
| error / aborted | 框架流式层注入 | 异常 / 用户中止 |

**核心规则**：实际驱动循环的不是 stopReason，而是 `toolCalls 数组长度 > 0 && !terminate`。循环只看一件事——**模型输出里有没有工具调用**。

### 退出路径
| 路径 | 触发条件 |
|---|---|
| 正常退出 | stop/length + 无 followUp + 无 pendingMessages |
| 硬停止 | error/aborted |
| 外部钩子停 | shouldStopAfterTurn() 返回 true |
| 工具终止 | 工具结果全部 terminate: true（every 不是 some） |

### 内核 + 叠加架构
**最简 Loop 十几行代码**就是所有 Agent 的内核。coding-agent 在内核上叠加：
- **steering**：紧急插队消息（用户工作期间输入新指令）
- **followUp**：任务追加（外层循环重启内层）
- **prepareNextTurn**：每轮结束可换模型/上下文（按任务复杂度切模型）
- **shouldStopAfterTurn**：安全阀（上下文快满）

### 流式"原地替换"机制
先 push 空壳消息 → 逐 token 覆盖最后一条 → UI 实时渲染。

### prompt cache（关键性能点）
Anthropic 三个 cache_control 位置：system 末尾 + 最后一个 tool + **最后一条 user message（rolling cache）**。命中价约是输入价的 1/10。

---

## 第 4 章：模型调用 —— 一行代码驾驭多个模型

### 问题：不同 Provider 四种写法
消息格式（content[] vs parts[]）、流式传输、思考模式（thinking.budget_tokens vs reasoning_effort）、缓存控制（cache_control vs cachePoint）四维度 × 30+ Provider = 巨大差异。

### 三层架构解法
```
第一层 统一入口：stream() 查表派活
第二层 事件协议：12 种统一事件（start / text|thinking|toolcall 三阶段 / done / error）
第三层 翻译器：每个 Provider 一个，精通自家"方言"
```

### StreamFunction："宪法"
输入相同、输出相同（必须返回 AssistantMessageEventStream）、**错误不抛异常**（发 error 事件而非 throw）。

### 接入新模型三步
写翻译器 → 注册（registerApiProvider）→ 配置模型信息。**Agent Loop 一行都不用改。**

### 关键设计
- **ThinkingLevel 六级刻度**：off/minimal/low/medium/high/xhigh，每个 Model 自带映射表，clamp 回退先向上再向下
- **CacheRetention 语义接口**：上层说 none/short/long，各家翻译器各自实现
- **错误编码进流**：异常不打断循环，stopReason: "error"/"aborted" 是 catch 块注入的
- **设计精华**：协议 > 实现（不设计 BaseProvider 抽象类）；统一枚举 + 映射表；语义统一、实现分散

---

## 第 5 章：工具系统 —— Agent 的手脚怎么被管住

### 三层类型
- `Tool`（pi-ai）：名片——name/description/parameters
- `AgentTool`（agent-core）：执行——+label/prepareArguments/execute/executionMode
- `ToolDefinition`（coding-agent）：产品——+ctx/promptSnippet/renderCall 等

### 五步管道（核心）
```
① prepareArguments（参数兼容性垫片）
② validateToolArguments（Schema 验证）
③ beforeToolCall（前置钩子，可阻止）
④ tool.execute（实际执行，AbortSignal + onUpdate 进度）
⑤ afterToolCall（后置钩子，可修改结果）
```
每一步出错都不抛异常，最终产物统一为 `ToolResultMessage { isError: true }`。

### 并行 vs 串行：一票否决
只要一个工具声明 sequential，整批串行。并行是三阶段：准备顺序、执行并行（Promise.all）、事件有序。

### 永不抛出："错误即消息"
**让模型自己决定下一步，比框架替它决定更好。** 错误描述越具体，模型纠错能力越强。Bash 工具是教科书级——主动识别中止/超时/非零退出码三类已知错误并包装具体描述。

### Operations 抽象
工具不直接调系统 API，通过最小化 Operations 接口间接调用。测试可 Mock、远程可 SSH，不改工具代码。

---

## 第 6 章：消息系统 —— Agent 的记忆如何组织

### 两层消息架构："内富外严"
```
AgentMessage（内层）: 7 种消息，字段丰富
  = Message（3种标准）+ CustomAgentMessages（4种自定义）
Message（外层）: 3 种标准格式（user/assistant/toolResult）
```

### 3 种标准消息
- UserMessage：用户的话/图片
- AssistantMessage：LLM 回复（含思考、工具调用），content 是内容块数组
- ToolResultMessage：工具结果，通过 toolCallId 精准关联 ToolCall

### 4 种自定义消息（coding-agent）
BashExecutionMessage / CustomMessage / BranchSummaryMessage / CompactionSummaryMessage

### 声明合并：类型安全扩展
核心包 `CustomAgentMessages` 默认为空，应用通过 `declare module` 声明合并扩展。核心包零依赖，扩展包全栈类型安全。

### convertToLlm：一切自定义消息终将变成 User
自定义消息在 LLM 边界翻译成 UserMessage（LLM API 要求 user/assistant 交替）。**有损、单向、最后一刻发生**——结构化字段 UI 早就用过了。

### 三种可见性级别
全可见 / LLM 不可见（excludeFromContext=true，UI 照常渲染）/ 仅持久化

### 可迁移三步法
① 识别"两个读者"分别要什么 ② 以内层结构化为源、外层翻译为流 ③ 用类型扩展点做"核心+应用"分层

---

## 第 7 章：事件驱动 —— Agent 的神经系统

### 10 种事件，4 层嵌套
```
agent_start/end（整个运行）
  turn_start/end（一轮模型调用+工具执行）
    message_start/update/end（一条消息）
      tool_execution_start/update/end（一次工具执行）
```
每层都是"开始 → 更新×N → 结束"配对。

### 核心设计：emit 是"同步屏障"而非"通知"
**每个 emit 都带 await**——"等这个事件被完全处理完，再继续"。保证消费者状态一致。

```
await emit({ type: "message_start", ... });  // 等 TUI 处理完才发下一个
await emit({ type: "message_update", ... });
```

**例外**：tool_execution_update 不 await——先收集 Promise 数组，工具结束后一次性 `Promise.all`。进度更新"高频、低价值、可合并"。

### 错误处理：监听器异常直接冒泡
没有 try-catch，一个 UI 渲染 bug 能把 Agent 干掉。哲学："监听器出错，运行就停下来，问题立刻可见。" 写扩展时务必自己 try-catch。

### 两层事件
Agent 内核 10 种 + Session 层扩展 7 种（queue_update/compaction/auto_retry 等）。"去掉某个事件后内核还能正常运行，它就属于外层。"

---

## 第 8 章：上下文工程 —— 让有限窗口装下无限对话

### 两层四道防线
| 环节 | 解决的问题 | 触发频率 |
|---|---|---|
| ① 工具输出截断（输入侧） | 单次工具结果太大 | 每次工具调用 |
| ② 系统提示词组装（输入侧） | 项目规范进上下文 | 每轮 prompt |
| ③ Compaction（历史侧） | 长对话累积超窗口 | 阈值触发 |
| ④ 分支摘要（历史侧） | 分支跳转旧分支不能丢 | 切换分支时 |

### 工具输出截断四件套
- **双重限制**：2000 行 / 50KB，哪个先触发用哪个
- **双向策略**：truncateHead（read 文件，头部信息密度高）/ truncateTail（bash 输出，错误在末尾）
- **边界安全**：UTF-8 代理对（emoji）整体处理
- **兜底逃生**：截断时提示"完整输出在 /tmp/...，可 read 按需取"

### 系统提示词组装（加法）
- 多层 CLAUDE.md 向上递归（全局 → 祖先 → 项目，从外到内）
- XML 结构化包装（边界明确 + path 属性）
- **Skills 懒加载**：清单进 prompt（10 skill ≈ 500 token），全文按需 read（≈ 50K token）——"拉模式"优于"推模式"

### 设计精华
1. **多层防护无银弹**：每层只解决自己擅长的问题，组合成漏斗
2. **加减法并重**：Compaction 是减、系统提示词是加，结构化是杠杆
3. **工具调用 = 按需上下文加载**：当 LLM 有工具能力时，"工具"本身就是上下文工程的载体

---

## 第 9 章：上下文压缩 —— 当对话太长怎么办

### 关键时序：压缩在两轮对话之间
```
用户问 → Agent 回答 → agent_end → 检查 token 是否超阈值
  → 超了：找切割点 → 生成摘要 → CompactionEntry 写进 Session Tree
  → 下一轮：buildSessionContext() 用摘要替代旧消息
```

### 触发条件
`contextTokens > contextWindow - reserveTokens`（200K - 16,384 = 183,616）。reserveTokens 给 LLM 回复预留空间。

### Token 估算（已知精度问题）
`chars / 4` 启发式，英文接近；**中文被严重低估**（1 汉字实际 1-2 token）。保守策略：高估无害、低估有害。

### 切割点算法
- **有效切点**：user/assistant 合法，toolResult 不合法（toolResult 必须紧跟 ToolCall）
- **切点 = 保留区第一条**（不是被切掉的最后一条）
- **向后遍历**：从最新往回累积 token 到 keepRecentTokens（20K），保护最近上下文

### 结构化摘要 6 section
```
Goal / Constraints & Preferences / Progress（Done/In Progress/Blocked）
/ Key Decisions / Next Steps / Critical Context
```
固定模板对抗 LLM"自由发挥"——把"易遗漏"变成"必须填"。

### 增量更新
传 previousSummary，LLM 做更新而非重写，避免累积误差。

### Turn 分割（极端情况）
assistant 切点会切断 Turn → 用 turnPrefix 摘要（3 段格式）并行生成弥补。权衡：只允许 user 切点保留区过大压不动，允许 assistant 切点精确控 token 但断 Turn。

### 文件跟踪
摘要附加 `<read-files>` / `<modified-files>` 列表跨压缩累积。对编码 Agent，"改过哪些文件"比"讨论过什么"更精确可验证。

---

## 第 10 章：会话管理 —— 对话的存储、恢复与分叉

### 两个独立维度（不可混淆）
| 维度 | 回答什么 | Pi 的选择 |
|---|---|---|
| 存储介质 | 存在哪里 | 本地 JSONL 文件 |
| 数据结构 | 长什么样 | 树（Session Tree） |

### Session Tree 四连问
- **为什么树**：对话不线性——用户会回退、重试、分支
- **为什么 append-only**：删了的数据找不回来，历史分支可能有价值
- **为什么认父不认子**：append-only 要求节点不可变，父节点不能维护 children 列表
- **为什么路径遍历**：LLM 只认线性数组，树形必须"压扁"

### 三个核心操作
1. **追加**：O(1)，不改任何旧节点（创建时 parentId 指向当前 leafId）
2. **回退**：只移动 leafId（`this.leafId = "e2"` 一行），不删节点
3. **分支**：回退后追加的自然结果

### buildSessionContext（树 → LLM 上下文）
```
path = leaf → root 往回走 → 反转
→ 按类型分派：消息进 messages、状态变更改状态变量、元数据跳过
→ 压扁成线性 messages 数组
```

### 节点化状态让回退天然正确
"切换模型"存成节点而非全局状态 → 回退到切换前节点时，路径不含那条 model_change，model 自动回到之前值。

### JSONL 细节
- 行级追加（appendFileSync），与 append-only 树完美契合
- 延迟写入：首次 assistant 消息到达前不落盘，避免"有问无答"的半截对话
- 偶尔全文件重写：创建分支副本 / 修复损坏

### 三个可迁移思路
1. 拆开"存哪里"和"长什么样"两个维度
2. append-only 用于"撤销/回退/分支"场景——不删旧数据，用指针定位
3. 节点化状态变量，让回退天然正确

---

## 全课程主线速查

### 设计哲学
- **做减法是有竞争力的产品立场**（第1章）
- **依赖方向单向向上**（第2章）
- **内核 + 叠加**：最简 Loop 十几行，产品功能按需叠加（第3章）
- **错误即消息**：永不抛出，让模型自己决定下一步（第5章）
- **内富外严**：内部丰富表达，边界严格翻译（第6章）
- **同步屏障**：emit 等监听器处理完（第7章）
- **多层防护无银弹**（第8章）
- **结构化摘要对抗自由发挥**（第9章）
- **append-only 树让回退不丢数据**（第10章）

### 与 FDE 学习的关联
| Pi Agent 概念 | 对应你的知识 |
|---|---|
| Agent Loop 的 Turn 概念 | 类似推理的 decode 迭代 |
| prompt cache 三位置标记 | 你学的 KV Cache / Prefix Caching |
| 上下文压缩（Compaction） | 类似长上下文管理的工程实践 |
| 工具输出截断 | 类似控制"什么进上下文"的显存管理思想 |
| Session Tree append-only | 类似操作系统/数据库的 undo log 思想 |
