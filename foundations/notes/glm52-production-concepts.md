# GLM-5.2 生产部署中的核心概念详解

> 基于《从 Day 0 到生产级 SLA：基于 vLLM 在 NVIDIA B300 上部署 GLM-5.2》一文
> 将文章中涉及但基础学习未覆盖的模型架构与引擎特性逐一讲解
> 配套阅读：[glm52-b300-vllm-pd-deploy.md](glm52-b300-vllm-pd-deploy.md)（部署案例笔记）

---

## 概念全景

```
① 模型架构层 —— GLM-5.2 本身怎么设计的
   MoE 混合专家 | DSA 稀疏注意力 | MTP 多token预测 | NVFP4 量化

② 推理引擎层 —— vLLM 怎么为这些架构做适配
   MRV2 Model Runner | EP 专家并行 | FlashInfer A2A | IndexerCache
   Local Argmax Reduction | FULL_DECODE_ONLY

③ 生产配置层 —— 实际部署怎么配
   VLLM_USE_V2_MODEL_RUNNER=1 | num_speculative_tokens | Speculative Padding
   Triton JIT 冷启动 | PCP 虚拟批处理
```

---

## ① 模型架构层

### 1. MoE（Mixture of Experts）混合专家

**为什么需要**：模型参数量决定知识容量，但推理时每个 token 不需要动用全部参数。

```
Dense 模型（Llama/Qwen）:
  744B 参数 = 每个 token 都要算全部权重 → 计算量大

MoE 模型（GLM-5.2）:
  总参数 744B，内部切成多个"专家"（Expert）
  每个 token 只激活 Top-K 个专家（GLM-5.2 实际激活 ~40B）
  → 知识容量仍由 744B 决定，计算成本只有 40B 级别
```

**MoE 核心结构**：

```
输入 token x
     ↓
┌──────────────────────┐
│  Router（路由门控）    │
│  对每个专家打分选 Top-K│
└──────────┬───────────┘
           ↓ 只激活 Top-K 个专家
   ┌───────┼────────┐
   │ E1    E5  ...  E9  ← 几百个专家只算选中的几个
   └───────┴────────┘
           ↓
       汇总输出
```

**对 FDE 的意义**：
- 显存要装下全部 744B 权重（Router 可能选任何专家），推理只算 40B
- 衍生出专用并行方式 EP（见引擎层）

---

### 2. DSA 稀疏注意力（Sparse Attention）

**为什么需要**：标准 Attention 是 O(n²)，GLM-5.2 支持 100 万 token 上下文，必须稀疏化。

**核心思想**：不是所有历史 token 都值得关注，只让每个 query 关注最相关的 Top-K 个。

```
标准 Dense Attention:
  Q(第100万token) 与全部 100 万历史 K 算相关度 → O(n²)

DSA 稀疏注意力:
  Indexer 粗筛 → 选出最相关 Top-K 个 key → 只对这 K 个做精确 Attention
  → 计算量从 O(n²) 降到 O(n·K)
```

**关键组件 Indexer（索引器）**：
```
每个 token 的 Query 先经过轻量 Indexer
→ 从历史 Key 中快速检索 Top-K 最相关的
→ 只有选中的 key 参与真实 Attention

类比: 搜索引擎先检索出 Top-K 结果，不读整个互联网
```

---

### 3. MTP（Multi-Token Prediction）多 Token 预测

**为什么需要**：普通 LLM 一步预测一个 token；MTP 一次预测未来 N 个。

```
普通生成: "今天天气" → 预测"真"（每步一个完整 forward pass）

MTP: "今天天气" → 同时预测["真","好","晴","朗"]
     模型含多个预测头（MTP Modules），一次输出 N 个 token 预测
```

**MTP 的两种用途**：
1. 训练加速：一次学习多个位置的 token
2. **推理加速（推测解码）**：MTP 预测头当 Draft Model，快速生成候选 token，主模型验证

**GLM-5.2 原生支持 MTP** → 文章 `num_speculative_tokens=3` 的基础。

---

### 4. NVFP4 量化

**为什么需要**：744B 参数 BF16 需 ~1.5TB 显存，普通机器放不下。

```
BF16:  2 字节/参数 → 744B × 2 = 1488 GB ❌
FP8:   1 字节/参数 → 744B × 1 = 744 GB  ✅ 需 8 张 H100
NVFP4: 0.5 字节/参数 → 744B × 0.5 = 372 GB ✅ 更少卡

NVFP4 = NVIDIA 4-bit 浮点格式（3 位指数 + 1 位尾数）
  精度比纯整数 INT4 好，专为 B300 新一代 GPU 优化
```

**对 FDE**：量化 = 用精度换容量。生产需验证质量损失（文章用 LongBench V2 64.01 验证）。

---

## ② 推理引擎层（vLLM 适配）

### 5. MRV2（Model Runner V2）—— 为什么 MoE 要手动开

**Model Runner**：vLLM 内部"准备输入 → 调模型 forward → 收集输出"的执行器。V2 重构了执行路径，更短更快。

```
MRV2 自 v0.25.0 起是 Dense 模型默认（稳定验证充分）
MoE 模型默认不开 → 执行路径更复杂（Router/多专家/EP），风险高
  → 显式 VLLM_USE_V2_MODEL_RUNNER=1 才启用
```

**MRV2 的三个生产关键收益**：
```
① 启动预热: DSA Indexer 的 Prefill Metadata Kernel 加入预热
   → 第一条请求不触发 Triton JIT 编译（避免冷启动延迟尖峰）
② Local Argmax Reduction（PR #46448）:
   → 多 GPU MTP 时 Draft Token 生成不需要对整词表 AllGather
   → TP 通信量从"词表大小"降到"2 × TP Size"
③ 动态推测长度 + Full CUDA Graph（PR #45953）:
   → 推测长度变化时不再频繁 miss CUDA Graph
```

---

### 6. EP（Expert Parallelism）专家并行

**为什么 Dense 的 TP/PP/DP 不够用**：MoE 的专家天然是"稀疏激活"，需要一种按"专家"来切分的并行方式。

#### EP 是什么

```
TP: 每个专家都切成 N 份分到 N 张 GPU → 所有 GPU 参与所有专家（浪费）
DP: 复制整个 MoE 模型 → 每张 GPU 装全部 744B 权重（显存爆炸）
PP: 按层切分 → 但不解决"专家怎么分散"的问题

EP: 不同专家放不同 GPU，Router 决定 token 用哪些专家 → 对应 GPU 才激活
通信方式: All-to-All（token 中间结果发给选中专家的所在 GPU）
```

#### EP 核心思想：按"专家"切

```
EP=4（4 张 GPU）:

  GPU0:  专家 1-64    ← 只装 1/4 的专家
  GPU1:  专家 65-128
  GPU2:  专家 129-192
  GPU3:  专家 193-256

token A 激活专家 10（GPU0）+ 专家 150（GPU2）
  → 只有 GPU0 和 GPU2 参与，GPU1/GPU3 空闲
token B 激活专家 200（GPU3）+ 专家 20（GPU0）
  → 只有 GPU3 和 GPU0 参与

→ 显存分摊（每卡只装 1/N 专家）+ 稀疏激活（每 token 只唤醒部分 GPU）
```

#### EP 与 TP/PP/DP 完整对比

| 维度 | TP | PP | DP | **EP** |
|------|----|----|----|--------|
| **切什么** | 权重矩阵（切碎） | 模型的层（按层切） | 整个模型（复制） | **专家（切分专家集合）** |
| **适用模型** | 所有模型 | 所有模型 | 所有模型 | **仅 MoE 模型** |
| **通信方式** | AllReduce（每层） | 层间传激活值 | 几乎无 | **All-to-All（token 发给所选专家）** |
| **显存效果** | 权重分片 | 层分片 | 不省显存 | **专家分摊，每卡只装 1/N 专家** |
| **GPU 利用率** | 高（全参与） | ~80%（bubble） | 高 | **按需激活（稀疏）** |

**最本质的区别**：TP/PP/DP 是"所有 GPU 都在为每个 token 工作"（只是分工不同），EP 是"每个 token 只唤醒部分 GPU"——这正是 MoE 稀疏激活的本质。

#### EP 的代价：All-to-All 通信

```
token A 在 GPU0 计算，但激活的专家 150 在 GPU2
  → 必须把 token A 的隐藏状态从 GPU0 发到 GPU2
  → All-to-All（每个 GPU 都可能跟任意其他 GPU 通信）

Router 选中的专家分布越散，通信越多
→ "Expert Dispatch/Combine 通信直接位于关键执行路径上"
```

#### 文章关键结论

```
EP 只有专家分布到足够多设备才有收益
  → TP2 + EP 不如单纯 TP2（2 张 GPU 时 All-to-All 开销 > 收益）
  → 专家太少，通信成本摊不薄
Decode 侧 DEP8 = Decode 阶段 8 路专家并行
```

#### 比喻

```
TP   = 一道菜 4 个厨师一起做（每人切一半食材，最后拼起来）
PP   = 一道菜分 4 道工序，每个厨师管一道（前面做完传后面）
DP   = 4 个厨师各做一整道菜（各自服务不同客人）
EP   = 一个"共享厨房"，有很多厨师（专家），每个客人只叫其中 2 个厨师做菜
        → 客人（token）来了，只唤醒被点名的厨师，其他厨师休息
```

#### 生产中如何组合

```
通常 TP（机内） + EP（专家并行）+ DP（多副本）组合使用
  机内: TP 处理权重切分 + EP 处理专家分布（都用 NVLink）
  跨机: DP 复制多份
  FlashInfer A2A 优化 EP 通信（见下一节）
```

---

### 7. FlashInfer A2A Backend —— MoE 通信优化

**问题**：EP 下每个 token 要发数据给选中专家的 GPU（All-to-All），在 Decode 关键路径上。

```
默认 EP Backend: AllGather / ReduceScatter（通信量大）
FlashInfer NVLink A2A: flashinfer_nvlink_two_sided
  → 直接利用 NVLink 双向带宽做 All-to-All → TPOT 再降 ~4%
  → 同机内多 GPU 之间，NVLink(900GB/s) 远快于跨机网络
```

---

### 8. IndexerCache —— 让 MTP 在 DSA 上真正可用

**问题**：DSA 的 Indexer 在 MTP 每个 Draft Step 都要重新执行，稀疏检索成本随上下文增长。

```
MTP 生成 N 个 Draft token: 每个 Step 都重跑 Indexer，很贵
IndexerCache（PR #44420）: index_share_for_mtp_iteration
  → 第一个 Draft Step 算出的 Top-K 索引，后续 Draft Step 直接复用
```

**文章强调**：IndexerCache 不只是省计算，还是**高并发下维持 MTP Acceptance Rate 的关键机制**。

---

## ③ 推测解码专项

### 9. 推测解码（Speculative Decoding）原理

**先理解问题**：Decode 阶段逐 token 生成，生成 N 个 token 就要跑 N 次完整 forward pass。

```
"今天天气" → forward → "真"
"今天天气真" → forward → "好"
"今天天气真好" → forward → "晴"
→ GPU 一次能算很多，但自回归限制每次只敢走一步
```

**推测解码的核心**：小而快的模型先"猜"多个 Draft Token，大模型一次性验证，一次 forward 产出多个 token。

```
① Draft 模型快速猜 N 个: ["真","好","晴","朗"]   ← Draft Token
② 大模型一次 forward 验证这 4 个
③ 前 3 个对了，第 4 个错了 → 采纳前 3 个，拒绝第 4 个
④ 从错的第 4 个重新开始（用大模型自己结果兜底）
→ 一次 forward 产出 3 个 token，原来要 3 次 → 吞吐 2-3x
```

**为什么猜错也不亏**：
```
猜对的 Draft Token 白赚（省下 N-1 次大模型 forward）
猜错的用主模型自己结果兜底 → 输出质量和大模型单独生成完全一致
只要 Acceptance Rate 够高，省的大模型计算 >> 多花的小模型计算
```

**关键术语**：
```
Draft Model / Proposal Token: 负责"猜"的模型 / 候选 token
Verification: 大模型一次性验证
Accepted Length: 一次验证通过几个 token（越大越高效）
Acceptance Rate: 猜中被采纳的比例（Decode 侧最早期的告警信号）
```

### 10. Draft Token 与各种"猜"的方案

**Draft Token 是结果，猜的方式有多种**：

```
推测解码（通用机制）:
  ┌──────────────────┐   ┌─────────────────────────┐
  │  Draft Token      │   │  "怎么猜" 的实现方案      │
  │  （被猜的候选）     │   │                         │
  │  是"结果/名词"     │   │  MTP   ← GLM-5.2 自带   │
  │                   │   │  EAGLE ← 预测隐藏状态     │
  └──────────────────┘   │  DFLash ← 专门训练小模型  │
                         │  DSpark ← 专门训练小模型  │
                         └─────────────────────────┘
```

```
MTP: GLM-5.2 自带预测头当 Draft，不用额外小模型（文章当前方案）
EAGLE: 基于特征自回归预测下一层隐藏状态（精度高，需额外训练小模型）
DFlash: 训练专门 Draft 模型，数据更广
DSpark: 训练专门 Draft 模型，数据更广（文章列为"下一步方向"）

共同点: 猜得更准 → 一次验证通过更多 token → 更快
```

**为什么文章"下一步"才提 DSpark**：
```
当前用 MTP: ✅ 不用额外训练小模型，部署简单，GLM-5.2 原生支持
           ❌ 猜的准确率受限于主模型自身预测头
未来换 DSpark: ✅ 专门训练的 Draft 猜得更准 → Acceptance Rate 更高
              ❌ 需要额外训练 + 部署一个小模型 → 复杂度高

文章原话: "Draft Model 决定能预测多远，
          Serving Runtime 决定这种预测多大程度真正有成本效益"
```

### 10.1 DSA（稀疏注意力）vs DSpark（推测解码）—— 别混淆

名字相近但**加速的原理完全不同**：

```
DSA 稀疏注意力 → 优化"Attention 怎么算"
  每个 query 只关注 Top-K 个历史 key → 计算量 O(n²) → O(n·K)
  属于"降低单次计算成本"（每步算得更少）

DSpark 推测解码 → 优化"一次生成几个 token"
  猜多个 Draft Token，一次 forward 产出多个 → 减少 forward 次数
  属于"一步产出更多"（每步跨更多级）

比喻:
  DSA    = 每步下楼梯跨的格数变少（算得更少）
  DSpark = 一步跨好几级楼梯（产出更多）
  两个维度独立，可叠加使用
```

**完整概念对照**：

| 概念 | 属于哪层 | 一句话 |
|------|---------|--------|
| 推测解码 | 机制 | 小模型猜 + 大模型验证的整体加速思路 |
| Draft Token | 结果 | 被猜出来的候选 token |
| Acceptance Rate | 指标 | 猜中被采纳的比例，越高越快 |
| MTP | 实现方案 | GLM-5.2 自带预测头当 Draft（当前） |
| EAGLE | 实现方案 | 预测隐藏状态而非直接猜 token |
| DFLash | 实现方案 | 专门训练的 Draft 模型 |
| DSpark | 实现方案 | 专门训练 Draft 模型，数据更广（下一步） |
| DSA | 注意力机制 | 稀疏化 Attention，降低单步成本（别与 DSpark 混淆） |

### 11. Speculative Padding —— 文章核心优化

**问题（Mixed Batch）**：
```
新请求到 Decode 节点第一个 Step 只算 1 个 token
其他 MTP 请求每步算 1+N 个 token
→ 计算形状不同 → 混合 Batch → 无法用 Full CUDA Graph → 退慢路径
DP 模式还要求各 Rank 一致 → 慢路径被持续触发
```

**方案**：
```
给新请求第一个 Decode Step 补虚拟推测 token（Dummy Speculative Tokens）
填充到 1+N，与其他请求一致 → 保持统一形状 → 持续跑 Full CUDA Graph
→ TPOT 40ms → 22ms（最大单项收益）
```

---

## ④ 生产配置层

### 12. 完整配置速查（文章最终方案）

```bash
# ===== Decode 实例 =====
VLLM_USE_V2_MODEL_RUNNER=1    # MoE 手动开 MRV2
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'  # 只编译 Decode 形状
--max-num-batched-tokens 1024 # 配合 FULL_DECODE_ONLY
num_speculative_tokens=3      # Decode 侧 MTP 深度
# EP Backend 用 FlashInfer A2A

# ===== Prefill 实例 =====
num_speculative_tokens=1      # Prefill 侧浅推测（快速生成 KV 交接）
```

```
VLLM_USE_V2_MODEL_RUNNER=1     → MoE 启用 MRV2
FULL_DECODE_ONLY               → 只录 Decode 形状 CUDA Graph（Decode 实例无需 Prefill 形状）
num_speculative_tokens=3/1     → 推测深度不对称：Decode 深（延迟关键路径），Prefill 浅（快速交接）
--max-num-batched-tokens 1024  → 配合 FULL_DECODE_ONLY 的 token 预算
```

### 13. Triton JIT 编译冷启动

```
Triton: GPU 内核自动生成框架（类似 PyTorch JIT）
问题: 第一条请求可能触发 Triton 编译（几百 ms 慢）
  → Benchmark 测不出（预热已完成），生产滚动部署后每次都有冷启动尖峰
方案: MRV2 启动预热把关键 Kernel 提前编译好
```

### 14. PCP 虚拟批处理 + Context Parallel（下一步方向）

```
Context Parallel（CP）: 超长上下文切到多张 GPU 并行
PCP 虚拟批处理: 单个请求拆成多个虚拟 Batch Row，多个 CP Rank 并行
  → 只聚合 MLA Latent Cache 和 DSA Indexer Cache（不传全部激活）

初步测试: 32K Prefill, TP=2+PCP=2 → 单卡吞吐 5.03K → 6.9K Token/s（+37%）
意义: 可能解决 KV Cache 容量 vs 单 GPU 利用率权衡
```

---

## 概念关联图（怎么串起来）

```
GLM-5.2 (744B MoE + DSA + MTP + NVFP4)
   │
   ├─ MoE → 需要 EP 并行（专家切分） → 需要 FlashInfer A2A 优化 All-to-All 通信
   ├─ DSA → 需要 Indexer → MTP 时需 IndexerCache 复用索引
   ├─ MTP → 生成 Draft Token → 推测解码 → 需要 Speculative Padding 保持 CUDA Graph
   ├─ NVFP4 → 需要 372GB 显存 → 决定并行拓扑（4P1D 而非 TGS 最优）
   └─ 全部 → 需要 MRV2 (VLLM_USE_V2_MODEL_RUNNER=1) 适配
```

## 面试考点提炼

| 概念 | 一句话回答 |
|------|-----------|
| MoE | 总参数大、每 token 只激活部分专家，知识容量与计算成本解耦 |
| DSA | 用 Indexer 粗筛 Top-K 历史 key，Attention 从 O(n²) 降到 O(n·K) |
| MTP | 一次预测 N 个 token，训练加速 + 推理时当推测解码的 Draft |
| Draft Token | 小模型/机制先猜出的候选 token，大模型验证后采纳或拒绝 |
| EP | 专家分散到不同 GPU，Router 决定激活哪些，用 All-to-All 通信 |
| EP vs TP/PP/DP | 前三种"所有 GPU 为每 token 工作"，EP 是"每 token 只唤醒部分 GPU" |
| DSpark vs DSA | DSpark=推测解码（一次产多 token），DSA=稀疏注意力（单步算更少），独立维度 |
| NVFP4 | NVIDIA 4-bit 浮点，744B 权重只需 372GB |
| MRV2 | vLLM 新执行路径，MoE 需 VLLM_USE_V2_MODEL_RUNNER=1 开启 |
| IndexerCache | 复用 DSA Indexer 的 Top-K 索引，让 MTP 成本效益为正 |
| Speculative Padding | 补虚拟推测 token 消除 Mixed Batch，保持 Full CUDA Graph |
| Acceptance Rate | 推测被采纳比例，Decode 侧最早期告警指标 |
