# KV Cache 深度解析笔记

> 从底层数学原理到生产环境缓存策略的完整知识体系

---

## 目录

1. [KV Cache 是什么](#1-kv-cache-是什么)
2. [为什么需要 KV Cache](#2-为什么需要-kv-cache)
3. [KV Cache 存什么、有多大](#3-kv-cache-存什么有多大)
4. [Prefill 和 Decode 中的 KV Cache](#4-prefill-和-decode-中的-kv-cache)
5. [缓存命中机制](#5-缓存命中机制)
6. [KV Cache 生命周期与淘汰策略](#6-kv-cache-生命周期与淘汰策略)
7. [PD 分离中的 KV Cache 传输](#7-pd-分离中的-kv-cache-传输)
8. [生产环境中的缓存行为](#8-生产环境中的缓存行为)
9. [Agent 设计启示](#9-agent-设计启示)
10. [显存预分配过程](#10-显存预分配过程)
11. [OOM 的真正原因](#11-oom-的真正原因)
12. [utilization 详解](#12-utilization-详解)
13. [临时 Activation](#13-临时-activation)
14. [KV Cache 大小估算公式](#14-kv-cache-大小估算公式)

---

## 1. KV Cache 是什么

### 从 Self-Attention 的三步计算理解 K 和 V

模型给每个 token 生成三个角色向量（通过 Wq、Wk、Wv 权重矩阵）：

| 角色 | 含义 |
|------|------|
| **Query (Q)** | "我是这个 token，我想知道应该关注前面的哪些 token" |
| **Key (K)** | "我是这个 token，这是我能提供的匹配线索" |
| **Value (V)** | "我是这个 token，这是我携带的实际信息" |

Attention 计算三步：

```
1. Q · K^T → 得到"相关性分数"（这个 token 和前面的每个 token 有多相关）
2. Softmax → 把分数变成权重
3. 加权求和 V → 用权重去聚合前面所有 token 的 Value，得到最终输出
```

**KV Cache 存的就是每一步计算中产生的 Key 和 Value 向量。**

---

## 2. 为什么需要 KV Cache

### 自回归生成的计算负担

生成第 N 个 token 时，模型需要 Attention 到前面全部 N-1 个 token。如果不缓存：

```
生成 token 1: 计算 Attention(1 个 token)    ← 1 次
生成 token 100: 计算 Attention(100 个 token) ← 100 次
生成 token 500: 计算 Attention(500 个 token) ← 500 次
总计算量: 1+2+3+...+500 ≈ 125,000 次  ← O(n²) 爆炸
```

### 有 KV Cache 之后

每个 token 的 K、V 只算一次，之后直接复用：

```
生成 token 1: 算 K,V(token_1) → 存起来，Attention(新 vs 全部历史)
生成 token 100: 只算 K,V(token_100) → 存起来，Attention(新 vs 全部历史)
生成 token 500: 只算 K,V(token_500) → 存起来，Attention(新 vs 全部历史)
总计算量: 500 次  ← O(n)，每步计算量恒定
```

**KV Cache = 用显存放中间结果，换取不再重复计算。空间换时间。**

---

## 3. KV Cache 存什么、有多大

### 存的是矩阵，不只是数字

```
每个 block (16 tokens) 存的 KV Cache 矩阵维度:

  Llama-3-70B 为例 (kv_heads=8, head_dim=128, layers=80):
  
  K: [16 tokens, 8 heads, 128 dims] × 80 layers
  V: [16 tokens, 8 heads, 128 dims] × 80 layers
  
  一个 block 的大小:
  = 16 × 8 × 128 × 80 × 2(K+V) × 2 bytes(BF16)
  ≈ 5.2 MB per block
```

### 每个 token 的 KV Cache 占用公式

```
每 token KV Cache 大小:
  = 2(K+V) × kv_heads × head_dim × layers × precision_bytes

70B 模型每 token:
  = 2 × 8 × 128 × 80 × 2
  = 320 KB per token

seq_len = 4096 时，单个请求的 KV Cache:
  = 4096 × 320 KB ≈ 1.25 GB

100 并发请求:
  = 100 × 1.25 GB = 125 GB  ← 超过单张 H100 的 80GB 总显存
```

**KV Cache 才是推理服务真正的显存大户，比模型权重更能决定并发容量。**

### 两层存储：哈希表（CPU）+ Block Pool（GPU）

```
CPU 内存 (Python 进程):
  cached_block_hash_to_block:
  ┌──────────────────┬─────────────────┐
  │  block_hash       │  physical_block_id │
  ├──────────────────┼─────────────────┤
  │  0x7a3f...       │  42               │
  │  0xb2e1...       │  137              │
  └──────────────────┴─────────────────┘
  存的不是 KV Cache 本身，而是"指纹 → 地址"的索引
  大小: 每个 entry ~100 bytes

GPU 显存 (HBM):
  Block Pool:
  ┌────────┬─────────────────────────────┐
  │block 42│ K: [16,8,128]×80 ≈ 2.6 MB  │
  │        │ V: [16,8,128]×80 ≈ 2.6 MB  │
  └────────┴─────────────────────────────┘
  存的才是真正的矩阵（BF16 浮点数）
  大小: 每个 block ~5.2 MB (70B 模型)
```

---

## 4. Prefill 和 Decode 中的 KV Cache

### Prefill：一次性生产全部 KV Cache

```
Prefill 阶段并行处理整个 prompt：

输入: "请帮我写一篇演讲稿" (9 tokens)

全部 9 个 token 同时计算每层的 K 和 V:
  Layer 0:  K₀[0..8],  V₀[0..8]
  Layer 1:  K₁[0..8],  V₁[0..8]
  ...
  Layer 79: K₇₉[0..8], V₇₉[0..8]

写入 KV Cache blocks → 采样生成第一个新 token

产出:
  - 全部 prompt 的 KV Cache
  - 第一个生成 token
  瓶颈: GPU 算力 (FLOPS)，计算密集型
```

### Decode：逐 token 消费并追加

```
Decode 阶段生成后续 token：

每步:
  1. 从显存读取全部历史 KV Cache (O(n) 读)
  2. 计算新 token 的 Q
  3. Q 与全部历史 K 做 Attention
  4. 加权 V → 生成新 token
  5. 新 token 的 K、V 追加写入 Cache (O(1) 写)

瓶颈: 显存带宽 (HBM bandwidth)，内存密集型
大部分时间在等显存把数据传过来
```

### 两者资源需求对比

| | Prefill | Decode |
|---|---|---|
| 资源瓶颈 | GPU 算力 (FLOPS) | 显存带宽 (HBM BW) |
| 显存压力 | 低 | 高 |
| GPU 利用率 | 算力 100%、显存闲置 | 算力 ~10%、显存趋于满载 |
| 每次处理 token 数 | N 个（全部 prompt） | 1 个 |
| 影响的核心指标 | TTFT | TPOT |

---

## 5. 缓存命中机制

### 因果注意力的铁律

Decoder-only 模型有一个硬约束：**每个 token 只能看到自己和前面的 token，绝对不能偷看后面。**

推论：`token_i` 的 KV Cache 只依赖于 `tokens 0~i`，与 `tokens i+1` 之后完全无关。

### 改前面的一个字会发生什么

```
原始: "你是 一个 乐于助人 的 助手。北京 的 首都 是 什么？"
位置:   0    1      2      3   4    5   6   7   8   9

修改: "你是 一个 友好 的 助手。北京 的 首都 是 什么？"
位置:   0    1    2'    3   4    5   6   7   8   9
         ↑ 只改了这个位置

缓存状态:
  位置 [0][1]      → 有效 ✅ (没变)
  位置 [2']...[9]  → 全部失效 ❌ (级联依赖)
```

**结论：改第 N 个 token，位置 0~(N-1) 缓存完好，位置 N 及之后全部失效。** 这就是 system prompt 必须放最前面且尽量保持不变的根本原因。

### vLLM：链式哈希（Hash Chain）

```
把 prompt 切成 16-token blocks:

  block_0: hash = H(prev=0, tokens[0..15])
  block_1: hash = H(prev=hash_0, tokens[16..31])
  block_2: hash = H(prev=hash_1, tokens[32..47])

链式哈希的特点:
  block_0 内容变了 → hash_0 变了
  → block_1 的 prev_hash 变了 → hash_1 也变了（即使 block_1 内容相同）
  → hash_1 的哈希表查不到 → 缓存全部 miss

优点: 保证"上下文一致性"，不会错误复用
缺点: 前面改了后面全 miss，极端场景复用率低
```

### SGLang：Radix Tree（前缀树）

```
所有请求共享一棵前缀树，每个节点存一个 token 的 KV Cache:

              root
               │
           "你是"       ← KV Cache
               │
           "一个"       ← KV Cache
            ╱    ╲
     "乐于助人"   "友好"  ← 分叉点，公共前缀自动共享
          │         │
         "的"      "的"

vLLM Hash vs SGLang Tree:
  场景 "同 system prompt + 不同 user prompt": 两者效果相同
  场景 "system prompt 有 A/B testing 两套": SGLang 分叉前自动共享，
  vLLM 从差异点往后全部 miss
```

### 缓存命中率公式

```
命中率 ≈ 公共前缀长度 / 新请求总长度

  - 短 prompt + 大段公共前缀 → 命中率极高
  - 长 prompt + 少量公共前缀 → 命中率低

❌ 错误直觉: "prompt 越长命中越高"
✅ 正确规律: "公共前缀占比越高命中越高"
```

---

## 6. KV Cache 生命周期与淘汰策略

### 没有 TTL（时间过期）

KV Cache 存的是 attention 中间结果。只要模型权重不变，这个结果永不变化。
**没有"过了 5 分钟自动过期"的设计。只有"这个结果还有没有人用"。**

### 生命周期绑定请求

```
请求到达 → 分配 blocks → Prefill 写入 → Decode 读写 → 请求结束 → 释放
                                                                  │
                                                    所有 blocks 回 free pool
                                                    或标记为"纯缓存"
                                                    (ref_count=0, in_hash_table=True)
```

### Block 的三种状态

| 状态 | ref_count | in_hash_table | 说明 |
|------|-----------|---------------|------|
| **空闲** (free pool) | 0 | False | 可以被新请求分配 |
| **使用中** | > 0 | True/False | 有活跃请求正在读写 |
| **纯缓存** | 0 | True | 无人用但哈希表指着。显存不够时最先淘汰 |

### 淘汰策略

| 策略 | 触发条件 | 做法 |
|------|---------|------|
| **正常释放** | 请求完成 (EOS / max_tokens) | blocks 进入 free pool，哈希表 entry 保留 |
| **LRU 淘汰** | 显存不够 + 有"纯缓存"blocks | 从最久未访问的纯缓存 block 开始清除哈希表引用，归还 free pool |
| **Recompute 抢占** | 显存不够 + 没有可淘汰的纯缓存 | 选一个正在运行的请求，释放其全部 blocks，请求回到 waiting queue 重新 Prefill |
| **Swap 抢占** | 同上，但配置了 Swap | 把 blocks 从 GPU 显存拷贝到 CPU 内存腾空间，稍后换回 |
| **Abort** | 用户断开连接 / 超时 | API Server 通知 Engine Core → 直接释放 |

### 引用计数防悬空指针

```python
# 只要 block 在哈希表中，就不会进 free pool
if block.in_hash_table and block.ref_count == 0:
    # 纯缓存状态：可以淘汰，但不能被新请求分配
    pass

# 淘汰时三步
hash_table.remove(hash)       # ① 从哈希表删除
block.in_hash_table = False   # ② 标记不再是缓存
free_pool.append(block.id)    # ③ 扔回 free pool

# 分配时还会做防御性检查
if block.id in reverse_index:
    del hash_table[reverse_index[block.id]]  # 清除残留
```

---

## 7. PD 分离中的 KV Cache 传输

### 传输通道

```
控制面（小数据量）:
  API Server ──ZMQ──▶ Engine Core
  传: request_id, token_ids, sampling_params (~几 KB)
  协议: ZeroMQ 消息队列

数据面（大体积）:
  Prefill GPU ──RDMA/NIXL/MoonCake──▶ Decode GPU
  传: KV Cache 矩阵 (每 block ~5MB)
  特点: GPU 显存到 GPU 显存直通，不经过 CPU 内存
```

**ZMQ 管"谁要算什么"（控制面），RDMA 管"算好的数据搬过去"（数据面）。两套通道各干各的。**

### 三种传输方案

| 方案 | 原理 | 带宽 | 适用场景 |
|------|------|------|---------|
| **NIXL** | NVIDIA 跨节点 GPU 内存直接拷贝 | 200-400 Gbps | 标准 PD 分离 |
| **MoonCake** | 字节跳动开源，异步流水线 | 同上但延迟更低 | 超大规模部署 |
| **裸 RDMA** | GPU→网卡→对端网卡→GPU | 200-400 Gbps | 自定义方案 |

### 传输量估算

```
短 prompt (128 tokens, 70B):
  KV Cache ≈ 40 MB → 100Gbps 网络 ≈ 3.2ms ← 可忽略

长 prompt (4096 tokens):
  KV Cache ≈ 1.25 GB → 100Gbps ≈ 100ms ← 有影响

超长 prompt (128K tokens):
  KV Cache ≈ 40 GB → 100Gbps ≈ 3.2s ← 严重瓶颈，必须 RDMA
```

### PD 分离中的双重生命周期

```
统一架构: blocks 只有一个"家"，请求结束 → 释放/纯缓存

PD 分离:
  Prefill 侧: 传输完成后 blocks 可保留为缓存 (ref=0, in_hash=True)
             LRU 淘汰，倾向保留（Prefill 节点显存压力不大）

  Decode 侧: 请求结束后释放，跟统一架构一样
```

---

## 8. 生产环境中的缓存行为

### 长时间离开后回来

```
上午 10:00 - 连续使用:
  system prompt + 对话历史持续命中 → 命中率 80%+

上午 10:05 离开 → 其他用户请求涌入 → LRU 淘汰你的对话历史

下午 2:00 - 回来看第一个请求:
  system prompt: 依然命中 ✅
  (所有人共享，高频访问，LRU 前端永驻)
  
  对话历史: 几乎肯定被淘汰 ❌
  (只有你访问，几小时没人碰，LRU 末端优先淘汰)

  整体命中率: system_prompt / 全部 ≈ 15-20%
```

### 真的需要"预热"吗？

```
❌ 不建议"预热请求":
  - 预热不减少总计算量（system prompt 反正要算一次）
  - 增加一次额外的 HTTP 往返、Tokenization、Scheduler 开销
  - 在并发高峰期，预热缓存可能在真正请求前就被挤掉 → 白做

✅ 正确做法:
  - 接受第一个请求稍慢（200ms 级别，几乎无感）
  - 第二个请求开始自动恢复（system prompt 缓存已重建）
  - 唯一例外：system prompt > 5000 tokens 且多租户 → 考虑 SGLang 保活
```

---

## 9. Agent 设计启示

### System Prompt 放在最前面（铁律）

```
✅ 好的设计:
  [system (固定)] [工具定义 (固定)] [上下文 (变化)] [用户输入 (变化)]
   └─ 缓存永驻 ─────┘└──────────────────────────────┘
                           └─ 从这里开始新算

❌ 不好的设计:
  [用户输入] [system prompt] [上下文]
  第一个 token 就不同 → 全部缓存失效 → 每次全量 Prefill
```

### 多轮对话自动受益

```
每轮是独立的 HTTP Request，但内容递增:

  第 1 轮: [system] + "问题1"
  第 2 轮: [system] + "问题1" + "回复1" + "问题2"
           └── 前缀完全命中 ──┘
  第 3 轮: [system] + 全部历史 + "问题3"
           └── 前缀命中 ──────┘

每轮自动复用前轮缓存，Agent 无需额外处理。
```

### SGLang vs vLLM 的缓存选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 同一 system prompt 多用户共享 | 两者均可 | 共享前缀命中率高 |
| system prompt 多版本 A/B | **SGLang** | Radix Tree 分叉前共享，vLLM 哈希全 miss |
| 简单问答，无共享前缀 | **vLLM** | 更成熟稳定 |
| Agent 多轮对话 + 长上下文 | **SGLang** | 前缀树天然适合增量 token 序列 |

---

## 10. 显存预分配过程

### 预分配在模型加载流程中的位置

模型权重加载到 GPU 后，vLLM 计算剩余显存，按 `gpu_memory_utilization` 比例一次性在 GPU 上分配 KV Cache Block Pool：

```
H100 80GB 显存布局（Llama-3-70B, TP=2）:

  CUDA Context + Driver:       ~1.5 GB   固定
  模型权重 (TP=2, 每 GPU 一半):  ~70 GB   固定
  临时缓冲区:                   ~1-2 GB   固定
  ═══════════════════════════════════════
  KV Cache Block Pool:          ~5-7 GB   预分配，固定大小
  预留空间 (1 - utilization):  ~1.5 GB   给 PyTorch、临时张量、NCCL 等
```

### 预分配计算流程

```python
# 1. 算可用显存
total_gpu_memory = 80 * 1024**3        # H100: 80 GB
model_weight_memory = torch.cuda.memory_allocated()  # 已占用的
available = total_gpu_memory - model_weight_memory   # 约 14.8 GB

# 2. 按比例分配给 KV Cache
cache_memory = available * 0.90         # --gpu-memory-utilization
# = 14.8 * 0.90 ≈ 13.3 GB

# 3. 算 block 大小
kv_per_token = 2 * 80 * 8 * 128 * 2  # 320 KB per token (70B)
block_size_bytes = kv_per_token * 16   # 5 MB per block

# 4. 算能分多少 block
num_gpu_blocks = 13.3 GB // 5 MB ≈ 2660 个

# 5. 一次性分配物理存储
k_cache = torch.empty(2660, 16, 8, 128, dtype=bf16, device="cuda")
v_cache = torch.empty(2660, 16, 8, 128, dtype=bf16, device="cuda")

# 6. 初始化 free block 队列
free_block_queue = [0, 1, 2, ..., 2659]
```

### 预分配之后不再变

Block Pool 大小固定，内部 block 只在"空闲/使用中/纯缓存"之间流转。**KV Cache 本身不会导致 CUDA OOM 崩溃——只会导致请求排队（no free blocks）。**

---

## 11. OOM 的真正原因

既然 KV Cache 是预分配的，**真正的 CUDA OOM 崩溃来自预分配之外、utilization 管不到的动态分配**：

### 五大不可控消费者

| 消费者 | 大小 | utilization 管得到？ | 触发场景 |
|--------|------|---------------------|---------|
| **PyTorch caching allocator 缓存** | ~1-3 GB 动态 | ❌ | 运行时膨胀，不缩小 |
| **临时 activation 张量** | ~0.1-5 GB 动态 | ❌ | 大 batch Prefill |
| **NCCL 通信 buffer** | ~0.1-1 GB 动态 | ❌ | 大 TP + AllReduce |
| **CUDA Graph 录制** | ~0.5-3 GB 一次性 | ❌ | 启动时录制 |
| **显存碎片化** | 动态 | ❌ | 找不到连续块 |

### PyTorch Caching Allocator 膨胀

PyTorch 有一层内部缓存：`torch.cuda.memory_reserved() - torch.cuda.memory_allocated()` = 缓存池大小。即使临时张量释放了，缓存池也不缩小（设计如此，为了性能）。大 Prefill 后缓存池扩张→侵占预留空间→下次大 batch OOM。

```python
# 监控方式
print(f"Allocated: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved(0)/1e9:.1f} GB")
print(f"Cached:    {(torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0))/1e9:.1f} GB")
# Cached 持续增长 = allocator 膨胀
```

---

## 12. utilization 详解

### utilization 只控制 KV Cache Block Pool

```
nvidia-smi 看到的 80 GB:
  ┌───────────────┐
  │ 模型权重: ~65GB│ ← 不管
  │ CUDA: ~1.5GB  │ ← 不管
  │ PyTorch 缓存池 │ ← 管不到
  ├═══════════════┤
  │ KV Cache Pool │ ← utilization 只控制这块
  │ ~11 GB        │
  ├───────────────┤
  │ 预留 10%       │ ← 给动态分配的缓冲
  │ ~1.2 GB       │
  └───────────────┘
```

### 为什么是 0.90？

```
0.80 → 太保守，KV Cache 太小 → 并发上不去
0.95 → 太激进，预留不够 → 大 batch Prefill 容易 OOM
0.90 → 工程实践的"甜点"

特殊场景调整:
  小模型 (7B, 单卡放得下): 0.95（权重占比小，剩余多）
  大模型 (70B, 显存紧张): 0.85-0.90
  PD 分离 Decode 节点: 0.95（不需要大 activation）
  PD 分离 Prefill 节点: 0.80-0.85（给 activation 留空间）
```

---

## 13. 临时 Activation

### Activation 是什么

Transformer 每层 forward pass 产生的中间结果——QKV 投影、attention scores、FFN 中间值。forward pass 期间必须在显存里，完成后释放。

### Prefill vs Decode 的 activation 差异

```
Decode: batch_size=N, 每请求 1 token
  每层 activation: ~500 KB × N
  N=256 时: ~128 MB  ← 很小

Prefill: batch_size=N, 每请求 M tokens
  每层 activation:
    attn_scores: [N, M, M]  ← 最大！
    QKV: N × M × hidden
    FFN: N × M × intermediate

  N=8, M=4096 时:
    attn_scores: 8 × 4096 × 4096 × 2 ≈ 256 MB per layer
    总峰值: ~3.5 GB  ← 危险！挤占预留空间
```

### --max-num-batched-tokens 的作用

限制单次 forward pass 的总 token 数 → 限制 activation 峰值 → 防止 OOM。

```bash
# 设置: 单次最多 8192 tokens
--max-num-batched-tokens 8192

# 4 个请求各 2000 token → 8000 total → 一起 Prefill ✅
# 4 个请求各 4000 token → 16000 total → 拆成 2 批 ✅
```

---

## 14. KV Cache 大小估算公式

### 标准 GQA 模型

```python
kv_per_token = (
    2                        # K + V
    * num_layers             # 每层都要存
    * num_kv_heads           # 注意：不是 attention_heads！
    * head_dim               # 通常 128
    * bytes_per_element      # BF16=2, FP8=1
)

# 一行代码
kv_per_token_kb = lambda layers, kv_heads, head_dim, bpe=2: (
    2 * layers * kv_heads * head_dim * bpe / 1024
)
```

### 常用模型速查

| 模型 | layers | kv_heads | head_dim | 每 token | 4K | 128K |
|------|--------|----------|----------|----------|-----|------|
| Llama-3-8B | 32 | 8 | 128 | 128 KB | 0.5 GB | 16 GB |
| Llama-3-70B | 80 | 8 | 128 | 320 KB | 1.25 GB | 40 GB |
| Qwen2.5-7B | 28 | 4 | 128 | 56 KB | 0.2 GB | 7 GB |
| Qwen2.5-72B | 80 | 8 | 128 | 320 KB | 1.25 GB | 40 GB |
| DeepSeek-V3 | 61 | MLA | - | ~137 KB | 0.5 GB | 17 GB |
| Llama-3.1-405B | 126 | 8 | 128 | **504 KB** | 2 GB | **63 GB** |

### 关键注意事项

- **用 num_kv_heads，不是 num_attention_heads**（GQA 模型 K/V head 少很多）
- **MoE 不影响 KV Cache**（FFN 参数多但 Attention 层一样）
- **MLA 架构有自己的公式**（DeepSeek-V3 等，KV 被压缩）
- **精度直接影响**：FP8=BF16 的一半，INT4=BF16 的 1/4

### 为什么传统 KV Cache 是连续的

Attention 计算的核心操作是矩阵乘法（Q × K^T），GPU 的矩阵乘法硬件要求数据在连续内存中才能高效 DMA 传输。最早的朴素实现用 `torch.cat` 追加新 token 的 K/V——自然产生连续内存。PagedAttention 的精妙在于没有打破 GPU 对连续数据的偏好：每个 block 内部 16 个 token 仍然连续，GPU 每次只看一个 block，通过 Block Table 把多个不连续 block 的逻辑顺序"组装"成连续视图。

| 概念 | 一句话 |
|------|--------|
| KV Cache | 存每层每个 token 的 Key 和 Value 矩阵 |
| 为什么需要 | 不用 → O(n²)；用了 → O(n) |
| 每 token 大小 | 70B ≈ 320KB；seq=4096 单请求 ≈ 1.25GB |
| Prefill 角色 | 一次性生产全部 KV Cache（计算密集） |
| Decode 角色 | 逐 token 读取全部历史、追加 1 个新 token（内存密集） |
| 缓存命中 | 公共前缀越长 → 命中率越高 |
| 没有 TTL | 不随时间过期，只被 LRU 淘汰 |
| 存活周期 | 请求内使用 → 请求结束纯缓存 → LRU 淘汰 |
| PD 传输 | RDMA/NIXL，GPU→GPU，不走 ZMQ |
| 代码翻译 | hash 值只是索引，真正的矩阵在 GPU 显存里 |
| 最佳实践 | system prompt 放最前、尽量短、保持不变 |
