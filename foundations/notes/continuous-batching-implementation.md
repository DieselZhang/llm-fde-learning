# Continuous Batching 底层实现笔记

> Continuous Batching 不是"动态加入 batch"这么简单。核心在于如何让不同长度、不同阶段的请求在一次 GPU forward pass 中同时计算。

---

## 1. 核心挑战

```
请求A: Decode 阶段，已有 5 个历史 token，只需算 1 个新 token
请求B: Prefill 阶段，prompt 有 4 个 token，需要全量计算
请求C: Decode 阶段，已有 3 个历史 token，只需算 1 个新 token

目标: 一次 forward pass 全部处理
```

## 2. 五个底层机制

### 机制一：超序列拼接

把所有请求的 token 拼成一个一维数组：

```
input_ids = [A_5, B_0, B_1, B_2, B_3, C_3]
             └A─┘ └────── B ────────┘ └C─┘
             1 token  4 tokens        1 token

GPU 看到的是一个 [6, hidden_dim] 的输入，不需要知道"这是 3 个不同请求"
```

### 机制二：Position IDs

拼接后 token 在数组中的位置不等于它在原始序列中的真实位置：

```
input_ids 数组:     [A_5, B_0, B_1, B_2, B_3, C_3]
在 batch 中的索引:    0     1    2    3    4    5
position_ids     =  [5,    0,   1,   2,   3,   3]
                     ↑                         ↑
                请求A 的第 5 个 token        请求C 的第 3 个 token
```

RoPE 按 position_ids 旋转 Q 和 K，不需要知道"前面有几个 chunk"。

### 机制三：Block Diagonal Attention Mask

拼接在一起的 token 绝对不能互相看到。每次 forward pass 动态构造块对角 mask：

```
Attention 矩阵:
            A_5  B_0  B_1  B_2  B_3  C_3
      A_5 [  1    0    0    0    0    0  ]  ← 只看自己和历史
      B_0 [  0    1    0    0    0    0  ]
      B_1 [  0    1    1    0    0    0  ]  ← Prefill 内部可以互看
      B_2 [  0    1    1    1    0    0  ]
      B_3 [  0    1    1    1    1    0  ]
      C_3 [  0    0    0    0    0    1  ]

同请求的 token 可以互相看到；跨请求的 Attention 被 block 为 0
```

### 机制四：Block Table -- 历史 KV Cache 访问

每个请求有独立的 block_table，指向物理上不连续的 KV Cache blocks：

```
请求A: block_table = [42, 137, 3]  → 3 个物理 block
请求B: block_table = [7, 8]        → Prefill 刚写入
请求C: block_table = [99]          → 1 个物理 block

Attention Kernel 根据各请求的 block_table 加载对应物理 block
→ 物理上不连续，但逻辑上每个请求看到完整的 KV Cache
```

### 机制五：Iteration-level 调度

```
Scheduler 每次迭代:
  ① 检查 waiting queue → 新请求进来 → 分配 KV blocks → waiting→running
  ② 检查 running queue → 已有请求继续
  ③ 准备数据 → input_ids + position_ids + attn_mask + block_tables
  ④ 一次 Forward Pass → 所有 token 同时计算
  ⑤ 采样 → 各请求生成新 token
  ⑥ 后处理 → 完成的释放 blocks，未完成的留在 running queue
```

### 为什么 Prefill 和 Decode 能混在一起

在 Attention 层面，两者只是"长度不同"的矩阵运算。Block Diagonal Mask 确保它们互不干扰。混合 batch 的额外开销（构造 mask + 拼接 input）相比较 forward pass 的计算时间几乎可忽略。

## 3. Chunked Prefill 融入

长 prompt 按 `max-num-batched-tokens` 拆成多个 chunk，chunk 间穿插 Decode：

```
Step M:   Prefill chunk tokens[0:4095]     → 写 KV blocks → 回到 waiting
Step M+1: Decode batch (其他短请求)         → 穿插
Step M+2: Prefill chunk tokens[4096:8191]  → 追加 KV blocks
Step M+3: Decode batch                     → 穿插
...直到全部 prompt 处理完 → 进入 running → 开始 Decode
```

## 4. Chunked Prefill 的信息传递

Chunk 之间**只需要传递 KV Cache**，不需要传递 hidden_states：

- Transformer 每层是"无状态函数"，只依赖输入 token + 历史 KV Cache
- 每个 chunk 自带正确的 `position_ids`（如 Chunk2 从 4096 开始）
- 历史 token 的 hidden_states 对后续计算没用--只有 K 和 V 需要在 Attention 层中被读取

```
Chunk 1 唯一产出 = tokens 0~4095 在 80 层的 K 和 V 矩阵 → 写入 GPU 显存
Chunk 2 启动时: position_ids = [4096, ...], 从显存加载 Chunk 1 的 K,V
```

## 5. 关键参数关系

| 参数 | 含义 | 约束对象 |
|------|------|---------|
| `max-model-len` | 单请求最长 token 数 | 每个请求的 KV blocks 上限 |
| `max-num-batched-tokens` | 单次 forward 总 token 数 | 临时 activation 峰值 |
| `max-num-seqs` | 最大并发请求数 | 调度器同时处理的请求数 |

**三者相互独立：max-model-len 可以远大于 max-num-batched-tokens（1M 上下文模型正是如此，Chunked Prefill 拆分了处理）。**

### 1M 上下文模型配置示例

```bash
--max-model-len 1048576          # 单请求 1M tokens 上限
--max-num-batched-tokens 8192    # 每批 token 数极小（防止 activation OOM）
--max-num-seqs 8                 # 并发极小（每个长请求占大量显存）
--enable-chunked-prefill         # 必须！1M 不可能一次 Prefill
--enable-kv-cache-offloading     # 显存放不下 → 溢写到 CPU/SSD
--block-size 32                  # 增大 block 减少 block table 条目
```

## 6. 核心总结

| 机制 | 作用 |
|------|------|
| 超序列拼接 | 不同请求的 token 拼成一维数组，GPU 一次处理 |
| Position IDs | 告诉 RoPE 每个 token 在原始序列中的真实位置 |
| Block Diagonal Mask | 阻止跨请求的 Attention 泄漏 |
| Block Table | 非连续物理 KV Cache 的逻辑映射 |
| Iteration-level 调度 | 每轮重新决定 batch 组成 |
