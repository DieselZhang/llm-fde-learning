# 大模型确定性输出与 Batch Invariance 详解

> **来源**：微信公众号《AI Infra进阶：如何让大模型输出确定的结果》（作者：binnnliu）
> **原文**：https://mp.weixin.qq.com/s/KUIEl2CF43bLqujvQ0psrw
> **笔记类型**：AI Infra 进阶 -- 浮点数非结合律、GEMM 优化与推理确定性的系统分析

---

## 一、问题场景

```
同一个 prompt、同一套硬件、temperature=0、random seed 锁定
→ 为什么两次请求输出可能不同？

答案：是也不是

✅ Run-to-run 确定性：同一时间、没有其他请求时，多次请求输出相同
❌ Batch Invariance（批次不变性）：同一时间、有其他请求时，输出可能不同
```

**核心洞察**：不同的请求之间 Attention 是严格物理隔离的，不存在信息污染。结果波动的根源不在信息干扰，而在--

> **推理引擎底层的动态组批与算子调度策略引发了浮点加法的顺序变化**

---

## 二、根因：浮点数加法不满足结合律

### 2.1 数学本质

```
(a + b) + c ≠ a + (b + c)

原因：FP16/BF16 的精度有限（FP16 尾数位只有 10bit）
大规模累加时，加法顺序不同 → 中间结果的舍入不同 → 最终结果不同
```

**FP16 具体例子**：

```
2048 + 1 = ?（FP16 下）
  2048 的 ULP（相邻可表示数的步长）= 2
  2048 + 1 = 2049，但 2049 无法用 FP16 精确表示
  触发"向偶数舍入"→ 结果被强行舍入为 2048
  → 2048 + 1 + 1 ≠ 2048 + (1 + 1)
```

### 2.2 哪些操作会改变浮点加法顺序

**GEMM (矩阵乘法)** 是主要来源。GPU 上的 GEMM 充满了为**掩盖内存墙（Memory Wall）**而做的优化，这些优化会重塑"累加拓扑"（Reduction Tree）。

```
内存墙：GPU Tensor Core 算力增长远超显存带宽增长
→ GPU 大部分时间在等数据搬运，计算单元空闲
→ 系统被迫用各种"化整为零、异步掩盖"策略
→ 这些策略的配置随 batch size 动态变化
→ 配置变化 → 累加顺序变化 → 输出变化
```

---

## 三、GEMM 的 Batch Invariance -- 哪些参数影响确定性

### 3.1 GEMM 优化策略全景与确定性影响

| 优化策略 | 硬件目标 | 核心作用 | 影响确定性？ |
|---------|---------|---------|-----------|
| BLOCK_M / BLOCK_N | SMEM / L1 | 子块大小（空间任务映射） | **无影响**（只是决定哪些元素打包在一起） |
| GROUP_M (Swizzle) | L2 Cache | CTA 调度顺序优化 | **无影响**（只改变"先算谁后算谁"） |
| **BLOCK_K** | SMEM | K 维度步长 | **引入确定性误差**（不同 config 结果不同，同 config 恒定） |
| **SPLIT_K** | SM 利用率 | K 维度多 CTA 并行切分 | **引入非确定性误差**（Atomic Add 顺序随机不可控） |
| num_warps | Registers | 资源分配（分母逻辑） | 无影响 |
| num_stages | HBM 延迟掩盖 | 流水线级数 | 无影响 |

### 3.2 BLOCK_K 如何改变累加顺序

```
BLOCK_K = 32: 每轮循环累加 32 个元素 → reduction tree 形状 A
BLOCK_K = 64: 每轮循环累加 64 个元素 → reduction tree 形状 B
→ 不同 config 产生不同结果，但同一 config 结果恒定
```

**特殊场景 -- Warp-level K-Slicing**：
```
大矩阵（如 128×128 tile）：各 Warp 在 K 维度独立串行，互不干涉
  → BLOCK_K 变化不产生 Batch Variance ✅

小矩阵（如 32×32 tile）：M/N 并行度不足
  → 多个 Warp 被迫在 K 维度切分，在共享内存做 Reduction Tree
  → BLOCK_K 变化 → 规约树改变 → Batch Variance ❌
```

### 3.3 SPLIT_K 为什么是非确定性的

```
SPLIT_K: 把长 K 维度切给多个 CTA 并行计算
合并方式 1 (Atomic Add): GPU 硬件调度 CTA 顺序完全随机 → 加法顺序随机
合并方式 2 (Workspace Reduction): 规约树形状随 split_k 段数变化
→ 都会导致不同 batch 下结果不同
```

### 3.4 各 GPU 架构差异

```
SM8x (Ampere): 需要同时锁定 BLOCK_K 和 SPLIT_K
  → 小矩阵下 Warp-level K-Slicing 可能被触发
  → cuBLASLt 不支持禁用 BLOCK_K，只能换 Triton 实现

SM90/SM100 (Hopper/Blackwell): 只需要锁定 SPLIT_K
  → TMA + WGMMA，Warp Group 共用 accumulator
  → 无 Warp-level K-Slicing 问题
  → "the only source of batch variance is split-k"
```

---

## 四、vLLM 的解决方案

### 4.1 核心开关

```bash
# 环境变量
VLLM_BATCH_INVARIANT=1

# 对所有 Linear 层生效
# 代码: if VLLM_BATCH_INVARIANT: return linear_batch_invariant(x, weight, bias)
```

### 4.2 matmul_persistent Kernel -- 确定性的关键

```
matmul_persistent 的做法:
  ① 硬编码 config，彻底关掉 @triton.autotune
  ② 每种 dtype 一套固定参数，特别是 BLOCK_K 与 M/N/K 无关
  ③ 没有实现 split-k → 不存在 Split-K 乱序

configs = {
    torch.bfloat16: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8
    },
    # fp16 / fp32 各有自己的固定配置
}
```

### 4.3 分架构处理

**SM80 (Ampere)**：
```
cuBLASLt 不支持禁用 BLOCK_K
→ 换成 Triton persistent matmul
→ 覆盖所有 aten::mm/addmm/matmul/linear
→ 固定 BLOCK_K，关闭 autotune
→ ⚠️ compile mode 下 aten override 不生效
```

**SM90/SM100 (Hopper/Blackwell)**：
```
bf16/fp16 Linear 层 → 走 matmul_persistent
裸 torch.mm → 走 cuBLASLt，通过 API 设置确保 SPLIT_K=1:
  preferred_blas_library(backend="cublaslt")
  CUBLASLT_REDUCTION_SCHEME_NONE  # Split-K=1

fp8: CUTLASS persistent config 或 workspace=1 限制 Split-K
fp4: PersistentScheduler（不拆 K）+ 固定配置
```

### 4.4 RMSNorm 的 Batch Invariance

```
问题: block_size 影响 BlockReduce 树形规约的形状 → 浮点加法顺序变化

vLLM 处理:
  - 开启 batch invariant: 使用 Triton rms_norm kernel（固定逻辑）
  - block_size 钉死在 1024（不使用动态 block_size）
  - 不同后端/dtype/compile 模式有对应的 dispatch 路径
```

---

## 五、对 FDE 的启示

### 核心理解

```
确定性 ≠ 温度 = 0
  温度 = 0 只是关了随机采样
  浮点算术的非结合律来自更底层（GPU 硬件指令调度）

生产环境获得确定性输出的条件:
  ① VLLM_BATCH_INVARIANT=1
  ② GEMM 不走 autotune（固定 BLOCK_K）
  ③ SPLIT_K 被禁用
  ④ 根据 GPU 架构了解底层差异
```

### 哪些场景需要确定性

| 场景 | 确定性要求 |
|------|-----------|
| 强化学习 Rollout | 严格--实验必须可复现 |
| 生产 AB 测试 | 需要--不同配置对比需隔离变量 |
| Benchmark 对比 | 需要--排除"随机性"干扰 |
| 一般聊天对话 | 不必须 |
| 代码补全 | 不必须--多个候选反而更好 |

### 性能代价

```
matmul_persistent: 固定 config → 不用 autotune 挑最优参数
  → 大部分情况下接近最优，极端 shape 下可能有 5-15% 性能损失
  → 但换来完全确定的结果

建议:
  测试环境或强化学习 → 开启 VLLM_BATCH_INVARIANT=1
  一般生产服务 → 不需要（吞吐优先）
```

---

## 核心概念速查

| 概念 | 一句话 |
|------|--------|
| Batch Invariance | 同一时间、不同请求组合下，同一 prompt 输出相同 |
| 浮点非结合律 | (a+b)+c ≠ a+(b+c)，FP16/BF16 精度有限导致 |
| BLOCK_K | K 维度分块步长，改变它 → 确定性误差（同 config 结果恒定） |
| SPLIT_K | K 维度多 CTA 并行切分，改变它 → 非确定性误差（合并顺序随机） |
| matmul_persistent | vLLM 的确定性 GEMM kernel：硬编码 config、无 autotune、无 split-k |
| SM80 vs SM90 | SM80 需锁 BLOCK_K+SPLIT_K；SM90+ 只需禁 SPLIT_K |
| VLLM_BATCH_INVARIANT | 环境变量开关，让 vLLM 走确定性执行路径 |
