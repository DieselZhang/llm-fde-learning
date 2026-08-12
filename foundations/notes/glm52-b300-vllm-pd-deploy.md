# 从 Day 0 到生产级 SLA：基于 vLLM 在 NVIDIA B300 上部署 GLM-5.2

> **来源**：微信公众号技术文章（DaoCloud 团队）
> **原文**：https://mp.weixin.qq.com/s/l0CpENAvEggmIOtiDsy-gw
> **笔记类型**：真实生产案例解析 —— P/D 分离 + 推测解码 + SLA 驱动的性能优化

---

## 文章核心信息

| 项目 | 内容 |
|------|------|
| **模型** | GLM-5.2（744B 参数 MoE，实际激活 ~40B） |
| **特性** | DSA 稀疏注意力（Sparse Attention）、原生 MTP 推测解码、NVFP4 量化 |
| **硬件** | NVIDIA B300 GPU |
| **拓扑** | P/D 分离（Prefill/Decode Disaggregation），4 Prefill + 1 Decode（4P1D） |
| **SLA 目标** | 平均 TTFT ≤ 2.5s，平均 TPOT ≤ 20ms（约 50 Token/s） |
| **成果** | TPOT 从 ~40ms 优化至 17ms |
| **团队** | DaoCloud，相关工作已合并至 vLLM v0.26.0 |

**核心方法论**：生产环境的成功不是"峰值吞吐高"，而是"**在不违反延迟目标的前提下，能稳定承载多少业务流量**"。所有配置搜索都以满足 SLA 为前提。

---

## 一、为什么优化 SLA 而不是峰值吞吐

### Colocated Serving（Prefill 与 Decode 共置）的问题

```
Prefill Chunk 不断插入到 Decode Batch 中
→ 每当长 Prompt 加入 Batch，都会拉长所有正在 Decode 请求的 Inter-token Latency
→ TPOT 的尾延迟取决于进入系统的 Prompt 长度分布
→ 而这是推理服务本身无法控制的
```

### P/D 分离的意义

```
P/D 分离将 Prefill 工作完全从 Decode 的关键路径中移除
→ TPOT 只与 Decode Batch 的组成有关
→ 这是能够实现严格 TPOT SLA 的基础
→ 是整个优化工作的起点，而不是众多优化手段中的一种
```

### 具体 SLA 目标

- 典型上下文长度：**16K–256K Token**
- 平均 TTFT ≤ 2.5s：用户发起请求到看到第一个 Token 的最大等待时间
- 平均 TPOT ≤ 20ms：对应约 50 Token/s 的流式输出速度（低于此速度用户能明显感到体验下降）
- 在满足上述两个硬约束的前提下，尽可能提升系统吞吐

### 并发规模（由请求速率自然形成）

```
--max-concurrency 1024 限制下:
  8K 输入:   ~700 并发请求
  16K 输入:  ~300 并发请求
  256K 输入: ~25 并发请求
  → 输入越长，Prefill 计算成本越高，能维持的并发越低
```

---

## 二、P/D 分离架构下优化 Decode 性能

### 初始状况

- Prefill 侧已达到目标，TTFT 有充足余量
- **瓶颈在 Decode 侧**：16K 输入 + 1K 输出时，平均 TPOT ≈ 40ms，P99 抖动大

### 2.1 根本原因：P/D 交接阶段的混合 Batch

```
问题出现位置: P/D 分离 × 推测解码 的相互作用

1. 请求通过 KVConnector 将 KV Cache 传输到 Decode 节点
2. 在 Decode 节点执行的第一个 Decode Step 只需计算 1 个 Token
3. 但 Decode 节点上已有的其他请求，启用 MTP 时每步按 1+N 个 Token 调度
4. 两类请求的计算形状（Shape）不同 → 形成混合 Batch（Mixed Batch）
5. 系统无法使用统一 Decode 的 Full CUDAGraph 快速路径
6. 只能退回到更慢的 Piecewise 或 Eager Execution 路径

DP 放大影响:
  在 DP 模式下，CUDA Graph 执行模式和 Padding 需要各 Rank 保持一致
  → 任意一个 DP Rank 收到新迁移请求，所有 Rank 都必须切换慢路径
  → P/D 分离架构中新请求持续进入，慢路径被持续触发
```

### 2.2 优化方案：Decode 侧 Speculative Padding

```
当一个请求到达 Decode 节点后的第一个 Decode Step 时:
  → 补充若干虚拟推测 Token（Dummy Speculative Tokens）
  → 将计算形状填充至 1+N，与 Decode Worker 中其他请求保持一致
  → 保持 Decode 阶段统一执行形状
  → 整个工作负载持续运行在 Full CUDA Graph 快速路径

不需要从 Prefill 节点传输任何已生成的 Token 或 Draft Token
（已由 vLLM 社区合并至 PR #45237）
```

### 2.3 性能收益

```
端到端平均 TPOT: ~40ms → ~22ms
（整个优化过程中收益最大的一项）

启示:
  最大性能损失未必来自某个具体 Kernel
  而可能出现在不同子系统的边界:
  请求状态、调度形状、CUDA Graph 执行模式的细微不一致
  会在 DP 规模 × 持续业务流量下被不断放大
```

---

## 三、Decode 侧进一步优化（TPOT 22ms → 17ms）

### 3.1 Model Runner V2（MRV2）：TPOT 再降 ~11%

```
MRV2 重构了运行时执行路径
  v0.25.0 起成为所有 Dense 模型默认
  GLM-5.2 是 MoE 模型 → 默认不启用，需显式设置:
    VLLM_USE_V2_MODEL_RUNNER=1

MRV2 带来的生产关键能力:
  ① PR #47285: GLM-5.2 DSA Indexer 的 Prefill Metadata Kernel 加入启动预热
     → 第一条生产请求不再触发 Triton JIT 编译
     → 避免冷启动延迟峰值（滚动部署后尤其明显）
  ② PR #46448: 多 GPU MTP 本地 Argmax Reduction
     → use_local_argmax_reduction 后，Draft Token 生成不需要对整词表 Logits AllGather
     → TP 通信量从与词表大小成正比，降到约 2 × TP Size
     → MTP/EAGLE/DFlash 等推测解码器都受益
  ③ PR #45953: 动态推测长度可与 Full CUDA Graph 配合
     → 减少 Draft Length 变化导致的 CUDA Graph Miss 和 Eager 回退
```

### 3.2 All-to-All Backend：TPOT 再降 ~4%

```
GLM-5.2 是 MoE 模型 → Decode 侧 DEP8，Expert Dispatch/Combine 在关键路径上

优化:
  默认 AllGather/ReduceScatter 的 EP Backend
  → 替换为 FlashInfer NVLink A2A Backend（flashinfer_nvlink_two_sided）

结论:
  EP 只有在足够多 Expert 分布到足够多设备时才能摊薄通信成本
```

### 3.3 CUDA Graph 模式

```
Decode 实例配置:
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --max-num-batched-tokens 1024

原因:
  Decode 实例无需为 Prefill 计算形状编译 CUDA Graph
  FULL_DECODE_ONLY → Decode 路径完整 CUDA Graph 覆盖 + 显著缩短启动编译时间
```

### 3.4 MTP 推测解码

```
Decode 侧: num_speculative_tokens=3
Prefill 侧: num_speculative_tokens=1

不对称是有意设计:
  Prefill: 目标是最快完成 KV Cache 生成并交接，深层推测收益有限
  Decode: 位于延迟关键路径，接受率高时深层推测能摊薄每个 Token 成本
```

---

## 四、Prefill 并行策略：为什么不选吞吐最高的配置

### 各配置对比（TGS = Tokens Generated per GPU，单 GPU 吞吐）

```
配置              | 8K 输入绝对吞吐 | 单 GPU TGS（相对）
TP1 DP4 EP（最终选） | 较高            | 第一名
TP1 DP2 EP         | -              | 最高（单 GPU 利用率）
TP2 + EP           | -              | 不如单纯 TP2
TP1 DP4 EP × 2    | 47,806 Token/s  | 第二（但用 2 倍 GPU）
```

### 关键结论

**1. TP2 + EP 不如单纯 TP2**
```
仅两张 GPU 规模下，引入 EP 的 All-to-All 通信开销 > 收益
EP 需要足够多 Expert 分布到足够多设备才能发挥优势
```

**2. 为什么选 TP1 DP4 EP 而非 TGS 最高的 TP1 DP2 EP**
```
TP1 DP2 EP 单 GPU 利用率最高
但每个实例仅 2 张 GPU，KV Cache 容量不足以支撑 GLM-5.2 的 100 万 Token 上下文

结论:
  不愿意为了 ~8% TGS 提升，牺牲模型最重要的能力之一（长上下文）
  最终: 以约 8% 单 GPU 效率损失，换取每实例 4 GPU 的 KV Cache 容量
```

> 💡 **生产 vs Benchmark 的核心区别**：Benchmark 追求峰值吞吐，生产优化需要考虑 KV Cache 容量、稳定性、SLA 等多重约束。这就是为什么"最高吞吐配置"不一定是最佳部署配置。

---

## 五、MTP + IndexerCache：如何提升 Acceptance Rate

### 5.1 IndexerCache（PR #44420）

```
不是传统 KV Cache，而是对 DSA Indexer 生成的 Top-K 稀疏索引进行复用

问题:
  最直接实现中，每个 MTP Draft Step 都要重新执行 Indexer
  稀疏检索计算成本随上下文长度增长
  → 消耗掉推测解码的大部分收益

方案:
  PR #44420: index_share_for_mtp_iteration
  → 第一个 Draft Step 算出的 Top-K 索引，后续 Draft Step 直接复用
  → MTP 在 GLM-5.2 上真正具有成本效益的前提
```

### 社区后续三项优化（共同提升高并发下 Acceptance Rate 稳定性）

```
① PR #45895: 跳过 Top-K Layer 时改进 Indexer 初始化，修复 GLM-5.2 MTP Normalization Loop
   测试: GLM-5.2-FP8 TP=8 下，平均 Accepted Length 从 ~3 → ~4
         Acceptance Rate ~60%，IFBench 74.62
② PR #47238: 批量请求共享索引缓冲区（Shared Index Buffer）布局优化
   第一个 Draft Step 后，仅保留每个请求最后一个 Query Token 的 Top-K 索引
   → Index Sharing 从单请求扩展到高并发 Batch 的关键
③ PR #47448: 确保 MTP 循环复用 Final Norm 后的 Hidden State
```

> 💡 **IndexerCache 双重价值**：既节省计算开销，也是高并发下维持 MTP Acceptance Rate 的关键机制。

### 5.2 组合配置的另外两项修复

```
① MRV2 调度分类（PR #47381）
   问题: 特定 Benchmark 下 TPOT 大幅波动（Issue #47239）
   根因: Uniform Decode 调度顺序中，推测解码步骤被错误分类为 Prefill，走到慢路径
   修复: 正确分类

② P/D 部署中异步 KV 加载的 Lookahead 处理（PR #46694）
   问题: GLM-5.2 + NIXL P/D + MTP 组合下的 Slot 分配时机
   修复: Decoder 等待远端 KV 传输完成后，再分配推测 Token 的 Slot
   意义: 正确处理最后一个不完整 KV Block 的边界情况
```

### 5.3 精度验证

```
最终配置运行公开 Benchmark，验证 NVFP4 + MTP + P/D 分离不降低输出质量
关键: LongBench V2 得分 64.01
  → 验证长上下文场景下 DSA 稀疏注意力和 IndexerCache 索引共享效果
  → 索引共享在长上下文保持稳定，推测解码性能提升未牺牲质量
```

---

## 六、上游进展（vLLM 社区协作）

### P/D 分离的二级缓存层（PR #42285，合并于 v0.25.0）

```
引入统一 CPU KV Cache 布局作为中间层
TieringManager 协调一级缓存层与 P/D Connector
→ 降低数据传输后端与模型执行路径之间的耦合
```

### PCP 虚拟批处理（PR #46570，合并于 2026-07-19）

```
将单个请求拆分为多个虚拟 Batch Row
在多个 Context Parallel（CP）Rank 上并行处理
仅对 MLA Latent Cache 和 DSA Indexer Cache 进行聚合

初步测试:
  GLM-5.2-NVFP4, 32K Prefill, TP=2 + PCP=2
  单卡 Prompt 吞吐: ~5.03K → ~6.9K Token/s（+37%）

意义: 可能改变第 4 节 KV Cache 容量 vs 单 GPU 利用率的权衡
```

---

## 七、可观测性：上线后如何验证 SLA

### P/D 分离的观测复杂度

```
一次请求的延迟被拆分到两个资源池:
  TTFT ← 主要由 Prefill Pool 决定
  TPOT ← 主要由 Decode Pool 决定
  两者之间还有一次 KV 传输

问题: 任何一个阶段性能下降，用户感知的只是"整个服务变慢了"
```

### 监控体系（Prometheus + Grafana）

```
① 每个资源池分别监控 TTFT 和 TPOT 的各百分位数
   → 判断问题发生在 Prefill 侧还是 Decode 侧的首要依据
② 监控 MTP Acceptance Rate 和 Mean Accepted Length（一级告警）
   → 最容易忽略，但最早出现异常的告警信号之一
   → Acceptance Rate 下降不触发错误，只让 TPOT 逐渐变差
③ 同时监控 Prefill/Decode 两侧的 KV Cache 利用率与 GPU 利用率
   → P/D 分离最大优势是两类资源独立扩缩容
   → 两组曲线相对关系 = 扩容决策依据
④ 监控 KV 传输延迟和队列深度
   → 判断节点间网络是否成为瓶颈
```

### 7.1 只有长期稳定性测试才能发现的问题

**现象**：vLLM 进程 RSS 持续线性增长，运行数十小时仍不收敛（721 GiB → 800 GiB）

**为什么 Benchmark 发现不了**：
```
① 增长速度极慢，需数小时连续运行才可观察
② 增长发生在主机内存（Host Memory），不是 GPU 显存 → GPU 监控全正常
③ 传统内存分析工具无法定位:
   EngineCore 启动时调用 gc.freeze()
   → 泄漏对象不出现在 gc.get_objects() 或 tracemalloc
   → 看起来像内存分配器碎片而非对象泄漏
```

**根本原因**（PR #47723 反馈，PR #44490 修复）：
```
生产者与消费者之间的条件判断（Gating）不一致

背景: PR #35219 为清理 Mamba SSM Cache State 引入
      SingleTypeKVCacheManager.new_block_ids
      → 根据 KV Cache 类型记录新 Block ID
      → 但只有模型包含 Mamba Layer 时才清理

问题: 对不含 Mamba Layer 的模型（大多数标准 Attention 模型，含 MLA 的 GLM-5.2）
      → 每次 Block 分配都持续写入该列表，但列表从不被清空
      → 随请求数量无限增长

修复: 每次调度都无条件调用 take_new_block_ids() 清空列表
      → 只有确需清理时才用返回结果
      → 保持 Mamba 行为不变，同时解决 Host Memory 持续增长
```

---

## 八、下一步工作（前瞻性判断）

### 方向一：降低 Target Forward Pass 计算成本

```
除 GEMM、Attention、Indexer 等核心 Kernel 外:
  探索 PDL（Programmatic Dependent Launch）
  Persistent Kernel、Localized Megakernel
  MoE 的 Dispatch、Expert GEMM、Combine 计算与通信协同优化

跨节点场景:
  WideEP、分层 All-to-All（Hierarchical All-to-All）
  跨节点通信与计算重叠执行

目标: 缩短 Decode Step 端到端关键路径
```

### 方向二：模型与运行时针对推测解码的协同优化

```
模型侧:
  训练能力更强的 Draft Model（如 DSpark）
  更广泛的数据集，提高连续 Proposal Token 预测准确率

Runtime 侧:
  动态推测解码（Dynamic Speculative Decoding）
  按请求动态调整 Proposal Length
  更紧凑的 Verification 机制

核心: Draft Model 决定能预测多远
     Serving Runtime 决定这种预测多大程度真正有成本效益
```

---

## 九、对 FDE 的启示（实战要点总结）

### 部署配置速查（可复现于 v0.26.0）

```bash
# Decode 实例关键配置
VLLM_USE_V2_MODEL_RUNNER=1      # MoE 模型启用 MRV2
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'  # Decode 专用 CUDA Graph
--max-num-batched-tokens 1024   # 配合 FULL_DECODE_ONLY
num_speculative_tokens=3        # Decode 侧 MTP 深度

# Prefill 实例
num_speculative_tokens=1        # Prefill 侧浅推测
```

### 生产优化核心方法论

| 原则 | 说明 |
|------|------|
| **SLA 优先** | 峰值吞吐高 ≠ 成功；满足延迟目标前提下能承载多少流量才是关键 |
| **P/D 分离是起点** | 移除 Prefill 对 Decode 关键路径的干扰，是严格 TPOT SLA 的基础 |
| **边界问题最隐蔽** | 最大性能损失常在子系统边界（P/D 交接 × 推测解码的 Mixed Batch） |
| **组合路径需专门验证** | P/D + MTP + MRV2 等新能力组合，会有 Benchmark 发现不了的长尾问题 |
| **可观测性分池监控** | TTFT/TPOT 分开监控，MTP Acceptance Rate 作为一级告警 |
| **长期稳定性测试必须** | 短时 Benchmark 无法发现内存泄漏等慢速问题 |

### 相关 PR 索引

| PR | 作用 |
|----|------|
| #45237 | Decode 侧 Speculative Padding（消除 Mixed Batch） |
| #47285 | DSA Indexer Prefill Metadata Kernel 启动预热 |
| #46448 | 多 GPU MTP 本地 Argmax Reduction |
| #45953 | 动态推测长度配合 Full CUDA Graph |
| #44420 | IndexerCache（DSA Top-K 索引复用） |
| #45895 | GLM-5.2 MTP Normalization 修复 |
| #47238 | Shared Index Buffer 布局优化 |
| #47448 | MTP 循环复用 Final Norm Hidden State |
| #47381 | MRV2 调度分类修复 |
| #46694 | P/D 异步 KV 加载 Lookahead 处理 |
| #42285 | P/D 二级缓存层 + TieringManager |
| #46570 | PCP 虚拟批处理 |
| #47723 / #44490 | Host Memory 泄漏定位与修复 |
