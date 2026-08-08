# Qwen3.5 PD 分离：混合注意力架构 + GB200 NVL72 单卡 25K TPS

> **来源**：微信公众号《GB200 NVL72 上的 Qwen3.5 PD 分离：单卡突破 25K total TPS》（vLLM 官方博客）
> **原文**：https://mp.weixin.qq.com/s/K7uhS_uHQHD8h-TbH-G_7A
> **官方博客**：https://vllm.ai/blog/2026-08-06-qwen35-25k-tps
> **笔记类型**：混合注意力模型（SSM + Attention）的 PD 分离生产实践

---

## 一、核心背景

```
Qwen3.5（2026 年初发布，客户侧最广泛使用的模型之一）
  → 混合注意力架构：full attention 层 + Gated Delta Network（GDN）层
  → PD 分离部署带来额外挑战：prefill 要往 decode 传两种语义完全不同的状态

本文内容:
  ① Blackwell GPU 上加速 GDN 计算
  ② prefill/decode 间正确传输异构的 attention / GDN 状态
  ③ 异步调度的竞态修复
  ④ GB200 NVL72 上单卡 25K total TPS 的实测数据与可复现 recipe
```

### 核心模型：Qwen3.5-397B-A17B-NVFP4

```
混合架构（hybrid SSM-attention）:
  full attention 层（传统 KV Cache）
  GDN 层（Gated Delta Network，类似 Mamba 的 SSM 状态）

对比之前的 GLM-5.2:
  GLM-5.2: 纯 MoE + DSA 稀疏注意力（上篇笔记）
  Qwen3.5: 混合注意力 = 传统 attention 层 + SSM（状态空间）层
```

---

## 二、关键优化一：面向 Blackwell 的 GDN prefill kernel

### GDN 是什么

**Gated Delta Network** 是类似 Mamba 的**状态空间模型（SSM）**层，它不用 KV Cache，而是维护一个**不断更新的隐藏状态（SSM state）**。GDN 层与 full attention 层交替出现，构成 Qwen3.5 的混合架构。

```
Full attention 层: 传统方式，KV Cache（你已熟悉）
GDN 层 (SSM): 维护一个固定大小的"状态"，每步更新，无需 KV Cache
  → 状态小、内存友好，但 prefill 计算复杂度特殊
```

### 优化内容

**FlashInfer #3001**：新增 Blackwell GDN prefill kernel

```
性能: 相比 FLA/Triton 实现，提升 1.02x 到 5.78x
     （不同模型尺寸 / TP 配置 / 序列长度 / batch 形状）
```

**vLLM PR #40717**：把 kernel 接到 prefill 侧

```
在 8xB200 上跑 Qwen3.5-397B-A17B-NVFP4:
  GDN kernel 微基准最高提升 5.92x
  prefill-only 负载 (8K/1) 端到端吞吐提升 1.13x
  同负载下 mean TTFT 降低 12%
```

**配置**：
```bash
# Blackwell 上 GDN backend=auto 时自动选择 FlashInfer
# 也可显式指定:
--gdn-prefill-backend flashinfer
```

---

## 三、关键优化二：混合 cache 与 GDN 状态传输

### 问题：两种不同的状态要同时传输

```
Qwen3.5 的 PD 分离需要 prefill → decode 传两种东西:
  ① full attention 层的 KV Cache（你已熟悉的）
  ② GDN/SSM 层的状态（Mamba 式，布局/大小/传输语义都不同）
```

### 三个关键 PR 的演进

**① HMA + NIXL Connector（#35758）**：必要前提
```
把 HMA（Hybrid Memory Allocation）的逻辑 block 映射到正确的物理内存区域
→ NIXL 只需传输属于每种层类型的那部分 cache 区域
→ 传输 descriptor 数量从 4,284 降到 1,650
→ 节点内 H100 环境吞吐提升最高约 7%
```

**② 混合 SSM-FA 分离主 PR（#36687）**：核心
```
引入 dual descriptor view + 同构 TP 支持
→ prefill/decode 能通过 NIXL 同时传输 KV Cache 和 Mamba 式 SSM 状态
```

**③ Qwen3.5 GDN 支持（#41869）**：扩展到 GDN 层

**配套 PR**：
- #37416: Mamba Conv state 不同 layout 支持
- #37635: Heterogeneous TP 下的 conv state 传输
- #37310: N-1 prefill（P/D 分离）

---

## 四、关键优化三：无竞态的异步调度

### 问题

```
两个竞态让异步调度完全不可用: 一开启精度直接归零
异步调度后来被证明是跨过 25K tok/s/GPU 的关键特性
```

### 两个竞态修复

```
#48481: 混合 attention 模型的 PD async scheduling 竞态
#45357: 延迟 block 释放（等 in-flight steps 完成再释放）
```

---

## 五、性能实测

### 环境配置

```
硬件: GB200 集群，NVLink72 互联
负载: ISL/OSL = 8192/1024
模型: Qwen3.5-397B-A17B-NVFP4

拓扑:
  decode 侧: 1 个 endpoint, DEP8（8卡 DP + EP）
  prefill 侧: 4-8 个 endpoint, 每个 DEP2
  并发: 从 64 扫到 5120
```

### 精度验证

```
GSM8K 基准: 5 个配置全部 88%，与聚合部署一致
→ 确认性能结果有效（PD 分离不损失精度）
```

### 性能结果

```
单卡 total TPS 达到 25,000 tokens/秒
并发上限 5120: decode 侧 KV cache 容量开始不够
  → 继续推并发需要在 decode 侧增加 GPU
```

---

## 六、Recipe 与最佳实践

### 关键配置项

| 配置 | 作用 |
|------|------|
| `VLLM_SSM_CONV_STATE_LAYOUT=DS` | **必须**：SSM 模型 PD 分离时 conv state 才能传输 |
| `--async-scheduling` | **跨过 25K 的关键**：需包含两个竞态修复 |
| `--mamba-ssm-cache-dtype bfloat16` | 显著提升 decode 侧 KV cache 容量 |
| `--language-model-only` | Qwen3.5 是多模态，纯文本时关多模态 + 打开 QK-norm+RoPE+gate 融合路径 |
| `--max-num-batched-tokens 16384` | = 2xISL，prefill 每步 batch 两条完整 prompt（+8% TPS） |
| `--max-cudagraph-capture-size` | 高并发时提到 cc/8+128（cc=4096->640, cc=5120->768） |
| `--stream-interval 100` | 降低高并发前端开销（注意影响 per-token 延迟） |
| prefix caching 关闭 | random 数据集无收益 |

### 排查经验

```
--api-server-count 1 很有用:
  DP endpoint 默认 API server 数量 = DP 大小
  多 API server 时关闭默认 stats 日志
  强制 1 → 找回每 10 秒的 stats 日志（VLLM_LOG_STATS_INTERVAL 可调）
  → 打印 prompt/gen 吞吐 + KV cache 利用率
  → 没有这些指标很难定位瓶颈

三个环境变量:
  DYN_LOG=error                   # 削减 Dynamo 日志量
  DYN_SDK_DISABLE_ANSI_LOGGING=1  # 抑制 ANSI 转义
  VLLM_LOGGING_COLOR=0            # 关颜色
```

---

## 七、下一步方向

```
当前: 聚焦 Pareto 左半部分（最大化单卡 total TPS）

未来: 转向"Gen TPS per user"最大的 PD 配置
  → 需要从 DEP 拓扑转向 TEP（TP + EP）或纯 TP
  → 这类拓扑在单用户性能上表现更好
  → 增加 GPU 数量是另一预期有回报的手段
```

---

## 对 FDE 的启示

### 混合注意力模型（SSM+Attention）与纯 Attention 的区别

```
纯 Attention 模型（GLM-5.2/Llama）:
  PD 分离只需传 KV Cache
  用 NIXL 传输 KV blocks

混合模型（Qwen3.5）:
  PD 分离要传 KV Cache + SSM 状态（两种语义不同的东西）
  → 需要 dual descriptor view、异构 TP、conv state 传输
  → 传输逻辑复杂得多
```

### 部署混合模型的检查清单

```
① GDN backend: --gdn-prefill-backend flashinfer（Blackwell）
② VLLM_SSM_CONV_STATE_LAYOUT=DS（必须！否则状态传不过去）
③ --async-scheduling（需确认包含竞态修复版本）
④ --mamba-ssm-cache-dtype bfloat16（提升 decode 容量）
⑤ 多模态模型用 --language-model-only 关掉多模态（纯文本时）
```

### 关键词汇速查

| 概念 | 一句话 |
|------|--------|
| GDN | Gated Delta Network，类似 Mamba 的 SSM 状态层 |
| HMA | Hybrid Memory Allocation，混合注意力模型的 cache 管理 |
| SSM State | 状态空间模型的隐藏状态（区别于 KV Cache） |
| dual descriptor view | 同时传输 KV Cache 和 SSM 状态两种描述符 |
| DEP | Data Parallel + Expert Parallel 组合拓扑 |
| TEP | Tensor Parallel + Expert Parallel（单用户性能更好） |
| async scheduling | 异步调度，跨过高并发的关键特性 |
| NIXL Connector | 跨节点 KV Cache / SSM 状态传输通道 |

---

## 参考链接

- [vLLM 官方博客原文](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)
- [NIXL 分离路线图 #33702](https://github.com/vllm-project/vllm/issues/33702)
- [hybrid SSM 分离博客](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg)
- [srt-slurm-recipes](https://github.com/NVIDIA/srt-slurm-recipes)
- [Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)
