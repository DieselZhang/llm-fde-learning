"""
Day 1 互动学习：完整推理管线解剖
==============================
逐步演示 模型加载 → Token化 → Prefill → Decode → 采样 → 输出
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# ============================================================
# 阶段 0：环境准备
# ============================================================
print("=" * 70)
print("🧠 LLM 推理管线 — 全局解剖")
print("=" * 70)

device = "cpu"
dtype = torch.float32
print(f"\n运行设备: {device}  |  精度: {dtype}")

# ============================================================
# 阶段 1：加载模型（从磁盘 → 内存）
# ============================================================
print("\n" + "=" * 70)
print("📦 阶段 1：模型加载 — 从 HuggingFace 下载权重到内存")
print("=" * 70)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 6亿参数，CPU可跑

print(f"\n  模型: {model_name}")
print(f"  参数量: 0.5B (约 6 亿参数)")
print(f"  显存估算: 6亿 × 4字节(float32) ≈ 2.4GB (CPU OK)")

t0 = time.time()
print("\n  正在从 HuggingFace Hub 加载 Tokenizer...", end=" ", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✅ {time.time()-t0:.1f}s")

t0 = time.time()
print("  正在加载模型权重（首次会下载，后续用缓存）...", end=" ", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    attn_implementation="eager",
)
model.to(device)
model.eval()  # 推理模式
print(f"✅ {time.time()-t0:.1f}s")

# 打印模型结构摘要
num_layers = len(model.model.layers)
hidden_size = model.config.hidden_size
num_heads = model.config.num_attention_heads
head_dim = hidden_size // num_heads
print(f"\n  模型结构:")
print(f"    Transformer 层数:    {num_layers}")
print(f"    隐藏维度 (d_model):   {hidden_size}")
print(f"    Attention Heads:     {num_heads}")
print(f"    Head 维度:           {head_dim}")
print(f"    词汇表大小:          {model.config.vocab_size}")
print(f"    最大序列长度:        {model.config.max_position_embeddings}")

# ============================================================
# 阶段 2：Tokenization（文本 → Token IDs）
# ============================================================
print("\n" + "=" * 70)
print("🔤 阶段 2：Tokenization — 自然语言 → Token ID 序列")
print("=" * 70)

prompt = "请简要介绍一下 KV Cache 是什么："
print(f"\n  输入文本: 「{prompt}」")

t0 = time.time()
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
print(f"  分词耗时: {(time.time()-t0)*1000:.1f}ms")

print(f"\n  Token IDs: {input_ids[0].tolist()}")
print(f"  序列长度:  {input_ids.shape[1]} tokens")

# 逐个打印 token 和对应的文本
print(f"\n  Token 逐项解码:")
for i, tid in enumerate(input_ids[0]):
    token_text = tokenizer.decode(tid)
    print(f"    [{i:3d}] ID={tid:6d} → 「{token_text}」")

# ============================================================
# 阶段 3：词嵌入（Token IDs → 向量）
# ============================================================
print("\n" + "=" * 70)
print("📐 阶段 3：Embedding — Token ID 映射为稠密向量")
print("=" * 70)

embedding_layer = model.model.embed_tokens
with torch.no_grad():
    hidden_states = embedding_layer(input_ids)

print(f"\n  输入形状:   {tuple(input_ids.shape)}  (batch_size, seq_len)")
print(f"  嵌入形状:   {tuple(hidden_states.shape)}  (batch_size, seq_len, hidden_size)")
print(f"  每个 token → {hidden_size} 维向量")
print(f"  向量示例 (token 0 前 10 维): {hidden_states[0, 0, :10].tolist()}")
print(f"  范数 (token 0): {torch.norm(hidden_states[0, 0]):.4f}")

# ============================================================
# 阶段 4：Prefill — 并行计算所有输入 token
# ============================================================
print("\n" + "=" * 70)
print("⚡ 阶段 4：Prefill — 一次性处理整个输入序列")
print("=" * 70)

print("  Prefill 特征: 计算密集型，GPU 利用率高")
print("  一次前向传播计算出所有输入 token 的 hidden states + KV Cache")

with torch.no_grad():
    t0 = time.time()
    outputs = model(
        input_ids,
        use_cache=True,        # 打开 KV Cache
        output_attentions=True, # 输出 attention 权重
    )
    prefell_time = time.time() - t0

logits = outputs.logits
past_kv = outputs.past_key_values  # KV Cache

print(f"\n  Prefill 耗时: {prefell_time*1000:.1f}ms")
print(f"  Logits 形状: {tuple(logits.shape)}  (batch, seq_len, vocab_size)")
print(f"  KV Cache 类型: {type(past_kv).__name__}")

# 提取 KV Cache 内容（兼容 DynamicCache 和 tuple 格式）
kv_layers = past_kv
if hasattr(kv_layers, "layers"):
    # DynamicCache 格式 (transformers 4.48+)
    kv_pairs = [(layer.keys, layer.values) for layer in kv_layers.layers if hasattr(layer, 'keys')]
elif isinstance(kv_layers, tuple):
    # 传统 tuple of tuples
    kv_pairs = [(kv[0], kv[1]) for kv in kv_layers]
else:
    kv_pairs = list(kv_layers)

print(f"  KV Cache 层数: {len(kv_pairs)}")
print(f"  KV Cache 每层形状:")

for i, (k, v) in enumerate(kv_pairs):
    print(f"    层 {i:2d}: K={tuple(k.shape)}  V={tuple(v.shape)}")
    if i >= 2:  # 只展示前3层
        print(f"    ... (共 {len(kv_pairs)} 层，形状相同)")
        break

# 计算单层 KV Cache 大小
k_layer_0 = kv_pairs[0][0]
cache_per_token_bytes = k_layer_0.shape[2] * k_layer_0.shape[3] * 2 * k_layer_0.element_size()
total_cache_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() 
                        for k, v in kv_pairs)
print(f"\n  单 token 单层 KV Cache: {cache_per_token_bytes / 1024:.1f} KB")
print(f"  总 KV Cache 大小: {total_cache_bytes / 1024 / 1024:.2f} MB")

# ============================================================
# 阶段 5：假设只有 Prefill（无 KV Cache）—— 对比用
# ============================================================
print("\n" + "=" * 70)
print(" 🔄 对比实验：不用 KV Cache 重复计算")
print("=" * 70)

# 模拟：生成 10 个 token 时，如果不用 KV Cache 每步要重新算全部
with torch.no_grad():
    # 第一步：完整输入都算
    t0 = time.time()
    _ = model(input_ids, use_cache=False)
    full_time = time.time() - t0
    
    # 第二步：输入 + 1 个新 token，全部重算
    dummy_next = torch.tensor([[42]], dtype=torch.long)  # 假的下一个token
    longer_input = torch.cat([input_ids, dummy_next], dim=1)
    t0 = time.time()
    _ = model(longer_input, use_cache=False)
    step_time = time.time() - t0

print(f"  完整序列 ({input_ids.shape[1]} tokens): {full_time*1000:.1f}ms")
print(f"  加 1 token 后重算全部 ({longer_input.shape[1]} tokens): {step_time*1000:.1f}ms")
print(f"  → 每多 1 token 序列变长，计算量 O(n²) 增长！")
print(f"  → 这就是 KV Cache 存在的意义：把 O(n²) 降为 O(n)")

# ============================================================
# 阶段 6：Decode — 逐 token 生成
# ============================================================
print("\n" + "=" * 70)
print("🚶 阶段 6：Decode（生成） — 逐 token 生成 + 缓存")
print("=" * 70)

print("  Decode 特征: 内存密集型，瓶颈在 KV Cache 读取")
print()

num_generate = 20  # 生成 20 个 token
generated_ids = input_ids.clone()
all_tokens = [prompt]
next_token_logits_list = []
times = []

with torch.no_grad():
    for step in range(num_generate):
        t0 = time.time()
        
        # 关键区别：这里只输入最后一个 token
        # 但模型从 past_key_values 里读取之前所有位置的 KV Cache
        next_input = generated_ids[:, -1:]  # shape: (1, 1)
        
        outputs = model(
            next_input,
            use_cache=True,
            past_key_values=past_kv,  # 传入之前累积的 KV Cache
        )
        
        elapsed = time.time() - t0
        times.append(elapsed)
        
        logits = outputs.logits[:, -1, :]  # 最后一个位置的 logits
        past_kv = outputs.past_key_values   # 更新 KV Cache
        
        # 采样：取概率最高的 token
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        
        token_text = tokenizer.decode(next_token_id[0])
        all_tokens.append(token_text)
        next_token_logits_list.append(logits)
        
        print(f"    步骤 {step+1:2d}: 输入 1 token → 输出 [{next_token_id.item():6d}] 「{token_text}」 ({(elapsed*1000):.1f}ms)")

# ============================================================
# 阶段 7：采样分析（最后一步的 token 概率分布）
# ============================================================
print("\n" + "=" * 70)
print("🎲 阶段 7：采样 — 从概率分布中选择下一个 token")
print("=" * 70)

last_logits = next_token_logits_list[-1]
probs = F.softmax(last_logits, dim=-1)
top_probs, top_indices = torch.topk(probs, k=10, dim=-1)

print(f"\n  最后一步 Top-10 token 概率:")
for i in range(10):
    token_txt = tokenizer.decode(top_indices[0, i])
    print(f"    [{i+1:2d}] 「{token_txt}」  prob={top_probs[0, i].item()*100:.2f}%  (ID={top_indices[0, i].item()})")

print(f"\n  采样方法对比:")
print(f"    Greedy (argmax): 永远选概率最高的 → 「{tokenizer.decode(top_indices[0,0])}」")
print(f"    Top-k (k=50):    从前 50 个按概率采样 → 增加多样性")
print(f"    Temperature:      T<1 锐化分布，T>1 平滑分布")

# ============================================================
# 阶段 8：反分词 — 输出最终结果
# ============================================================
print("\n" + "=" * 70)
print("📝 阶段 8：Detokenization — Token ID → 可读文本")
print("=" * 70)

full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"\n  完整输出:\n{full_output}")

# ============================================================
# 性能总结
# ============================================================
print("\n" + "=" * 70)
print("📊 性能总结")
print("=" * 70)

avg_decode_ms = sum(times) / len(times) * 1000
total_prefill_ms = prefell_time * 1000

print(f"""
  ┌─────────────────────────────────────────────┐
  │  Prefill:         {total_prefill_ms:7.1f}ms  ({input_ids.shape[1]:3d} tokens)  │
  │  Decode (avg):    {avg_decode_ms:7.1f}ms  per token    │
  │  总生成:          {num_generate:3d} tokens                │
  │  Total latency:   {(prefell_time + sum(times))*1000:7.1f}ms           │
  │  Throughput:      {(num_generate)/(prefell_time + sum(times)):7.1f} tok/s       │
  │  TTFT:            {total_prefill_ms:7.1f}ms (Time To First Token)  │
  │  TPOT:            {avg_decode_ms:7.1f}ms (Time Per Output Token)   │
  └─────────────────────────────────────────────┘
""")

print("=" * 70)
print("✅ 推理管线全流程演示完毕")
print("=" * 70)
