# Day 4 学习日志 — SGLang 入门

## 📅 日期：YYYY-MM-DD

## ✅ 今日完成

- [ ] SGLang 安装
- [ ] 基本 API 调用
- [ ] 前缀共享实验
- [ ] Benchmark 测试
- [ ] mini-sglang 源码阅读

## 📝 SGLang vs vLLM 对比总结

| 维度 | vLLM | SGLang |
|------|------|--------|
| KV Cache | PagedAttention | RadixAttention |
| 调度方式 | iteration-level | 零开销调度 |
| 前缀复用 | 需手动开启 | 自动 |
| 结构化输出 | 不支持原生 | 原生支持 |
| Router | 外部 | 内置 sgl-router |
| 安装复杂度 | | |

## 🧪 前缀共享实验

```
冷启动时间: ___ s
前缀复用时间: ___ s
加速比: ___ x
```

## 📖 mini-sglang 源码笔记

```
重点理解的几个模块：
1. Scheduler 做了什么事？
2. RadixAttention 怎么管理缓存？
3. Model Runner 如何调用 kernel？
```

## 🤔 遇到的问题
