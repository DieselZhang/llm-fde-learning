# 贡献指南 / Contribution Guide

> 欢迎每一位贡献者！无论是修正一个错别字，还是贡献一套完整的部署脚本，都对你的成长和这个仓库有价值。
>
> Welcome every contributor! Whether it's fixing a typo or contributing a complete deployment script, it's valuable both to your growth and this repo.

## 目录 / Table of Contents

- [如何贡献 / How to Contribute](#如何贡献--how-to-contribute)
- [Issue 规范 / Issue Guidelines](#issue-规范--issue-guidelines)
- [PR 规范 / PR Guidelines](#pr-规范--pr-guidelines)
- [目录约定 / Directory Conventions](#目录约定--directory-conventions)
- [代码风格 / Code Style](#代码风格--code-style)

## 如何贡献 / How to Contribute

### 我发现了错误 / I found an error

1. 先搜索 [Issues](https://github.com/DieselZhang/llm-fde-learning/issues) 是否已被报告
2. 如果没有，[提交一个 Issue](https://github.com/DieselZhang/llm-fde-learning/issues/new)，说明：
   - 文件路径和行号（或段落位置）
   - 错误内容与建议修正
   - 如果可能，附上官方文档链接作为依据

### 我想补充内容 / I want to add content

1. Fork 本仓库
2. 创建分支：`git checkout -b fix/xxx` 或 `git checkout -b feat/xxx`
3. 按照[目录约定](#目录约定--directory-conventions)放置内容
4. 提交 PR，描述清楚你做了什么、为什么

### 我想报告部署问题 / I want to report a deployment issue

在 Issue 中附上以下信息（缺一不可，否则无法复现）：

```markdown
- 环境: OS / GPU型号 / GPU数量 / 驱动版本 / CUDA版本
- 框架: vLLM 或 SGLang，版本号
- 命令: 你执行的完整启动命令
- 日志: 关键报错日志片段
- 期望行为: 你希望它怎么做
- 实际行为: 它实际做了什么
```

## Issue 规范 / Issue Guidelines

- 使用清晰的标题，如 `[vLLM] TP=2 启动时 NCCL 超时`
- 中文或英文均可，但请保持一致
- 如果是 Bug，务必附上可复现的环境信息（见上文模板）
- 如果是功能建议，请说明使用场景和期望效果

## PR 规范 / PR Guidelines

- 标题简洁，如 `docs: 修正 KV Cache 公式` 或 `feat: 新增 SGLang PD 分离脚本`
- 描述中说明：改了什么、为什么改、如何验证
- 一次 PR 只做一件事（一个 fix 或一个 feature）
- 修改内容尽量保持与现有文件的中英双语风格一致（如适用）
- 脚本类改动请在本地运行验证过再提交

## 目录约定 / Directory Conventions

请按内容类型放置到对应目录：

```
foundations/    # 理论笔记、深度解析、部署指南、交互式页面
  ├── *.md                      # 深度笔记（如 kv-cache-deep-dive.md）
  ├── *-interactive-learning.html  # 交互式学习页面
  └── *-deployment-guide.md     # 逐行部署指南

practices/docs/    # 按学习天数编号
  ├── 0N-主题.md               # 例如 08-multimodal-deployment.md

practices/scripts/ # 可运行的部署/监控脚本
  ├── NN-用途.sh              # 例如 10-vllm-multimodal.sh
  └── *.py                    # Python 脚本

practices/references/  # 速查手册、FAQ
reports/             # 性能报告模板和示例
```

### 编号规则

- 脚本编号：在当前最大编号基础上 +1（如当前到 `09-`，新脚本用 `10-`）
- 学习文档：在 `practices/docs/` 当前最大天数基础上 +1

## 代码风格 / Code Style

### Shell 脚本

- 以 `#!/bin/bash` 开头
- 文件头加注释块，说明脚本用途和用法
- 使用 `set -e` 确保错误即退出
- 变量使用大写（如 `MODEL`、`TP_SIZE`）
- 关键步骤输出 `echo` 提示

```bash
#!/bin/bash
# ============================================================
# Script: xxx
# 用途：xxx
# ============================================================
set -e

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
```

### Markdown 文档

- 中文为主，专业术语保留英文原词（如 PagedAttention、KV Cache）
- 代码块标注语言（如 ```` ```bash ````）
- 表格用于对比类内容
- 相对链接使用相对路径（不要用绝对路径）

---

> **最终提醒**: 你的每一次贡献都在帮助下一位从传统运维转型 FDE 的工程师。请认真对待，也感谢你的付出。
>
> **Final note**: Every contribution helps the next engineer moving from traditional ops to FDE. Please take it seriously, and thank you for your effort.
