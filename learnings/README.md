# learnings/ — 源码项目学习产出

> 使用 repo-mastery skill 学习开源项目源码后的产出目录。
> 每个源码项目一个子目录，存放完整的课程文档（Markdown + HTML 双格式）。

## 组织约定

```
learnings/
├── vllm/                 # 每个源码项目一个子目录
│   ├── course.md         # Markdown 版课程（可同步到 git）
│   ├── course.html       # HTML 版课程入口（跳转到 .learning/export/）
│   └── .learning/        # repo-mastery 完整学习状态（章节、进度、复习队列、HTML 站点）
├── sglang/
├── flash-attention/
└── README.md             # 本说明
```

> 说明：`course.md` 是自包含的 Markdown 主课程；完整学习状态（各模块教材 `chapters/`、进度 `progress.json`、掌握度 `MASTERY.md`、生态定位 `positioning.md`、HTML 课程站点 `export/`）统一放在该项目的 `.learning/` 子目录下。

## 如何开始学习一个新项目

```bash
/repo-mastery start <github:owner/repo>
# 或 /repo-mastery start <本地路径>
```

## 目录清单

| 项目                             | 状态    | 说明                                          |
| ------------------------------ | ----- | ------------------------------------------- |
| [pi-agent](pi-agent/course.md) | ✅ 已整理 | Pi Agent 源码精读笔记（10 章，Agent SDK 核心机制）        |
| [mem0](mem0/course.md)         | ✅ 已整理 | mem0 记忆层源码课程（7 模块 / 23 知识点，Markdown + HTML） |
