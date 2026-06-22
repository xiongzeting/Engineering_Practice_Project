# Engineering Practice Project · 工程实践

中国科学技术大学软件学院 · 工程实践项目仓库。
项目主题：**手写作业 OCR 识别与分步批改系统**（step-based scoring system for mathematics）。

## 仓库结构

| 目录 / 文件 | 内容 |
| --- | --- |
| [`Step-based scoring system for mathematics/`](./Step-based%20scoring%20system%20for%20mathematics) | **项目源码** —— FastAPI 后端 + 原生前端静态页 + 评测脚本，运行说明见该目录的 `README.md` |
| [`datasets/`](./datasets) | **评测数据集** —— 9 个数据集的物理存放位置（v1–v6 共 9 个子集），说明见该目录的 `README.md` |
| [`代码说明文档.md`](./代码说明文档.md) | **代码说明** —— 后端模块 / 前端模块 / 评测脚本的功能与设计说明 |
| [`用户手册.md`](./用户手册.md) | **用户手册** —— 学生 / 教师 / 管理员三角色的使用流程与截图说明 |
| [`工程实践开题/`](./工程实践开题) | 开题阶段材料 |
| [`工程实践中期/`](./工程实践中期) | 中期检查材料 |
| [`工程实践结题/`](./工程实践结题) | 结题材料 —— 结题 PPT、半月进度跟踪表、系统演示视频 |

## 快速开始（运行源码）

```bash
cd "Step-based scoring system for mathematics"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # 填入 LLM_API_KEY 等配置
uvicorn app.main:app --reload
# 浏览器打开 http://127.0.0.1:8000
```

详细架构、模块说明、运行参数请看 [源码目录的 README](./Step-based%20scoring%20system%20for%20mathematics/README.md)。

## 阶段产出索引

- **开题**：选题报告、可行性分析 —— 见 [`工程实践开题/`](./工程实践开题)
- **中期**：阶段性进展、中期答辩 PPT —— 见 [`工程实践中期/`](./工程实践中期)
- **结题**：结题答辩 PPT、半月进度跟踪表、系统演示视频 —— 见 [`工程实践结题/`](./工程实践结题)
