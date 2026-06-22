# 手写作业 OCR 识别与基础批改系统

基于 **FastAPI + Paddle/Pix2Text OCR + 前端静态页面** 的可运行作业识别与分步评分系统。  
当前版本已集成：**账号登录、用户区分、数据库持久化、OCR与评分历史**。

## 项目结构

```text
Step-based scoring system for mathematics/
├── app/
│   ├── main.py                 # API 入口（认证、OCR、评分、历史）
│   ├── config.py               # 配置项
│   ├── schemas.py              # Pydantic 数据结构
│   └── services/
│       ├── auth.py             # 令牌与密码校验
│       ├── db.py               # SQLite 持久化
│       ├── ocr_service.py      # OCR 主服务
│       ├── ocr_postprocess.py  # 分块排序、分题、清洗
│       ├── step_parser.py      # 步骤切分
│       ├── scorer.py           # 规则评分 + LLM评分
│       └── vision_ocr.py       # 纯多模态OCR
├── static/
│   ├── index.html              # 前端页面（登录、识别、评分）
│   ├── app.js                  # 前端逻辑
│   └── styles.css              # 样式
├── outputs/
│   └── app.db                  # SQLite 数据库（运行后自动创建）
├── requirements.txt
└── .env.example
```

## 快速启动

### 1) 安装并启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

启动后访问：`http://127.0.0.1:8001`

### 2) 首次使用流程

1. 在页面顶部“账户登录”区域先注册/登录  
2. 上传图片执行 OCR（支持按题分页查看）  
3. 选择“只评分当前分页题目”执行评分  
4. 查看步骤得分、反馈、评分诊断  

## 技术栈

| 层次 | 技术 |
|---|---|
| 前端 | HTML + CSS + 原生 JavaScript + KaTeX |
| 后端 | FastAPI + Uvicorn |
| OCR | Pix2Text / Paddle 兼容流程 + 多模态OCR |
| 评分 | 规则评分 + OpenAI 兼容 LLM 评分 |
| 认证 | Token（HMAC签名） |
| 数据库 | SQLite（自动初始化） |

## 主要功能模块

### 认证模块

- `POST /api/auth/register`：注册（student/teacher）
- `POST /api/auth/login`：登录（返回 Bearer Token）
- `GET /api/auth/me`：获取当前登录用户

### OCR 模块

- `POST /api/ocr`：图片OCR + 分题 + 步骤切分（需登录）
- `POST /api/ocr-vision-only`：纯多模态OCR（需登录）

### 评分模块

- `POST /api/grade`：步骤评分（需登录）
  - 规则评分兜底
  - LLM 评分优先（失败自动回退）
  - 返回 `grading_meta` 诊断信息

### 历史模块

- `GET /api/history/ocr`：当前用户 OCR 历史
- `GET /api/history/grading`：当前用户评分历史

## 批改引擎说明

| 模式 | 说明 |
|---|---|
| 规则评分 | 无模型依赖，快速稳定兜底 |
| LLM评分 | 评分更细致，支持“正确步骤满分”“连锁错误不重复重罚” |
| 回退策略 | LLM异常/超时时自动回退规则评分 |

## 数据库说明

- 默认数据库：`outputs/app.db`
- 自动初始化表：
  - `users`
  - `ocr_records`
  - `grading_records`

## 配置说明（.env）

复制 `.env.example` 为 `.env` 后按需修改：

```env
LLM_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

LLM_SCORE_TIMEOUT_SEC=90
LLM_SCORE_CONNECT_TIMEOUT_SEC=15
LLM_SCORE_NO_READ_TIMEOUT=false
LLM_SCORE_USE_STREAM=true
LLM_SCORE_PREFER_CHAT=true

AUTH_SECRET=change-this-in-production
AUTH_EXP_MINUTES=1440
DB_PATH=outputs/app.db
MAX_UPLOAD_MB=10
```

## 常见问题

### Q1：为什么评分显示“规则评分(回退)”？
说明 LLM 调用失败或超时。可在评分结果里查看 `评分诊断`，定位具体失败点（状态码、耗时、错误）。

### Q2：是否必须接入数据库和登录？
建议必须接入。  
原因：中期检查通常要求“用户区分、数据留痕、可追溯”，本项目已支持。

### Q3：OCR 结果太长导致评分慢怎么办？
启用“只评分当前分页题目”，按题逐个评分，显著降低超时概率。

## 开发建议

- 生产环境请更换 `AUTH_SECRET`
- 建议将 SQLite 升级为 MySQL/PostgreSQL（中后期）
- 增加角色权限（教师查看班级统计，学生仅看个人记录）
