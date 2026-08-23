# MyFitness

基于 LangGraph 的多 Agent 健康监控系统，支持训记 Open API 同步、PostgreSQL 持久化、日报与对话分析。

详细需求见 [docs/PRD.md](docs/PRD.md)。

## 快速开始

### 1. 环境准备

- Python 3.11+
- PostgreSQL 14+

### 2. 安装

```powershell
cd D:\MyFitness
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

> **注意**：必须在项目虚拟环境中运行。`(base)` Conda 环境里没有 `myfitness` 命令。

激活 venv 后，命令行提示符前应出现 `(.venv)`。

### 3. 配置

```powershell
copy .env.example .env
# 编辑 .env：DATABASE_URL（PostgreSQL 真实账号密码）与训记 API Keys
```

`DATABASE_URL` 示例：
```env
DATABASE_URL=postgresql+psycopg://postgres:你的密码@localhost:5432/myfitness
```

### 4. 初始化数据库

```powershell
# 方式一（推荐，需已 Activate venv）
myfitness db migrate

# 方式二（不激活 venv 也可）
.\.venv\Scripts\myfitness.exe db migrate

# 方式三（开发备用）
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python -m myfitness.api.cli db migrate
```

### 5. 同步训记数据

```bash
myfitness sync --days 7
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `myfitness db migrate` | 执行 Alembic 数据库迁移 |
| `myfitness sync --days 7` | 同步最近 7 天训记数据 |
| `myfitness sync --start 2026-08-01 --end 2026-08-21` | 同步指定日期范围 |
| `myfitness llm config` | 查看 LLM 配置 |
| `myfitness llm test` | 测试 LLM 连通性 |

## LLM 配置

使用 **OpenAI 兼容通用接口**，在 `.env` 中填写 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 即可激活：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=120
```

支持 DeepSeek、Ollama 代理等任意 OpenAI 兼容服务，只需修改 `LLM_BASE_URL` 与 `LLM_MODEL`。

```bash
myfitness llm config   # 查看配置（Key 脱敏）
myfitness llm test     # 发送测试请求
```

Agent 功能需额外安装：`pip install -e ".[agents]"`，代码中通过 `myfitness.llm.get_llm()` 获取 LangChain 客户端。

## 训记集成

训记 Open API 严格按项目内 Skill 实现，文档位于 `skills/xunji-*/SKILL.md`：

| Skill | 模块 | 解析/写确认 |
|-------|------|-------------|
| xunji-body-open-api | `xunji/body.py` | `parsers/body.py`, `write_flow.py` |
| xunji-food-open-api | `xunji/food.py` | `parsers/food.py`, `write_flow.py` |
| xunji-training-open-api | `xunji/training.py` | `parsers/training.py`, `write_flow.py` |

Agent 与同步层应调用 `myfitness.xunji`，不要直接 httpx。详见 `.cursor/rules/xunji-skills.mdc`。

## 项目结构

```
src/myfitness/
├── config.py          # 配置
├── llm/               # LLM 通用 API（base_url + model）
├── xunji/             # 训记 Skill 客户端
├── db/                # 模型与 Repository
├── sync/              # 训记同步
└── api/cli.py         # CLI 入口
```

## 开发阶段

- **M1**（当前）：数据库、训记同步、CLI
- **M2**：LangGraph Agent
- **M3**：定时日报
- **M4**：多轮对话
- **M5**：测试与打磨
