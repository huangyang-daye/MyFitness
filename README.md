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
| `myfitness report generate --date 2026-08-24` | 生成单日日报 |
| `myfitness report generate --start 2026-08-20 --end 2026-08-27` | 生成区间周期报表（含趋势图） |
| `myfitness chart show --metric weight --days 7` | 输出体重折线图（Mermaid） |
| `myfitness chart show --metric calories --days 30 --type bar --save` | 生成热量柱状图并保存为文档 |
| `myfitness chart insert reports/2026-08-24.md --metric weight --days 7` | 把折线图插入已有日报 |
| `myfitness llm config` | 查看 LLM 配置 |
| `myfitness llm test` | 测试 LLM 连通性 |
| `myfitness chat --session <UUID>` | 按 UUID 恢复一段历史对话 |
| `myfitness ui` | 启动三栏式本地 Agent 可视化界面 |

## 对话历史与可视化界面

每次对话使用 UUID v4 作为唯一索引，并自动保存为一个独立 JSON 文档：

```text
<DATA_DIR>/chat-history/<session-uuid>.json
```

文档包含版本、标题、创建/更新时间及完整 LangGraph 对话状态。写入采用同目录临时文件替换，避免进程中断留下半份 JSON。该目录位于项目之外（见「目录约定」），健康对话内容不会进入版本库。

启动可视化界面：

```powershell
myfitness ui
# 不自动打开浏览器，或指定端口
myfitness ui --no-open --port 8765
```

若在 UI 中同步训记时出现 `WinError 10013`，说明启动 UI 的进程被 Windows 防火墙或 Codex 受限运行环境禁止访问外网。请停止该进程，打开项目目录外部的普通 PowerShell，再运行 `myfitness ui`；若仍出现同一错误，请在 Windows 防火墙中允许项目虚拟环境的 `python.exe` 出站访问 HTTPS（443）。

界面默认只监听 `127.0.0.1`，包含左侧历史对话、中间 Agent 对话和右侧项目文件/全文搜索。文件侧栏是只读的，并屏蔽 `.env`、`.git`、`.chatHistory`、`skills` 和虚拟环境等敏感或大型目录。

CLI 也使用同一份历史仓库。启动新对话时会打印 UUID，之后可恢复：

```powershell
myfitness chat --session 550e8400-e29b-41d4-a716-446655440000
```

## 周期报表与统计图

### 周期报表（日报的区间泛化）

- 只给一天（或只给 `--date`）→ 退化为原来的**日报**，格式与文件名（`YYYY-MM-DD.md`）不变；
- 给出区间（`--start` + `--end`，或对话里说「8月20日到8月25日的报告」）→ 生成**周期报表**：
  - 文件名 `YYYY-MM-DD_YYYY-MM-DD.md`；
  - 在「分析摘要」后增加 **身体数据趋势** 章节，用 Mermaid 折线图绘制每个身体指标（至少 2 个数据点才出图，优先体重 / 体脂）；
  - 增加 **每日明细** 表（体重 / 体脂 / 热量 / 蛋白 / 训练次数）与 **区间汇总**（首末对比、日均摄入、训练总量）。

### 统计图 Tool（`agents/tools/chart_tools.py`）

用 Mermaid `xychart-beta` 渲染，横轴日期、纵轴数值：

```mermaid
xychart-beta
    title "体重趋势（2026-08-22 ~ 2026-08-28）"
    x-axis ["08-22", "08-23", "08-24"]
    y-axis "体重 (kg)" 70.02 --> 71.58
    line [71.4, 71.2, 71]
```

对话中的三种输出方式：

| 说法 | 行为 |
|------|------|
| 「生成前 7 天的体重折线图」 | 对话内联返回图表 + 数据表 |
| 「…保存成文档」 | 生成独立 Markdown 文档到 `CHART_OUTPUT_DIR` |
| 「…插入到 8 月 24 日的日报」 | 插入已有文档（可用 `--anchor` / 「## 某小节下面」指定位置） |

支持指标：`weight / bodyfat / 各围度`（body）、`calories / protein_g / carbs_g / fat_g`（nutrition）、
`volume_kg / sets / sessions / duration_min / calories`（training）。

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
├── paths.py           # 路径常量：PROJECT_ROOT / SKILLS_DIR
├── config.py          # 配置（含 DATA_DIR 及派生目录）
├── llm/               # LLM 通用 API（base_url + model）
├── xunji/             # 训记 Skill 客户端
├── db/                # 模型与 Repository
├── sync/              # 训记同步
├── agents/tools/      # 查询 / 写入 / 统计图（chart_tools.py，Mermaid）
├── services/          # 周期报表 period_report.py（单日退化为日报）、日报入口
└── api/cli.py         # CLI 入口
```

## 目录约定：项目本体 vs 使用记录

项目目录只存放**本体**（源码、测试、文档、迁移），可以安全地提交、克隆、甚至删掉重装；
运行时产生的**使用记录**一律写在项目之外的 `DATA_DIR`：

```text
D:\MyFitness\              ← 项目本体（进 Git）
├── src/ tests/ docs/      源码、测试、文档
├── migrations/            数据库迁移
├── scripts/               一次性脚本
└── skills/                Skill 定义（本地，不随 Git）

D:\MyFitness-data\         ← 使用记录（不进 Git，可单独备份或清理）
├── reports/               日报与周期报表（YYYY-MM-DD.md）
│   └── charts/            统计图（Mermaid）独立文档
├── chat-history/          对话记录（<session-uuid>.json）
└── logs/                  日志
```

`DATA_DIR` 在 `.env` 中配置；留空时回落到平台默认目录：

| 平台 | 默认 `DATA_DIR` |
| --- | --- |
| Windows | `%LOCALAPPDATA%\MyFitness` |
| macOS | `~/Library/Application Support/MyFitness` |
| Linux | `~/.local/share/MyFitness` |

需要分盘或分目录存放时，可在 `.env` 中单独覆盖 `DAILY_REPORT_OUTPUT_DIR`、
`CHART_OUTPUT_DIR`、`CHAT_HISTORY_DIR`；留空即回落到上表对应子目录。

代码侧约定（新增能力时请遵守）：

- 项目内的静态路径常量统一放在 `src/myfitness/paths.py`，不要各处重复 `Path(__file__).parents[...]`。
- 运行期路径一律从 `Settings` 读取（`data_dir` 及其派生字段），**不要**拼接项目目录下的 `reports/`、`.chatHistory/`。
- 测试需要隔离目录时，用 `tmp_path` 或环境变量覆盖，不要写入项目目录。

## 开发阶段

- **M1**（当前）：数据库、训记同步、CLI
- **M2**：LangGraph Agent
- **M3**：定时日报
- **M4**：多轮对话
- **M5**：测试与打磨
