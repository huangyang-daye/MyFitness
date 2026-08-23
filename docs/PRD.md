# MyFitness 多 Agent 健康监控 — 产品需求文档（PRD）

| 属性 | 值 |
|------|-----|
| 版本 | v1.0 |
| 状态 | 草案 |
| 最后更新 | 2026-08-22 |
| 数据源策略 | 训记 Open API 自动同步 + 用户手动补充/修正（混合模式） |
| 交互模式 | 定时日报 + 随时对话 |

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [范围定义](#2-范围定义)
3. [用户画像与场景](#3-用户画像与场景)
4. [用户故事](#4-用户故事)
5. [系统架构](#5-系统架构)
6. [Agent 职责与边界](#6-agent-职责与边界)
7. [LangGraph State 与 Agent 契约](#7-langgraph-state-与-agent-契约)
8. [数据模型与 DDL](#8-数据模型与-ddl)
9. [训记 Open API 集成规范](#9-训记-open-api-集成规范)
10. [交互设计](#10-交互设计)
11. [接口草案](#11-接口草案)
12. [非功能需求](#12-非功能需求)
13. [技术栈与项目结构](#13-技术栈与项目结构)
14. [里程碑与验收标准](#14-里程碑与验收标准)
15. [风险与待定项](#15-风险与待定项)
16. [附录](#16-附录)

---

## 1. 背景与目标

### 1.1 问题陈述

个人健身/减脂/增肌过程中，体重、体脂、饮食、训练等数据分散在训记 App 与日常零散记录中。用户难以：

- 跨维度关联分析（如训练量与热量摄入、体重波动与睡眠/饮食的关系）
- 基于历史趋势制定可执行的下一步训练与饮食计划
- 获得持续、结构化的每日反馈

### 1.2 产品目标

构建一个基于 **LangGraph** 的多 Agent 系统，实现：

```
采集（训记同步 + 手动录入）→ 存储（MySQL）→ 分析（ Specialist Agents）→ 规划 → 反馈（日报 + 对话）
```

### 1.3 成功指标

| 指标 | 目标 |
|------|------|
| 日报覆盖率 | 有数据的天数 ≥ 90% 能自动生成日报 |
| 数据同步延迟 | 训记同步完成后 5 分钟内入库 |
| 对话响应 | 常见查询（近 7/30 天趋势）30 秒内返回 |
| 写操作安全 | 100% 写库/写回训记前经用户确认 |
| 数据可追溯 | 每条记录可区分来源（sync / manual / agent_suggested） |

---

## 2. 范围定义

### 2.1 In Scope（一期）

- 每日追踪：体重、体脂率、饮食、训练/运动记录
- 基于历史 + 当日数据的分析与规划（训练计划、饮食计划、体重目标路径）
- MySQL 持久化全部历史数据
- LangGraph 编排 4 类 Specialist Agent + 1 个汇总分析 Agent
- 训记 Open API 自动同步 + 用户手动补充/修正
- 定时日报生成 + 随时对话式查询/调整

### 2.2 Out of Scope（一期明确不做）

- 医疗诊断、疾病治疗建议
- 移动端 App 原生开发（一期以 CLI / REST API / 轻量 Web 为主）
- 多用户 SaaS 化（一期默认单用户 / 本地部署）
- 训记以外的第三方健康平台集成（Apple Health、Garmin 等列为二期）

---

## 3. 用户画像与场景

### 3.1 目标用户

个人健身爱好者，已在训记 App 中记录部分身体、饮食、训练数据，希望获得 AI 辅助的趋势分析与可执行规划。

### 3.2 核心场景

| # | 场景 | 描述 |
|---|------|------|
| S1 | 晨间同步 | 系统定时从训记拉取昨日数据，写入 MySQL，生成「昨日回顾 + 今日建议」日报 |
| S2 | 手动补录 | 用户通过对话补充训记未记录的一餐、临时体重测量 |
| S3 | 计划调整 | 用户说「今天练腿改成休息」，健身规划师 Agent 更新当日/本周计划 |
| S4 | 趋势追问 | 「近 30 天体脂变化如何？和饮食热量有关系吗？」→ 汇总 Agent 跨域检索并回答 |
| S5 | 目标设定 | 用户设定目标体重/体脂，体重监测 Agent 给出阶段性里程碑 |

---

## 4. 用户故事

### 4.1 数据同步

| ID | 用户故事 | 优先级 |
|----|----------|--------|
| US-01 | 作为用户，我希望系统每天自动从训记同步体重、体脂、饮食、训练数据，以便无需手动导出 | P0 |
| US-02 | 作为用户，我希望手动触发同步指定日期范围的数据，以便补全历史或纠正延迟 | P1 |
| US-03 | 作为用户，我希望同步失败时在日报中看到明确提示，以便知道哪些数据缺失 | P1 |

### 4.2 手动录入

| ID | 用户故事 | 优先级 |
|----|----------|--------|
| US-04 | 作为用户，我希望通过对话录入一餐饮食，以便补充训记未记录的内容 | P0 |
| US-05 | 作为用户，我希望通过对话录入体重/体脂，且手动数据优先于同步数据，以便使用更准确的测量值 | P0 |
| US-06 | 作为用户，我在确认写入前能看到变更摘要，以便避免误操作 | P0 |

### 4.3 分析与规划

| ID | 用户故事 | 优先级 |
|----|----------|--------|
| US-07 | 作为用户，我希望每天收到结构化日报，包含昨日摘要、趋势、今日饮食与训练建议 | P0 |
| US-08 | 作为用户，我希望设定目标体重/体脂并获得阶段性里程碑，以便跟踪进度 | P1 |
| US-09 | 作为用户，我希望随时询问「近 N 天」任意维度的趋势，以便灵活分析 | P0 |
| US-10 | 作为用户，我希望 Agent 建议写回训记时先预览再确认，以便保持 App 数据一致 | P1 |

### 4.4 计划管理

| ID | 用户故事 | 优先级 |
|----|----------|--------|
| US-11 | 作为用户，我希望调整今日/本周训练计划并通过对话完成，以便适应实际情况 | P1 |
| US-12 | 作为用户，我希望获得基于历史的渐进超负荷建议，以便安全提升训练量 | P2 |
| US-13 | 作为用户，我希望饮食建议考虑当日训练消耗，以便训练日/休息日差异化安排 | P1 |

---

## 5. 系统架构

### 5.1 总体架构

```mermaid
flowchart TB
    subgraph input [数据输入层]
        XunjiAPI[训记OpenAPI同步]
        ManualInput[手动录入/对话修正]
        Scheduler[定时任务调度]
    end

    subgraph storage [持久化层]
        MySQL[(MySQL)]
        Cache[查询缓存]
    end

    subgraph orchestrator [LangGraph编排层]
        Router[路由/意图识别]
        BodyAgent[体重监测与规划师]
        NutritionAgent[营养师]
        FitnessAgent[健身计划规划师]
        SummaryAgent[汇总分析Agent]
    end

    subgraph output [输出层]
        DailyReport[每日日报]
        ChatReply[对话回复]
        PlanStore[计划存储]
    end

    XunjiAPI --> MySQL
    ManualInput --> MySQL
    Scheduler --> Router
    MySQL --> Router
    Cache --> Router
    Router --> BodyAgent
    Router --> NutritionAgent
    Router --> FitnessAgent
    Router --> SummaryAgent
    BodyAgent --> SummaryAgent
    NutritionAgent --> SummaryAgent
    FitnessAgent --> SummaryAgent
    SummaryAgent --> DailyReport
    SummaryAgent --> ChatReply
    SummaryAgent --> PlanStore
    PlanStore --> MySQL
```

### 5.2 LangGraph 工作流

#### 5.2.1 日报生成流（Daily Report Flow）

```mermaid
flowchart LR
    Start([Scheduler触发]) --> Sync[同步昨日数据]
    Sync --> LoadData[加载State上下文]
    LoadData --> Parallel{并行}
    Parallel --> BodyAgent
    Parallel --> NutritionAgent
    Parallel --> FitnessAgent
    BodyAgent --> SummaryAgent
    NutritionAgent --> SummaryAgent
    FitnessAgent --> SummaryAgent
    SummaryAgent --> SaveReport[写入daily_reports]
    SaveReport --> Output[输出日报]
    Output --> End([结束])
```

#### 5.2.2 对话交互流（Chat Flow）

```mermaid
flowchart LR
    Start([用户消息]) --> Router[意图识别]
    Router -->|数据查询| QueryTools[DB/API查询]
    Router -->|手动录入| SpecialistAgent[对应Agent]
    Router -->|计划调整| FitnessOrNutrition[健身/营养Agent]
    Router -->|趋势分析| SummaryAgent
    Router -->|目标设定| BodyAgent
    Router -->|同步触发| SyncJob[触发同步]
    QueryTools --> SummaryAgent
    SpecialistAgent --> Confirm{需确认?}
    Confirm -->|是| WaitConfirm[等待用户确认]
    Confirm -->|否| SummaryAgent
    WaitConfirm --> SummaryAgent
    FitnessOrNutrition --> SummaryAgent
    BodyAgent --> SummaryAgent
    SyncJob --> SummaryAgent
    SummaryAgent --> Reply[生成回复]
    Reply --> Checkpoint[持久化Checkpoint]
    Checkpoint --> End([结束])
```

### 5.3 编排原则

| 原则 | 说明 |
|------|------|
| Supervisor/Router | 根据用户意图或任务类型路由到 Specialist Agent |
| Shared State | 所有节点读写统一的 LangGraph State |
| 汇总汇聚 | 日报、跨域分析、最终用户可见输出均经 SummaryAgent |
| 写前确认 | 写 MySQL（manual）、写回训记均需展示摘要并等待确认 |
| 并行分析 | 日报流程中 Body/Nutrition/Fitness 三个 Agent 并行执行 |
| 可观测 | 每个节点记录耗时、token 用量、输入输出摘要 |

---

## 6. Agent 职责与边界

### 6.1 体重监测与规划师（BodyMonitorAgent）

| 维度 | 内容 |
|------|------|
| **职责** | 体重/体脂/围度趋势分析；异常波动提醒；目标分解（周/月里程碑）；预测达成时间 |
| **输入** | 身体指标历史、用户目标、近期饮食与训练摘要（只读引用） |
| **输出** | 趋势解读、监测频率建议、目标调整建议、里程碑计划 |
| **工具** | `query_body_metrics`、`upsert_body_metric`（manual，需确认）、训记 Body API |
| **边界 — 可做** | 统计分析、趋势预测、目标路径规划、数据录入协助 |
| **边界 — 不可做** | 医疗诊断；替代专业体脂秤校准建议；修改饮食/训练计划（仅提供参考摘要） |
| **Prompt 约束** | 不对单次体重波动过度解读；注明「非医疗建议」；围度字段使用训记历史拼写 `weist`（腰围） |

### 6.2 营养师（NutritionistAgent）

| 维度 | 内容 |
|------|------|
| **职责** | 热量与宏量营养素分析；餐次结构评价；下一日/下一周饮食建议 |
| **输入** | 饮食记录、TDEE 估算、训练消耗、用户目标（减脂/增肌/维持） |
| **输出** | 营养缺口/盈余分析、具体食物与份量建议 |
| **工具** | `query_nutrition_logs`、`upsert_nutrition_log`、训记 Food API（query/search/upsert/custom） |
| **边界 — 可做** | 基于 Mifflin-St Jeor 的 TDEE 估算；训练日/休息日差异化热量建议；食物搜索与录入 |
| **边界 — 不可做** | 针对疾病（糖尿病、肾病等）的饮食处方；未经确认创建自定义食物；猜测食物营养值 |
| **Prompt 约束** | 创建自定义食物前必须展示营养来源；不确定餐次/份量时追问用户 |

### 6.3 健身计划规划师（FitnessPlannerAgent）

| 维度 | 内容 |
|------|------|
| **职责** | 训练量评估；分化方式建议；当日/本周训练安排；渐进超负荷建议 |
| **输入** | 训练历史、当前计划、恢复状态、用户目标 |
| **输出** | 结构化训练计划（动作、组数、重量/次数/时间、休息） |
| **工具** | `query_training_logs`、`upsert_training_plan`、训记 Training API、Plan API（只读） |
| **边界 — 可做** | 基于历史训练量的渐进建议；引用训记官方计划；RPE/难度分析 |
| **边界 — 不可做** | 带伤训练建议；编造动作中文名（须查 Xunji-movements 表）；擅自删除未完成组 |
| **Prompt 约束** | 写回动作只传中文 `name`；更新旧训练保留 `localid`、`start`、`end` |

### 6.4 汇总分析 Agent（SummaryAgent）

| 维度 | 内容 |
|------|------|
| **职责** | 跨域关联分析；生成日报；回答开放式问题；计划冲突消解 |
| **输入** | 其他 Agent 结构化输出 + 原始数据快照 + 用户消息 |
| **输出** | Markdown 日报/对话回复；持久化 `daily_reports` 与 `agent_plans` |
| **工具** | `save_daily_report`、`save_agent_plan`、`query_cross_domain` |
| **边界 — 可做** | 综合三个 Specialist 输出；发现矛盾（如热量不足但高强度训练）并提示 |
| **边界 — 不可做** | 绕过确认直接写入；医疗诊断；覆盖 Specialist 的专属工具操作 |
| **Prompt 约束** | 输出末尾附带免责声明；日报使用固定模板结构 |

### 6.5 Agent 协作矩阵

| 触发场景 | 涉及 Agent | 执行顺序 |
|----------|-----------|----------|
| 日报生成 | Body + Nutrition + Fitness → Summary | 前三者并行，Summary 串行汇聚 |
| 体重趋势问答 | Body → Summary | 串行 |
| 饮食录入 | Nutrition →（确认）→ Summary | 串行 |
| 训练计划调整 | Fitness → Summary | 串行 |
| 跨域分析 | Body + Nutrition + Fitness → Summary | 并行 → 汇聚 |
| 全量同步 | 无 Agent（Sync Service） | 独立流程 |

---

## 7. LangGraph State 与 Agent 契约

### 7.1 LangGraph State 结构

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MyFitnessGraphState",
  "type": "object",
  "required": ["user_id", "session_id", "mode"],
  "properties": {
    "user_id": { "type": "integer", "description": "用户 ID，一期默认 1" },
    "session_id": { "type": "string", "description": "对话会话 ID" },
    "mode": {
      "type": "string",
      "enum": ["daily_report", "chat"],
      "description": "运行模式"
    },
    "target_date": {
      "type": "string",
      "format": "date",
      "description": "分析目标日期，日报默认为 today，回顾 yesterday"
    },
    "user_message": { "type": "string", "description": "用户当前消息（chat 模式）" },
    "intent": {
      "type": "string",
      "enum": [
        "data_query", "manual_entry", "plan_adjust",
        "trend_analysis", "goal_setting", "sync_trigger", "general"
      ]
    },
    "messages": {
      "type": "array",
      "items": { "$ref": "#/$defs/ChatMessage" },
      "description": "对话历史"
    },
    "context": { "$ref": "#/$defs/ContextSnapshot" },
    "agent_outputs": {
      "type": "object",
      "properties": {
        "body": { "$ref": "#/$defs/BodyAgentOutput" },
        "nutrition": { "$ref": "#/$defs/NutritionAgentOutput" },
        "fitness": { "$ref": "#/$defs/FitnessAgentOutput" },
        "summary": { "$ref": "#/$defs/SummaryAgentOutput" }
      }
    },
    "pending_confirmation": { "$ref": "#/$defs/PendingConfirmation" },
    "errors": {
      "type": "array",
      "items": { "type": "string" }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "started_at": { "type": "string", "format": "date-time" },
        "token_usage": { "type": "object" },
        "agents_invoked": { "type": "array", "items": { "type": "string" } }
      }
    }
  },
  "$defs": {
    "ChatMessage": {
      "type": "object",
      "required": ["role", "content"],
      "properties": {
        "role": { "type": "string", "enum": ["user", "assistant", "system", "tool"] },
        "content": { "type": "string" },
        "timestamp": { "type": "string", "format": "date-time" }
      }
    },
    "ContextSnapshot": {
      "type": "object",
      "properties": {
        "date_range": {
          "type": "object",
          "properties": {
            "start": { "type": "string", "format": "date" },
            "end": { "type": "string", "format": "date" }
          }
        },
        "body_metrics_summary": { "type": "object" },
        "nutrition_summary": { "type": "object" },
        "training_summary": { "type": "object" },
        "user_goals": { "type": "array" },
        "active_plans": { "type": "array" },
        "data_gaps": {
          "type": "array",
          "items": { "type": "string" },
          "description": "缺失的数据项，如同步失败日期"
        }
      }
    },
    "PendingConfirmation": {
      "type": "object",
      "properties": {
        "action_type": {
          "type": "string",
          "enum": ["db_write", "xunji_write", "plan_update"]
        },
        "summary": { "type": "string", "description": "给用户看的变更摘要" },
        "payload": { "type": "object", "description": "待执行的写入数据" },
        "expires_at": { "type": "string", "format": "date-time" }
      }
    }
  }
}
```

### 7.2 BodyAgentOutput

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "BodyAgentOutput",
  "type": "object",
  "required": ["agent", "analysis_date"],
  "properties": {
    "agent": { "const": "body_monitor" },
    "analysis_date": { "type": "string", "format": "date" },
    "current_metrics": {
      "type": "object",
      "properties": {
        "weight_kg": { "type": "number" },
        "bodyfat_pct": { "type": "number" },
        "measurements": { "type": "object", "additionalProperties": { "type": "number" } }
      }
    },
    "trend": {
      "type": "object",
      "properties": {
        "period_days": { "type": "integer" },
        "weight_change_kg": { "type": "number" },
        "bodyfat_change_pct": { "type": "number" },
        "trend_direction": { "type": "string", "enum": ["up", "down", "stable", "insufficient_data"] },
        "weekly_avg_weight_kg": { "type": "number" }
      }
    },
    "anomalies": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "date": { "type": "string", "format": "date" },
          "metric": { "type": "string" },
          "description": { "type": "string" },
          "severity": { "type": "string", "enum": ["info", "warning"] }
        }
      }
    },
    "goal_progress": {
      "type": "object",
      "properties": {
        "goal_type": { "type": "string" },
        "target_value": { "type": "number" },
        "current_value": { "type": "number" },
        "progress_pct": { "type": "number" },
        "estimated_target_date": { "type": "string", "format": "date" },
        "milestones": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "date": { "type": "string", "format": "date" },
              "target_value": { "type": "number" },
              "label": { "type": "string" }
            }
          }
        }
      }
    },
    "recommendations": {
      "type": "array",
      "items": { "type": "string" }
    },
    "monitoring_frequency": {
      "type": "string",
      "enum": ["daily", "every_other_day", "weekly"]
    },
    "narrative": { "type": "string", "description": "自然语言解读，供 Summary 引用" }
  }
}
```

### 7.3 NutritionAgentOutput

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NutritionAgentOutput",
  "type": "object",
  "required": ["agent", "analysis_date"],
  "properties": {
    "agent": { "const": "nutritionist" },
    "analysis_date": { "type": "string", "format": "date" },
    "daily_totals": {
      "type": "object",
      "properties": {
        "calories": { "type": "number" },
        "protein_g": { "type": "number" },
        "carbs_g": { "type": "number" },
        "fat_g": { "type": "number" }
      }
    },
    "tdee_estimate": {
      "type": "object",
      "properties": {
        "method": { "const": "mifflin_st_jeor" },
        "bmr": { "type": "number" },
        "activity_factor": { "type": "number" },
        "tdee": { "type": "number" },
        "target_calories": { "type": "number" },
        "goal_mode": { "type": "string", "enum": ["cut", "bulk", "maintain"] }
      }
    },
    "balance": {
      "type": "object",
      "properties": {
        "calorie_delta": { "type": "number", "description": "实际 - 目标，负值为缺口" },
        "protein_per_kg": { "type": "number" },
        "assessment": { "type": "string", "enum": ["deficit", "surplus", "on_target", "unknown"] }
      }
    },
    "meal_analysis": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "meal_type": { "type": "string" },
          "calories": { "type": "number" },
          "comment": { "type": "string" }
        }
      }
    },
    "tomorrow_suggestions": {
      "type": "object",
      "properties": {
        "target_calories": { "type": "number" },
        "macro_targets": {
          "type": "object",
          "properties": {
            "protein_g": { "type": "number" },
            "carbs_g": { "type": "number" },
            "fat_g": { "type": "number" }
          }
        },
        "meal_ideas": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "meal_type": { "type": "string" },
              "foods": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "name": { "type": "string" },
                    "amount": { "type": "number" },
                    "unit": { "type": "string" }
                  }
                }
              }
            }
          }
        }
      }
    },
    "recommendations": {
      "type": "array",
      "items": { "type": "string" }
    },
    "narrative": { "type": "string" }
  }
}
```

### 7.4 FitnessAgentOutput

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "FitnessAgentOutput",
  "type": "object",
  "required": ["agent", "analysis_date"],
  "properties": {
    "agent": { "const": "fitness_planner" },
    "analysis_date": { "type": "string", "format": "date" },
    "recent_training_summary": {
      "type": "object",
      "properties": {
        "sessions_last_7d": { "type": "integer" },
        "total_volume_kg": { "type": "number" },
        "avg_rpe": { "type": "number" },
        "muscle_groups_trained": { "type": "array", "items": { "type": "string" } }
      }
    },
    "recovery_assessment": {
      "type": "string",
      "enum": ["well_recovered", "moderate_fatigue", "high_fatigue", "unknown"]
    },
    "today_plan": {
      "type": "object",
      "properties": {
        "session_type": { "type": "string", "enum": ["strength", "cardio", "rest", "active_recovery"] },
        "focus": { "type": "string" },
        "movements": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": { "type": "string" },
              "sets": { "type": "integer" },
              "reps": { "type": "string" },
              "weight": { "type": "string" },
              "rest_seconds": { "type": "integer" },
              "notes": { "type": "string" }
            }
          }
        },
        "estimated_duration_min": { "type": "integer" }
      }
    },
    "weekly_outline": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "date": { "type": "string", "format": "date" },
          "session_type": { "type": "string" },
          "focus": { "type": "string" }
        }
      }
    },
    "progressive_overload": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "movement": { "type": "string" },
          "suggestion": { "type": "string" },
          "basis": { "type": "string" }
        }
      }
    },
    "recommendations": {
      "type": "array",
      "items": { "type": "string" }
    },
    "narrative": { "type": "string" }
  }
}
```

### 7.5 SummaryAgentOutput

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SummaryAgentOutput",
  "type": "object",
  "required": ["agent", "output_type"],
  "properties": {
    "agent": { "const": "summary" },
    "output_type": { "type": "string", "enum": ["daily_report", "chat_reply"] },
    "content_md": { "type": "string", "description": "最终 Markdown 输出" },
    "cross_domain_insights": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "insight": { "type": "string" },
          "domains": { "type": "array", "items": { "type": "string" } },
          "confidence": { "type": "string", "enum": ["high", "medium", "low"] }
        }
      }
    },
    "conflicts_resolved": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "conflict": { "type": "string" },
          "resolution": { "type": "string" }
        }
      }
    },
    "action_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "priority": { "type": "string", "enum": ["high", "medium", "low"] },
          "action": { "type": "string" },
          "domain": { "type": "string" }
        }
      }
    },
    "data_quality_notes": {
      "type": "array",
      "items": { "type": "string" },
      "description": "如同步失败、数据缺失提示"
    },
    "disclaimer": {
      "type": "string",
      "const": "以上内容仅供参考，不构成医疗建议。如有健康问题请咨询专业医生。"
    }
  }
}
```

---

## 8. 数据模型与 DDL

### 8.1 ER 关系

```mermaid
erDiagram
    users ||--o{ body_metrics : has
    users ||--o{ nutrition_logs : has
    users ||--o{ training_logs : has
    users ||--o{ user_goals : has
    users ||--o{ daily_reports : has
    users ||--o{ agent_plans : has
    users ||--o{ sync_jobs : has
    users ||--o{ chat_sessions : has
    nutrition_logs }o--o| foods : references
    training_logs ||--o{ training_exercises : contains
    chat_sessions ||--o{ chat_messages : contains

    users {
        bigint id PK
        varchar name
        json profile
        datetime created_at
        datetime updated_at
    }
    body_metrics {
        bigint id PK
        bigint user_id FK
        date record_date
        varchar metric_type
        decimal value
        varchar unit
        varchar source
        varchar xunji_ref
        datetime synced_at
        datetime updated_at
    }
    foods {
        bigint id PK
        varchar name
        varchar uniquekey
        json ntr
        json units
        varchar source
    }
    nutrition_logs {
        bigint id PK
        bigint user_id FK
        date record_date
        varchar meal_type
        bigint food_id FK
        varchar food_name
        decimal amount
        varchar unit
        json nutrients_snapshot
        varchar source
        varchar xunji_record_id
        datetime updated_at
    }
    training_logs {
        bigint id PK
        bigint user_id FK
        date record_date
        json raw_payload
        varchar source
        varchar xunji_localid
        datetime updated_at
    }
    training_exercises {
        bigint id PK
        bigint training_log_id FK
        varchar movement_name
        int set_count
        json sets_detail
    }
    user_goals {
        bigint id PK
        bigint user_id FK
        varchar goal_type
        decimal target_value
        decimal start_value
        date start_date
        date target_date
        json metadata
        varchar status
    }
    daily_reports {
        bigint id PK
        bigint user_id FK
        date report_date
        text content_md
        json agent_outputs
        datetime created_at
    }
    agent_plans {
        bigint id PK
        bigint user_id FK
        varchar plan_type
        date start_date
        date end_date
        json plan_json
        varchar status
        datetime created_at
        datetime updated_at
    }
    sync_jobs {
        bigint id PK
        bigint user_id FK
        varchar sync_type
        date sync_start_date
        date sync_end_date
        datetime last_run_at
        varchar status
        text error_log
        json stats
    }
    chat_sessions {
        bigint id PK
        bigint user_id FK
        varchar session_id
        datetime created_at
        datetime last_active_at
    }
    chat_messages {
        bigint id PK
        bigint session_id FK
        varchar role
        text content
        json metadata
        datetime created_at
    }
```

### 8.2 DDL 草案

```sql
-- MyFitness v1.0 DDL
-- MySQL 8.x, utf8mb4

CREATE TABLE users (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL DEFAULT 'default',
    profile     JSON COMMENT '性别、年龄、身高、活动水平等',
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE body_metrics (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id     BIGINT NOT NULL,
    record_date DATE NOT NULL,
    metric_type VARCHAR(32) NOT NULL COMMENT 'weight|bodyfat|neck|chest|weist|...',
    value       DECIMAL(10, 2) NOT NULL,
    unit        VARCHAR(16) NOT NULL COMMENT 'kg|%|cm',
    source      VARCHAR(32) NOT NULL DEFAULT 'xunji_sync' COMMENT 'xunji_sync|manual|agent_suggested',
    xunji_ref   VARCHAR(128) NULL COMMENT '训记记录关联',
    synced_at   DATETIME NULL,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_body_metric (user_id, record_date, metric_type, source),
    INDEX idx_body_user_date (user_id, record_date),
    CONSTRAINT fk_body_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE foods (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(200) NOT NULL,
    uniquekey   VARCHAR(128) NULL,
    ntr         JSON NOT NULL COMMENT '每100g: cal, protein, fat, carb',
    units       JSON NULL,
    source      VARCHAR(32) NOT NULL DEFAULT 'xunji' COMMENT 'xunji|custom|manual',
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_food_name (name),
    INDEX idx_food_uniquekey (uniquekey)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE nutrition_logs (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id             BIGINT NOT NULL,
    record_date         DATE NOT NULL,
    meal_type           VARCHAR(32) NOT NULL COMMENT 'breakfast|lunch|dinner|snack|other',
    food_id             BIGINT NULL,
    food_name           VARCHAR(200) NOT NULL,
    amount              DECIMAL(10, 2) NOT NULL,
    unit                VARCHAR(32) NOT NULL DEFAULT 'g',
    nutrients_snapshot  JSON NOT NULL COMMENT '写入时快照 cal/protein/fat/carb',
    source              VARCHAR(32) NOT NULL DEFAULT 'xunji_sync',
    xunji_record_id     VARCHAR(128) NULL,
    updated_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_nutrition_user_date (user_id, record_date),
    CONSTRAINT fk_nutrition_user FOREIGN KEY (user_id) REFERENCES users(id),
    CONSTRAINT fk_nutrition_food FOREIGN KEY (food_id) REFERENCES foods(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE training_logs (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id         BIGINT NOT NULL,
    record_date     DATE NOT NULL,
    title           VARCHAR(200) NULL,
    raw_payload     JSON NOT NULL COMMENT '训记原始 JSON，含 movements/sets/RPE/heartRate',
    source          VARCHAR(32) NOT NULL DEFAULT 'xunji_sync',
    xunji_localid   VARCHAR(64) NULL,
    updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_training_localid (user_id, xunji_localid),
    INDEX idx_training_user_date (user_id, record_date),
    CONSTRAINT fk_training_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE training_exercises (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    training_log_id BIGINT NOT NULL,
    movement_name   VARCHAR(100) NOT NULL,
    set_count       INT NOT NULL DEFAULT 0,
    sets_detail     JSON NULL COMMENT '组详情摘要',
    CONSTRAINT fk_exercise_log FOREIGN KEY (training_log_id) REFERENCES training_logs(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE user_goals (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id       BIGINT NOT NULL,
    goal_type     VARCHAR(32) NOT NULL COMMENT 'weight|bodyfat|muscle|custom',
    target_value  DECIMAL(10, 2) NOT NULL,
    start_value   DECIMAL(10, 2) NULL,
    start_date    DATE NOT NULL,
    target_date   DATE NULL,
    metadata      JSON NULL COMMENT '如 goal_mode: cut/bulk/maintain',
    status        VARCHAR(16) NOT NULL DEFAULT 'active' COMMENT 'active|achieved|abandoned',
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_goal_user (user_id, status),
    CONSTRAINT fk_goal_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE daily_reports (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id       BIGINT NOT NULL,
    report_date   DATE NOT NULL,
    content_md    TEXT NOT NULL,
    agent_outputs JSON NULL COMMENT '各 Agent 结构化输出快照',
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_report (user_id, report_date),
    CONSTRAINT fk_report_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE agent_plans (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id     BIGINT NOT NULL,
    plan_type   VARCHAR(32) NOT NULL COMMENT 'nutrition|fitness|body',
    start_date  DATE NOT NULL,
    end_date    DATE NOT NULL,
    plan_json   JSON NOT NULL,
    status      VARCHAR(16) NOT NULL DEFAULT 'active' COMMENT 'active|superseded|cancelled',
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_plan_user (user_id, plan_type, status),
    CONSTRAINT fk_plan_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE sync_jobs (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id         BIGINT NOT NULL,
    sync_type       VARCHAR(32) NOT NULL COMMENT 'body|food|training|all',
    sync_start_date DATE NULL,
    sync_end_date   DATE NULL,
    last_run_at     DATETIME NULL,
    status          VARCHAR(16) NOT NULL DEFAULT 'pending' COMMENT 'pending|running|success|failed|partial',
    error_log       TEXT NULL,
    stats           JSON NULL COMMENT '同步条数、耗时等',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_sync_user (user_id, sync_type),
    CONSTRAINT fk_sync_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE chat_sessions (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id         BIGINT NOT NULL,
    session_id      VARCHAR(64) NOT NULL,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_active_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_session (session_id),
    CONSTRAINT fk_chat_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE chat_messages (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id  BIGINT NOT NULL,
    role        VARCHAR(16) NOT NULL COMMENT 'user|assistant|system|tool',
    content     TEXT NOT NULL,
    metadata    JSON NULL,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_msg_session (session_id, created_at),
    CONSTRAINT fk_msg_session FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 8.3 字段与枚举说明

#### metric_type（身体指标）

| 值 | 含义 | 单位 |
|----|------|------|
| weight | 体重 | kg |
| bodyfat | 体脂率 | % |
| neck, chest, weist, shoulder, bot | 脖/胸/腰/肩/臀围 | cm |
| arm_left/right, forearm_left/right | 臂/小臂围 | cm |
| leg_left/right, cav_left/right | 腿/小腿围 | cm |

> 注意：腰围字段沿用训记历史拼写 `weist`，不要改为 `waist`。

#### source（数据来源）

| 值 | 含义 | 优先级 |
|----|------|--------|
| manual | 用户手动录入 | 最高 |
| xunji_sync | 训记 API 同步 | 中 |
| agent_suggested | Agent 建议（未写回训记） | 最低 |

#### 有效值合并规则

查询「当日有效值」时，按 `source` 优先级取最高优先级的记录；同优先级取 `updated_at` 最新。

---

## 9. 训记 Open API 集成规范

### 9.1 API 端点一览

| 数据域 | Base URL | 读接口 | 写接口 |
|--------|----------|--------|--------|
| 身体 | `https://api.xunjiapp.cn` | `POST /open/body/query_gzip` | `POST /open/body/upsert_gzip` |
| 饮食 | `https://eatings.xunjiapp.cn` | `POST /open/food/query_gzip` | `POST /open/food/upsert_gzip` |
| 食物搜索 | `https://api.xunjiapp.cn` | `POST /open_agent/food/search_gzip` | — |
| 自定义食物 | `https://eatings.xunjiapp.cn` | — | `POST /open/food/custom/upsert_gzip` |
| 饮食模板 | `https://eatings.xunjiapp.cn` | `POST /open/food/templates/list_gzip` | `POST /open/food/templates/apply_gzip` |
| 训练 | `https://trains.xunjiapp.cn` | `POST /api_trains_for_llm_v2` | `POST /api_upsert_trains_for_llm_v2` |
| 官方计划 | `https://api.xunjiapp.cn` | `POST /open/plan/query_gzip` | 只读 |

### 9.2 API Key 管理规范

| 规则 | 说明 |
|------|------|
| 存储位置 | 环境变量 / `.env` 文件，**禁止**硬编码在代码或提交到 Git |
| 变量命名 | `XUNJI_BODY_API_KEY`、`XUNJI_FOOD_API_KEY`、`XUNJI_FOOD_SEARCH_KEY`、`XUNJI_TRAINING_API_KEY` |
| 请求头 | `Authorization: Bearer <key>`，兼容 `x-api-key` |
| 日志脱敏 | 日志中 Key 只显示末 4 位，如 `****c3bf` |
| 失效处理 | 收到 `apikey invalid` 时提示用户回 App 重新申请 Key |
| VIP 限制 | 收到 `仅VIP可用` 时在 UI/回复中明确提示 |
| `.gitignore` | 必须包含 `.env` |

### 9.3 混合同步流程

```mermaid
sequenceDiagram
    participant Scheduler
    participant SyncService
    participant XunjiAPI
    participant MySQL
    participant Cache

    Scheduler->>SyncService: 触发 sync_jobs
    SyncService->>MySQL: 读取上次同步水位线
    SyncService->>Cache: 检查缓存是否命中

    alt 缓存未命中
        SyncService->>XunjiAPI: 按日期范围 query
        XunjiAPI-->>SyncService: 返回 records
        SyncService->>Cache: 写入缓存
    end

    SyncService->>SyncService: 标准化为内部 schema
    SyncService->>MySQL: upsert (source=xunji_sync)

    alt 与 manual 冲突
        SyncService->>SyncService: manual 优先，跳过覆盖
    end

    SyncService->>MySQL: 更新 sync_jobs status/stats
```

#### 9.3.1 同步策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 定时同步 | 每日 07:00 | 可通过配置修改 |
| 增量范围 | 上次成功同步日期 → 昨天 | 首次同步默认近 90 天 |
| 手动触发 | 用户指定 start_date / end_date | 对话或 CLI 触发 |
| 并行度 | body / food / training 三路并行 | 各自独立 sync_job |
| 失败重试 | 最多 3 次，指数退避 | 基于 `retry_after_ms` |

#### 9.3.2 标准化映射

**身体数据**

```
训记 record → body_metrics
  datestr     → record_date
  type        → metric_type
  value       → value
  unit        → unit
  (generated) → xunji_ref = "{datestr}:{type}"
  source      → xunji_sync
```

**饮食数据**

```
训记 food entry → nutrition_logs + foods
  date          → record_date
  meal_type     → meal_type
  name          → food_name
  amount/unit   → amount/unit
  ntr           → nutrients_snapshot
  uniquekey     → foods.uniquekey
  source        → xunji_sync
```

**训练数据**

```
训记 train → training_logs + training_exercises
  datestr     → record_date
  localid     → xunji_localid
  full JSON   → raw_payload
  movements[] → training_exercises (解析摘要)
  source      → xunji_sync
```

### 9.4 数据冲突策略

| 场景 | 策略 | 实现 |
|------|------|------|
| 同天同指标，训记 vs 手动 | manual 优先，不覆盖 | upsert 时检查 `(user_id, date, type)` 是否存在 manual 记录 |
| 同 source 重复同步 | 以训记最新值更新 | 比较 `synced_at`，更新 value |
| 训记同步失败 | 保留本地数据 | sync_jobs.status=failed，日报 data_quality_notes 标注 |
| Agent 建议写回训记 | dry_run → 确认 → write | Body: `confirmed: true`；Food/Training: 展示摘要等用户确认 |
| 查询有效值 | 按 source 优先级 | 见 8.3 合并规则 |

### 9.5 限流与缓存

| 接口 | 限频 | 缓存 TTL |
|------|------|----------|
| Body query/upsert | 15 秒/endpoint/key | query: 同条件 5 分钟 |
| Food query/upsert/custom | 15 秒/endpoint/key | query: 同条件 5 分钟 |
| Food search | 15 秒/key | 同 keyword: 10 分钟 |
| Training read (light) | 15 秒/日/key | 同日: 5 分钟 |
| Training read (full) | 30 秒/日/key | 同日: 5 分钟 |
| Training write | 45 秒/日/key | 不缓存 |
| Plan list/get | 15 秒/操作/key | list: 30 分钟；get: 10 分钟 |

**退避策略**：收到 `too frequent` 时，等待响应中的 `retry_after_ms`（缺省 15000ms），然后重试，最多 3 次。

### 9.6 写回确认流程

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant XunjiAPI

    Agent->>XunjiAPI: dry_run=true
    XunjiAPI-->>Agent: res.summary
    Agent->>User: 展示变更摘要
    User->>Agent: 确认 / 取消
    alt 用户确认
        Agent->>XunjiAPI: dry_run=false, confirmed=true
        XunjiAPI-->>Agent: 写入成功
        Agent->>User: 确认完成
    else 用户取消
        Agent->>User: 已取消，未写入
    end
```

---

## 10. 交互设计

### 10.1 定时日报

#### 10.1.1 触发与流程

| 步骤 | 动作 |
|------|------|
| 1 | Scheduler 在配置时间（默认 07:00）触发 |
| 2 | 执行训记增量同步（body + food + training） |
| 3 | 加载 State 上下文（昨日数据 + 近 30 天摘要 + 用户目标） |
| 4 | 并行调用 Body / Nutrition / Fitness Agent |
| 5 | Summary Agent 生成日报 |
| 6 | 写入 `daily_reports` 表 |
| 7 | 输出到配置渠道（一期：本地文件 `reports/YYYY-MM-DD.md` + stdout） |

#### 10.1.2 日报 Markdown 模板

```markdown
# MyFitness 日报 — {report_date}

> 生成时间：{generated_at}  
> 数据覆盖：{data_range_start} ~ {data_range_end}  
> {data_quality_banner}

---

## 1. 昨日数据摘要

### 身体
| 指标 | 数值 | 较7日均 |
|------|------|---------|
| 体重 | {weight_kg} kg | {weight_delta} |
| 体脂 | {bodyfat_pct} % | {bodyfat_delta} |

### 饮食
| 项目 | 数值 |
|------|------|
| 总热量 | {calories} kcal |
| 蛋白质 | {protein_g} g |
| 碳水 | {carbs_g} g |
| 脂肪 | {fat_g} g |
|  vs 目标 | {calorie_delta} kcal ({assessment}) |

### 训练
{training_summary_or_rest_day}

---

## 2. 趋势亮点与异常

{trend_highlights}

{anomaly_alerts}

---

## 3. 跨域洞察

{cross_domain_insights}

---

## 4. 今日饮食建议

**目标热量**：{target_calories} kcal  
**宏量目标**：蛋白 {protein_g}g / 碳水 {carbs_g}g / 脂肪 {fat_g}g

{meal_suggestions}

---

## 5. 今日训练建议

**类型**：{session_type} — {focus}

{movement_list}

---

## 6. 体重目标进度

| 项目 | 值 |
|------|-----|
| 目标 | {goal_description} |
| 当前进度 | {progress_pct}% |
| 预计达成 | {estimated_date} |
| 下一里程碑 | {next_milestone} |

---

## 7. 行动项

{action_items}

---

*{disclaimer}*
```

#### 10.1.3 日报配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `DAILY_REPORT_TIME` | `07:00` | 触发时间（本地时区） |
| `DAILY_REPORT_LOOKBACK_DAYS` | `30` | 趋势分析回溯天数 |
| `DAILY_REPORT_MODE` | `full` | `full` 完整版 / `summary` 摘要版（节省 token） |
| `DAILY_REPORT_OUTPUT_DIR` | `./reports` | 本地输出目录 |

### 10.2 对话式交互

#### 10.2.1 意图分类

| 意图 | 标识 | 示例 utterance | 路由目标 |
|------|------|----------------|----------|
| 数据查询 | `data_query` | 「昨天吃了多少蛋白质？」 | Query Tools → Summary |
| 手动录入 | `manual_entry` | 「记录午餐：鸡胸肉 200g」 | Nutrition/Body Agent → 确认 → Summary |
| 计划调整 | `plan_adjust` | 「今天不练了，改成休息」 | Fitness/Nutrition Agent → Summary |
| 趋势分析 | `trend_analysis` | 「近 30 天体脂变化趋势」 | Body (+ 可选 Nutrition/Fitness) → Summary |
| 目标设定 | `goal_setting` | 「目标体重设为 70kg，12 周达成」 | Body Agent → Summary |
| 同步触发 | `sync_trigger` | 「同步最近 7 天训记数据」 | Sync Service → Summary |
| 通用 | `general` | 「你好」 | Summary 直接回复 |

#### 10.2.2 意图识别规则

1. **LLM 分类优先**：Router 节点使用 LLM + 上述意图枚举做结构化输出
2. **关键词兜底**：

| 关键词模式 | 意图 |
|-----------|------|
| 记录/录入/添加 + 食物/餐/吃 | manual_entry (nutrition) |
| 记录/录入 + 体重/体脂 | manual_entry (body) |
| 改成/调整/取消 + 训练/计划/休息 | plan_adjust |
| 近 N 天/趋势/变化/对比 | trend_analysis |
| 目标/降到/增到 + kg/% | goal_setting |
| 同步/拉取/更新 + 训记/数据 | sync_trigger |
| 多少/查询/昨天/今天 + 数据 | data_query |

3. **多意图拆分**：「同步数据然后分析近 7 天趋势」→ 串行执行 sync_trigger → trend_analysis

#### 10.2.3 多轮上下文规则

| 规则 | 说明 |
|------|------|
| 会话持久化 | LangGraph Checkpoint 存 MySQL/SQLite，key = `session_id` |
| 上下文窗口 | 最近 20 轮对话 + 结构化 State 摘要 |
| 数据上下文 | 每次请求加载最近 7 天 MySQL 摘要（可配置） |
| 确认超时 | `pending_confirmation` 30 分钟过期，过期需重新发起 |
| 确认续接 | 用户回复「确认」/「取消」→ Router 识别为 confirmation_response，读取 pending payload |
| 计划上下文 | 加载当前 active 的 `agent_plans`（nutrition + fitness） |
| 日报引用 | 对话可引用当日/昨日 `daily_reports.content_md` 片段 |

#### 10.2.4 对话示例

**示例 1：手动录入**

```
用户：午餐吃了 200g 鸡胸肉和一个苹果
系统：[营养师] 估算如下，请确认：
  - 鸡胸肉 200g：约 330 kcal，蛋白 62g
  - 苹果 1 个（约 180g）：约 94 kcal
  是否写入本地数据库？
用户：确认
系统：已记录。今日午餐合计 xxx kcal...
```

**示例 2：跨域分析**

```
用户：为什么这周体重涨了但体脂降了？
系统：[汇总] 综合分析如下：
  1. 体重 +0.8kg，但 7 日平均训练量 +15%
  2. 蛋白质摄入均值 1.8g/kg，处于增肌区间
  3. 推测原因为肌肉糖原+水分增加...
```

---

## 11. 接口草案

### 11.1 CLI 命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `myfitness sync` | 触发训记同步 | `myfitness sync --days 7` |
| `myfitness report` | 生成日报 | `myfitness report --date 2026-08-21` |
| `myfitness chat` | 启动对话 | `myfitness chat` |
| `myfitness goals set` | 设定目标 | `myfitness goals set weight 70 --by 2026-11-22` |
| `myfitness db migrate` | 执行数据库迁移 | `myfitness db migrate` |

### 11.2 REST API（一期可选）

| Method | Path | 说明 |
|--------|------|------|
| POST | `/api/v1/sync` | 触发同步 `{ "start_date", "end_date", "types" }` |
| GET | `/api/v1/reports/{date}` | 获取日报 |
| POST | `/api/v1/reports/generate` | 手动生成日报 |
| POST | `/api/v1/chat` | 发送消息 `{ "session_id", "message" }` |
| POST | `/api/v1/chat/confirm` | 确认待写入 `{ "session_id", "confirmed": true }` |
| GET | `/api/v1/metrics/body` | 查询身体数据 `?start=&end=` |
| GET | `/api/v1/metrics/nutrition` | 查询饮食数据 |
| GET | `/api/v1/metrics/training` | 查询训练数据 |
| GET/POST | `/api/v1/goals` | 目标 CRUD |
| GET | `/api/v1/health` | 健康检查 |

---

## 12. 非功能需求

| 类别 | 要求 |
|------|------|
| **安全** | API Key / DB 凭据仅环境变量；健康数据本地优先存储；日志脱敏 |
| **可靠性** | 训记 API 限流退避；同步失败降级；Agent 超时可跳过并标注 |
| **性能** | 日报生成 < 120 秒（含 LLM）；对话常见查询 < 30 秒 |
| **可观测** | 结构化日志：agent_name, duration_ms, token_count, sync_stats |
| **可扩展** | Agent 可插拔；表结构预留 user_id；LLM provider 可配置 |
| **合规** | 所有输出含免责声明；不做医疗诊断 |
| **可测试** | 同步/DB 层可 mock 训记 API；Agent 输出可 JSON Schema 校验 |

---

## 13. 技术栈与项目结构

### 13.1 技术栈

| 层级 | 选型 |
|------|------|
| 语言 | Python 3.11+ |
| Agent 编排 | LangGraph + LangChain |
| LLM | 可配置（OpenAI / Anthropic / 本地 Ollama） |
| 数据库 | MySQL 8.x |
| ORM / 迁移 | SQLAlchemy 2.x + Alembic |
| 调度 | APScheduler |
| 配置 | pydantic-settings + `.env` |
| HTTP 客户端 | httpx（训记 API） |
| CLI | Typer |
| 测试 | pytest |

### 13.2 项目结构

```
MyFitness/
├── docs/
│   └── PRD.md                 # 本文档
├── src/
│   ├── __init__.py
│   ├── config.py              # pydantic-settings 配置
│   ├── agents/
│   │   ├── body_monitor.py
│   │   ├── nutritionist.py
│   │   ├── fitness_planner.py
│   │   ├── summary.py
│   │   └── tools/             # Agent 工具函数
│   ├── graph/
│   │   ├── state.py           # LangGraph State 定义
│   │   ├── router.py          # 意图路由
│   │   ├── daily_report.py    # 日报工作流
│   │   └── chat.py            # 对话工作流
│   ├── db/
│   │   ├── models.py
│   │   ├── repositories/
│   │   └── session.py
│   ├── sync/
│   │   ├── body_sync.py
│   │   ├── food_sync.py
│   │   ├── training_sync.py
│   │   └── xunji_client.py    # 训记 API 客户端（含限流/缓存）
│   ├── scheduler/
│   │   └── daily_jobs.py
│   └── api/
│       ├── cli.py
│       └── routes.py          # REST API（可选）
├── migrations/                # Alembic 迁移
├── reports/                   # 日报输出目录
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

### 13.3 环境变量（.env.example）

```env
# MySQL
DATABASE_URL=mysql+pymysql://user:pass@localhost:3306/myfitness

# 训记 API Keys
XUNJI_BODY_API_KEY=
XUNJI_FOOD_API_KEY=
XUNJI_FOOD_SEARCH_KEY=
XUNJI_TRAINING_API_KEY=

# LLM
LLM_PROVIDER=openai
OPENAI_API_KEY=
LLM_MODEL=gpt-4o

# 调度
DAILY_REPORT_TIME=07:00
DAILY_REPORT_MODE=full
DAILY_REPORT_OUTPUT_DIR=./reports

# 应用
LOG_LEVEL=INFO
DEFAULT_USER_ID=1
```

---

## 14. 里程碑与验收标准

### M1 — 基础数据层（预计 1-2 周）

**交付物**
- MySQL schema 迁移脚本
- 训记只读同步（body / food / training）
- 数据查询 repository 层
- CLI: `sync`, `db migrate`

**验收标准**
- [ ] 执行迁移后全部表创建成功
- [ ] `myfitness sync --days 7` 成功拉取并入库三类数据
- [ ] 同步失败写入 sync_jobs.error_log
- [ ] manual 记录不被 sync 覆盖
- [ ] API Key 从环境变量读取，代码/日志无明文 Key

### M2 — Agent 核心（预计 2-3 周）

**交付物**
- 4 个 Agent 实现 + 工具函数
- LangGraph State + Router
- 手动录入（对话 → 确认 → 写库）
- Agent 输出 JSON Schema 校验

**验收标准**
- [ ] 对话录入体重/饮食，确认后写入 MySQL（source=manual）
- [ ] 各 Agent 输出符合第 7 节 JSON Schema
- [ ] Router 正确分类 7 类意图（测试集 ≥ 90% 准确率）
- [ ] Agent 不做医疗诊断，输出含免责声明

### M3 — 日报（预计 1 周）

**交付物**
- 日报 LangGraph 工作流
- APScheduler 定时任务
- 日报 Markdown 输出

**验收标准**
- [ ] `myfitness report` 生成符合模板的 Markdown 文件
- [ ] 定时任务在配置时间自动触发
- [ ] 数据缺失时日报含 data_quality_notes
- [ ] 日报写入 daily_reports 表且可查询

### M4 — 对话完善（预计 1-2 周）

**交付物**
- 多轮对话 + Checkpoint 持久化
- 计划调整（agent_plans CRUD）
- 训记写回（带 dry_run 确认）
- 跨域趋势分析

**验收标准**
- [ ] 多轮对话上下文保持（20 轮内指代消解）
- [ ] 「今天改成休息」更新 agent_plans 并在后续对话生效
- [ ] 写回训记前展示摘要，取消则不写入
- [ ] 「近 30 天趋势」返回跨 body/nutrition/training 的综合分析

### M5 — 打磨（预计 1 周）

**交付物**
- 缓存层（训记 query 缓存）
- 结构化日志 + sync 审计
- 单元测试 + 集成测试
- README + .env.example

**验收标准**
- [ ] 相同 query 条件不重复请求训记 API（缓存命中）
- [ ] 限流时正确退避重试
- [ ] 核心路径测试覆盖率 ≥ 70%
- [ ] 新用户可按 README 完成部署并跑通 sync → report → chat

---

## 15. 风险与待定项

| 风险/待定 | 影响 | 缓解措施 |
|-----------|------|----------|
| 训记 API 限流 | 同步延迟 | 退避重试 + 缓存 + 串行化同 endpoint 请求 |
| 训记 API 变更 | 同步失败 | xunji_client 抽象层隔离；schema_version 校验 |
| LLM 成本 | 日报费用 | 可配置 summary 模式；缓存 Agent 中间结果 |
| LLM 幻觉 | 错误建议 | 结构化输出 + Schema 校验；Disclaimer |
| TDEE 精度 | 饮食建议偏差 | 一期 Mifflin-St Jeor；标注可后续接入 InBody 等 |
| 推送渠道 | 用户触达 | 一期本地文件；二期邮件/微信/Telegram |
| 多用户 | 架构限制 | 一期单用户；表结构预留 user_id |
| 心率/睡眠数据 | 分析维度不足 | 训记训练内心率已支持；睡眠列为二期 |

---

## 16. 附录

### 16.1 TDEE 估算（一期）

采用 **Mifflin-St Jeor** 公式：

```
BMR (男) = 10 × 体重(kg) + 6.25 × 身高(cm) - 5 × 年龄 + 5
BMR (女) = 10 × 体重(kg) + 6.25 × 身高(cm) - 5 × 年龄 - 161
TDEE = BMR × 活动系数
```

| 活动水平 | 系数 |
|----------|------|
| 久坐 | 1.2 |
| 轻度活动 | 1.375 |
| 中度活动 | 1.55 |
| 高度活动 | 1.725 |
| 极高活动 | 1.9 |

目标热量调整：
- 减脂：TDEE - 300~500 kcal
- 增肌：TDEE + 200~400 kcal
- 维持：TDEE

### 16.2 免责声明（固定文案）

> 以上内容仅供参考，不构成医疗建议。如有健康问题请咨询专业医生。

### 16.3 参考链接

- 训记身体数据 Open API Skill
- 训记饮食数据 Open API Skill
- 训记训练数据 Open API Skill
- [Xunji 标准动作中文名表](https://github.com/Foveluy/Xunji-movements)

---

*文档结束*
