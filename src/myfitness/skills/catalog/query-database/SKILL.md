---
name: query-database
description: >
  查询数据库中的身体指标、饮食与训练记录。用户询问体重、体脂、围度、
  热量、蛋白质、碳水、脂肪、训练动作/容量，或需要近 N 天趋势、减肥进度、
  个性化饮食/训练建议时使用。
handler: myfitness.skills.handlers.query_database:run
when: context
triggers:
  intents:
    - data_query
    - trend_analysis
    - goal_setting
    - plan_adjust
    - general
  keywords:
    - 体重
    - 体脂
    - 饮食
    - 训练
    - 热量
    - 蛋白
---

# 查询本地健身数据库

从用户消息解析日期范围与数据域，读取 PostgreSQL 中已同步/手动录入的记录。
**不要**用本 Skill 直连训记 Open API；训记数据应先同步入库。

## 何时执行

- 意图为 `data_query` / `trend_analysis` / `goal_setting` / `plan_adjust`
- `general` 但消息含身体/饮食/训练关键词，或需要个性化建议
- Orchestrator 已给出 `QueryPlan`（例如补齐「最新体重」「近 30 天训练史」）

寒暄、纯动作指令（同步/出图/定时任务）且无数据关键词时跳过。

## 流程

1. 用 `build_query_plan` 解析日期、域（body / nutrition / training）、metric_type、meal_type。
2. 趋势/进度类且涉及身体数据时，把起点扩展到该用户最早一条身体记录。
3. 调用 `execute_query_plan`（内部是 `query_body_metrics` / `query_nutrition_logs` / `query_training_logs`）。
4. 把各域结果写入 `SkillResult.data`，底层 tool 名写入 `tools_invoked`。

## 日期约定

- 「最近 N 天」含今天
- 未写日期：默认近 7 天；趋势分析默认近 30 天
- 「今天练 x + 参考过往记录」：训练史查近 30 天，不要缩成单日

## 输出

`data` 按域键名：

- `body`：`records` + 可选 `latest_metrics.weight/bodyfat`
- `nutrition`：`entries` + `daily_totals`
- `training`：`sessions`（含动作与组数）；可按 `muscle_group` 过滤

`extra.plan` 为实际使用的 `QueryPlan`（含拓宽后的日期）。

## 即插即用

同名文件夹放到项目 `skills/query-database/` 可覆盖本内置 Skill。其它自定义查询 Skill 同样：`skills/<name>/SKILL.md` + `handler.py`（实现 `run(ctx)`，可选 `should_run(ctx)`）。
