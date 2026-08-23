# MyFitness 意图数据集评估报告

- 生成时间：2026-08-23 19:39
- 样本数量：**100**
- 评估范围：Router 关键词分类 + Query Planner（不含 LLM 增强）
- 固定 today：`2026-08-23`

## 1. 总体指标

| 指标 | 准确率 | 说明 |
|------|--------|------|
| 意图分类 | **95.0%** (95/100) | 7+1 类意图 |
| 域识别 | **50.7%** (35/69) | body/nutrition/training |
| DB 查询触发 | **96.4%** (54/56) | 是否应查库 |
| 查询 Tool 匹配 | **60.9%** | expected_tools ⊆ predicted |

## 2. 分意图准确率

| 意图 | 样本数 | 意图准确 | 域准确 | DB触发准确 |
|------|--------|----------|--------|------------|
| 数据查询 | 22 | 86% | 10% | 91% |
| 手动录入 | 16 | 100% | 100% | — |
| 计划调整 | 10 | 80% | 80% | 100% |
| 趋势分析 | 14 | 100% | 0% | 100% |
| 目标设定 | 10 | 100% | 90% | 100% |
| 同步触发 | 8 | 100% | — | — |
| 通用对话 | 12 | 100% | — | — |
| 确认响应 | 8 | 100% | — | — |

## 3. 能力拆解

### 3.1 Router 层
- **关键词规则**：同步、录入、趋势、目标、查询等主路径
- **LLM 兜底**：未启用时（本次评估）歧义句落 general
- **确认续接**：依赖 pending_confirmation，无 pending 时「确认/取消」会误判

### 3.2 Query Planner 层
- **日期解析**：昨天/今天/前天/近N天/YYYY-MM-DD
- **域推断**：关键词 → body/nutrition/training；无关键词时 data_query 查三域
- **Tool 映射**：body→query_body_metrics, nutrition→query_nutrition_logs, training→query_training_logs

### 3.3 Agent 编排层
- data_query / trend_analysis → 三 Specialist + Summary
- manual_entry → 解析 + 确认 + 写库
- sync_trigger → run_sync
- plan_adjust / goal_setting → 部分能力（M4 待完善）

## 4. 能力评估（M2 现状）

| 能力项 | 评级 | 说明 |
|--------|------|------|
| 数据查询（单域） | ⭐⭐⭐⭐ | 蛋白/体重/训练等关键词路径稳定 |
| 趋势分析 | ⭐⭐⭐⭐ | 「近N天」「趋势」触发良好 |
| 手动录入 | ⭐⭐⭐⭐ | 体重/饮食录入识别准确 |
| 同步触发 | ⭐⭐⭐⭐⭐ | 关键词覆盖充分 |
| 域精细识别 | ⭐⭐⭐ | 缺 domain 时默认三域，开销大 |
| 计划调整 | ⭐⭐ | 仅识别意图，agent_plans CRUD 未实现 |
| 目标设定 | ⭐⭐ | 识别尚可，目标写入未闭环 |
| 多意图句 | ⭐ | 「同步然后分析趋势」未拆分 |
| 确认续接 | ⭐⭐⭐⭐ | 有 pending 时准确 |
| 训练明细解析 | ⭐⭐⭐⭐ | raw_payload 组次/重量已支持 |

## 5. 错误样本（意图分类失败）

| ID | 输入 | 期望 | 预测 |
|----|------|------|------|
| 7 | 午餐吃了什么 | data_query | manual_entry |
| 13 | 查询近7天吃了多少蛋白 | data_query | trend_analysis |
| 19 | 昨天有没有记录体重 | data_query | manual_entry |
| 44 | 今天训练取消改成恢复 | plan_adjust | data_query |
| 48 | 改成 active recovery 今天 | plan_adjust | data_query |

## 6. 改进建议（优先级）

1. **P0**：data_query 与 trend_analysis 优先级 — 「近7天蛋白」不应先命中 trend
2. **P0**：plan_adjust 中「取消训练」与 confirmation cancel 歧义
3. **P1**：domain 推断 — 无关键词时按 intent 默认单域而非三域
4. **P1**：goal_setting 写入 user_goals 闭环
5. **P2**：多意图拆分（sync + analysis）
6. **P2**：LLM Router 启用后复测 100 条
