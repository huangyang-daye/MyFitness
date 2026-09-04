# MyFitness 意图测评集（300 条）评估报告

- 生成时间：2026-09-02 15:56
- 样本数量：**300**（简单/中等/复杂 各 100）
- 评估模式：**关键词规则（use_llm=False）**
- 固定 today：`2026-08-23`

## 1. 总体指标

| 指标 | 准确率 |
|------|--------|
| 主意图（intents[0]） | **67.7%** (203/300) |
| 完整意图序列 | **57.7%** (173/300) |

## 2. 分难度

| 难度 | 样本 | 主意图准确 | 完整序列准确 |
|------|------|------------|--------------|
| 简单 | 100 | 92.0% (92/100) | 92.0% (92/100) |
| 中等 | 100 | 68.0% (68/100) | 51.0% (51/100) |
| 复杂 | 100 | 43.0% (43/100) | 30.0% (30/100) |

## 3. 分主意图（Top）

| 主意图 | 样本 | 主意图准确 | 完整序列准确 |
|--------|------|------------|--------------|
| trend_analysis | 54 | 74% | 70% |
| sync_trigger | 50 | 82% | 58% |
| manual_entry | 49 | 55% | 49% |
| data_query | 37 | 57% | 51% |
| schedule_manage | 22 | 95% | 95% |
| plan_adjust | 21 | 38% | 24% |
| web_search | 20 | 65% | 30% |
| report_trigger | 16 | 75% | 75% |
| goal_setting | 15 | 67% | 60% |
| chart_trigger | 12 | 83% | 83% |
| general | 4 | 0% | 0% |

## 4. 错误样本（完整意图序列不匹配，前 40 条）

| ID | 难度 | 输入 | 期望 | 预测 |
|----|------|------|------|------|
| 6 | simple | 午餐吃了什么 | ['data_query'] | ['general'] |
| 12 | simple | 昨天有没有记录体重 | ['data_query'] | ['manual_entry'] |
| 58 | simple | 我想增肌到78kg | ['goal_setting'] | ['general'] |
| 64 | simple | 从训记同步饮食 | ['sync_trigger'] | ['general'] |
| 72 | simple | 把近7天蛋白摄入画成图 | ['chart_trigger'] | ['trend_analysis'] |
| 77 | simple | 出一份8月22日的报告 | ['report_trigger'] | ['general'] |
| 83 | simple | 出一份完整健康报告 | ['report_trigger'] | ['general'] |
| 100 | simple | 把明天的训练改成有氧 | ['plan_adjust'] | ['general'] |
| 103 | medium | 拉取昨天数据并出报告 | ['sync_trigger', 'report_trigger'] | ['sync_trigger'] |
| 114 | medium | 出报告同时生成热量柱状图 | ['report_trigger', 'chart_trigger'] | ['chart_trigger'] |
| 115 | medium | 同步并生成报告，再附训练量折线图 | ['sync_trigger', 'report_trigger', 'chart_trigger'] | ['report_trigger', 'chart_trigger'] |
| 116 | medium | 拉取数据、生成日报、画体重图 | ['sync_trigger', 'report_trigger', 'chart_trigger'] | ['sync_trigger', 'report_trigger'] |
| 123 | medium | 录入今天体脂后分析近30天变化 | ['manual_entry', 'trend_analysis'] | ['trend_analysis'] |
| 124 | medium | 记录午餐后看看今天蛋白够不够 | ['manual_entry', 'data_query'] | ['manual_entry'] |
| 125 | medium | 我昨天蛋白质摄入和推荐量差多少 | ['data_query', 'web_search'] | ['data_query'] |
| 126 | medium | 查昨天训练记录并搜一下背部训练动作推荐 | ['data_query', 'web_search'] | ['web_search'] |
| 127 | medium | 对比近7天摄入并查减脂期蛋白建议 | ['trend_analysis', 'web_search'] | ['trend_analysis'] |
| 135 | medium | 根据我过往训练记录安排今天练背 | ['trend_analysis'] | ['data_query'] |
| 136 | medium | 结合历史数据给我个性化饮食建议 | ['trend_analysis'] | ['general'] |
| 137 | medium | 今天练什么，参考我最近的腿部训练 | ['trend_analysis'] | ['data_query'] |
| 138 | medium | 查询今天体重并分析近7天趋势 | ['data_query', 'trend_analysis'] | ['trend_analysis'] |
| 139 | medium | 昨天吃了多少蛋白，顺便看近一周变化 | ['data_query', 'trend_analysis'] | ['trend_analysis'] |
| 140 | medium | 同步后分析近7天体重变化 | ['sync_trigger', 'trend_analysis'] | ['trend_analysis'] |
| 142 | medium | 设置每晚21点同步最近3天数据 | ['schedule_manage'] | ['sync_trigger'] |
| 144 | medium | 今天不练了，顺便查一下昨天训练量 | ['plan_adjust', 'data_query'] | ['data_query'] |
| 145 | medium | 取消今天训练，分析本周训练频率 | ['plan_adjust', 'trend_analysis'] | ['plan_adjust'] |
| 146 | medium | 调整计划：明天练背，参考上次背部训练 | ['plan_adjust', 'trend_analysis'] | ['plan_adjust'] |
| 147 | medium | 目标降到70kg，分析当前进度 | ['goal_setting', 'trend_analysis'] | ['goal_setting'] |
| 148 | medium | 设定体脂15%并看近30天变化 | ['goal_setting', 'trend_analysis'] | ['trend_analysis'] |
| 154 | medium | 搜一下减脂期每天蛋白吃多少，并对照我昨天摄入 | ['web_search', 'data_query'] | ['web_search'] |
| 155 | medium | 联网查HIIT训练频率，再结合我上周训练记录给建议 | ['web_search', 'trend_analysis'] | ['web_search'] |
| 163 | medium | 同步训记后出一份完整健康报告 | ['sync_trigger', 'report_trigger'] | ['sync_trigger'] |
| 164 | medium | 更新数据、分析趋势、生成报告 | ['sync_trigger', 'trend_analysis', 'report_trigger'] | ['sync_trigger', 'report_trigger'] |
| 165 | medium | 录入体重后同步训记再分析趋势 | ['manual_entry', 'sync_trigger', 'trend_analysis'] | ['trend_analysis'] |
| 166 | medium | 拉取近7天数据，画体重图，写进日报 | ['sync_trigger', 'chart_trigger', 'report_trigger'] | ['sync_trigger', 'report_trigger'] |
| 167 | medium | 设定目标68kg，生成饮食规划文档 | ['goal_setting', 'trend_analysis'] | ['trend_analysis'] |
| 168 | medium | 记录体脂并生成近30天体脂报告 | ['manual_entry', 'trend_analysis'] | ['report_trigger'] |
| 169 | medium | 查今天热量并搜减脂热量缺口建议 | ['data_query', 'web_search'] | ['data_query'] |
| 170 | medium | 分析近7天训练并搜背部恢复方法 | ['trend_analysis', 'web_search'] | ['trend_analysis'] |
| 172 | medium | 同步、出报告、插入折线图一条龙 | ['sync_trigger', 'report_trigger', 'chart_trigger'] | ['chart_trigger'] |
| … | 共 127 条 | | | |