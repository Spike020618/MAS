# AEIC-MD 实验执行总计划

> 状态：实验阶段总控文档（2026-08-02）  
> 终点：完成全部数据采集、统计检验、稳健性复验和结果冻结；之后只剩绘图与论文写作。

## 1. 冻结后的研究主线

### 1.1 总问题

在多智能体存在有限理性、努力成本、从众或策略性报告时，如何利用 AEIC 作为可观测报告接口，通过激励、反馈和信息结构的联合机制设计，获得可靠、低成本且保持信息多样性的集体决策？

### 1.2 方法定位

- **AEIC**：可观测报告空间。用于测量证据增量、无依据立场翻转、复制、推理一致性与报告完整度。
- **机制策略**：Leader 公布信用/采纳规则、反馈规则、验证规则与通信规则。
- **Follower 行为**：Agent 在看到规则后实际选择努力、报告、模仿、坚持或修正，而非根据生成后信号事后划分“策略”。
- **Stackelberg**：动态机制的核心实现之一；只有存在可观测 follower response 时才使用 Stackelberg/IC 表述。
- **拓扑**：机制控制的信息结构，不作为独立主线。
- **评审场景**：自然群体病理和策略性行为的验证域。

暂定方法名：**AEIC-MD (AEIC-mediated Adaptive Mechanism for Multi-Agent Deliberation)**。

## 2. 已有资产的处理决定

| 资产 | 决定 | 在新叙事中的作用 |
|---|---|---|
| QA parallel/sequential/star/debate 缓存 | 复用，不重采 | 证明信息结构影响输出；作为旧静态聚合基线 |
| AEIC+SG 离线权重优化 | 保留但降级 | 静态聚合基线，不再作为真实 Stackelberg response 的证据 |
| 在线 static/dynamic feedback，3 数据集 × 50 | 复用作预实验 | 证明反馈可能改变结果；因缺少完整跨轮 AEIC 与行动日志，不作为正式机制实验 |
| ICLR 评审 Study 1 | 完整复用 | 自然病理诊断：偏置、负向棘轮、方差坍缩 |
| 软校准 Study 2a | 完整复用 | 非策略、稳定偏置环境下的最强简约基线和机制边界 |
| 话术调控 Study 2b | 完整复用 | prompt 提醒不足以构成有效机制的负结果 |
| 路由 Study 2c | 完整复用 | 成本执行层；不宣称决策质量提升 |
| 异质面板与旧高维 θ* | 留档 | 已证伪路线，不再投入实验预算 |

## 3. 必须回答的研究问题与假设

### RQ1：AEIC 是否能测量行为，而不只是格式化输出？

- **H1a**：AEIC 可区分“新证据导致的修正”和“无新证据的从众翻转”。
- **H1b**：结构化行为指标对已知行为操纵强度呈单调响应。
- **验收**：操纵检查效应方向一致；核心指标 AUROC ≥ 0.70 或标准化效应量 |d| ≥ 0.5；跨数据集至少 2/3 成立。

### RQ2：机制规则是否产生真实 follower response？

- **H2a**：局部可验证信用相对无奖励/仅结果奖励，提高高努力和独特可验证证据的选择率。
- **H2b**：对抄袭、无依据翻转和低努力报告的收益优势下降。
- **验收**：高努力选择率显著提高；行为效应通过配对/分层模型显著；同时报告 token 与调用成本。

### RQ3：机制响应是否改善系统效用？

- **H3a**：在策略性/有成本环境中，AEIC-MD 相对多数票、软校准、静态 AEIC 聚合和无引导 debate 提高综合效用。
- **H3b**：在诚实固定偏置环境中，AEIC-MD 不必胜过软校准，但不得显著劣化决策质量。
- **验收**：主要策略环境中综合效用显著提高；决策准确率或一致率不低于最强基线的预设非劣界限 2pp。

### RQ4：拓扑何时有价值？

- **H4**：通信拓扑的收益由当前分歧、证据互补性和策略风险共同决定；自适应信息结构可在质量非劣前提下降低成本。
- **验收**：相对最优固定协议成本下降 ≥ 30%，质量非劣界限 2pp；如质量提升则作为次要结论。

### RQ5：机制是否可泛化？

- **H5**：机制在未参与调参的数据集、模型或激励强度上仍保持行为改善方向。
- **验收**：leave-one-domain-out 和至少一个 held-out backend 上方向一致；报告最坏组性能。

## 4. 新实验体系

### E0：资产审计与旧结论复算

目的：冻结现有结果的样本量、数据血缘、代码入口和可复现状态。

输出：

- `results/audit/asset_inventory.json`
- `results/audit/result_consistency.json`
- `results/audit/data_lineage.md`

必须检查：重复样本、不同 JSON 口径（200/300）、训练测试泄漏、缺失/错误 API 条目、同题配对关系、随机种子和统计口径。

### E1：行为测量校准实验

任务：StrategyQA、MMLU-Hard、TruthfulQA，各 50 题 pilot；正式阶段各 150 题。

每题 5 Agent、3 轮。构造可控行为条件：

1. `independent`：只能独立生成，作为基准；
2. `evidence_update`：向 Agent 提供一条可验证的新证据；
3. `social_only`：只提供多数立场，不提供新证据；
4. `copy_pressure`：提示与多数一致可提高采纳机会；
5. `counterevidence`：提供针对其 E 项的具体反证。

保存每轮完整 AEIC、可见信息、时间、token、模型、温度、prompt 版本及立场。

主要指标：

- unsupported flip rate；
- evidence-conditioned flip rate；
- evidence novelty / copying；
- AEIC completeness；
- inference–conclusion continuity；
- belief change calibration。

### E2：真实 follower-response 机制实验

在调用前向 Agent 公布机制，并允许实际行动选择：

- `low_effort`：短报告，成本低；
- `aeic_effort`：完整 AEIC，成本中等；
- `verify`：追加核验/反事实步骤，成本高；
- `defer/imitate`：参考他人后报告，成本低但独立信息少。

条件：

1. **NoMechanism**：无信用规则；
2. **OutcomeOnly**：只按最终是否正确/一致分配信用；
3. **StaticAEIC**：固定 AEIC 质量权重；
4. **AEIC-Contract**：奖励独特、可验证且真正影响决策的局部报告；
5. **Adaptive/Stackelberg**：Leader 根据历史行为调整奖励、反馈和验证机会。

Follower 收益必须在 prompt 中可见，并在多题序列中真实累计为声誉、下一题影响权或验证预算。不能仅在生成结束后计算虚拟 utility。

正式比较以 4 vs 1/2/3 为机制有效性，5 vs 4 为动态机制增益。

### E3：策略强度与失效边界

操纵三个更自然的轴：

- 努力成本：低/中/高；
- 采纳奖励：低/中/高；
- 社会信息强度：隐藏立场/只见多数/见完整报告。

形成响应曲线而不是单一“诚实/策略”二分类。阵营 prompt 只作为压力测试，不作为唯一策略性证据。

输出：校准充分区、静态机制区、动态机制必要区的 phase diagram 数据表。

### E4：信息结构与成本控制

固定同一机制，只改变可选动作：parallel、targeted-review、debate、star、stop。

对比：

- 最优固定协议；
- 规则路由；
- 学习路由；
- oracle 上界。

路由输入仅使用决策时可见变量：首轮分歧、AEIC 冲突位置、证据新颖性、历史可靠性和预算。禁止使用 ground truth、官方 stratum 或事后收益特征。

### E5：评审域策略性验证

先运行 50 篇 pilot，只选择能产生稳定且非饱和畸变的条件进入正式实验。

正式条件：

- E0：现有诚实评审；
- E1：公开署名/问责；
- E2：采纳或决议对齐奖励；
- E3：历史声誉影响后续权重；
- E4：AEIC-Contract 修复；
- E5：Adaptive mechanism 修复。

阵营分配可保留为 adversarial stress test。正式结论必须同时包含至少一个自然的跨任务声誉或预算激励。

基线：多数票、固定 5.5、软校准、静态 θ*、无引导 debate。

终点：官方决议一致率只称 agreement；行为健康和 truthfulness 为主要终点；决策一致率为安全终点。

### E6：泛化、消融与稳健性

- 跨数据集：leave-one-dataset-out；
- 跨模型：至少 DeepSeek-V3 与一个 held-out backend；
- 跨激励强度：训练中档，测试低/高档；
- 消融：去 A/E/I/C 局部信号、去历史、去动态调节、固定拓扑；
- 负对照：随机奖励、打乱 agent 历史、不可见机制；
- 稳健性：不同 seed、不同阈值、缺失 Agent、错误多数、成本权重敏感性。

## 5. Pilot 决策门

Pilot 不是为了挑最好看的配置，而是验证实验是否可识别。

只有同时满足以下条件才进入正式采集：

1. 行动选择不是饱和的：任一动作比例不得长期低于 10% 或高于 90%；
2. 激励强度与至少一个目标行为指标呈预期单调关系；
3. AEIC JSON 有效解析率 ≥ 95%；
4. 各条件 API 失败率 < 5%；
5. 单题成本、延迟和 token 日志完整率 ≥ 99%；
6. 不查看正式 test 集调参。

若失败，最多允许两轮 pilot 修订。所有 pilot 版本和失败配置留档，不能覆盖。

## 6. 数据划分与统计规范

- 每个域固定 `train/dev/test`，机制优化只看 train，pilot 只用 dev，最终只报告一次 test。
- 同一道题在不同条件下使用相同 agent seed/模型分配，形成配对设计。
- 二元决策：McNemar 或分层 mixed-effects logistic model。
- 连续行为指标：配对 Wilcoxon；多条件使用 mixed-effects model，并对题目和 Agent 设置随机效应。
- 比率：Wilson 95% CI；多重检验使用 Holm 校正。
- 同时报效应量、置信区间和原始计数，不以 `p < 0.05` 代替实际效应。
- 主要终点、次要终点、非劣界限和排除规则在正式运行前写入 manifest。
- 所有结果保留 question-level prediction，禁止只保存汇总准确率。

## 7. 工程要求

每个新实验必须具备：

- 原子写入；
- question/condition/agent 三级断点续跑；
- 失败条目可重试且不计为完成；
- prompt 与机制配置哈希；
- 完整输入输出和 token/cost/latency 日志；
- 独立的 `collect`、`validate`、`analyze` 命令；
- dry-run 与最小 smoke test；
- 固定 schema 版本；
- 自动生成数据完整性报告。

建议新增结构：

```text
core/mechanism/
  schema.py
  contracts.py
  controller.py
  behavioral_metrics.py
experiments/mechanism/
  manifest.json
  collect.py
  validate.py
  analyze.py
results/mechanism/
  pilot/
  formal/
  frozen/
```

## 8. 执行顺序

| 阶段 | 工作 | 进入下一阶段的条件 |
|---|---|---|
| P0 | 资产审计、数据血缘、旧结果复算 | 所有旧结论可定位到原始 question-level 数据 |
| P1 | 冻结 schema、行为指标、机制动作和统计 manifest | dry-run、单元测试、无数据泄漏检查通过 |
| P2 | E1+E2 pilot | 满足 §5 六项决策门 |
| P3 | E1+E2 正式 QA 实验 | 三域数据完整、主检验完成 |
| P4 | E3 激励强度边界 + E4 信息结构 | phase diagram 和成本非劣检验完成 |
| P5 | E5 评审 pilot 与正式实验 | 至少一个自然策略条件完成 |
| P6 | E6 泛化、消融、稳健性 | 预注册矩阵无缺项 |
| P7 | 结果冻结 | 所有表格由 frozen question-level 数据一键复算 |

## 9. “全部实验完成”的定义

必须同时满足：

- E0–E6 中所有标记为正式的实验均完成或依据预先规则明确停止；
- 所有主要假设都有正结果、负结果或不可识别结论，不选择性丢弃；
- test 集只进行一次正式解封分析；
- 数据完整性、配对关系、失败率和成本审计通过；
- 主结果至少完成一次独立重跑或 held-out 复验；
- `results/mechanism/frozen/` 包含不可变原始数据、预测、统计表和配置哈希；
- 从 frozen 数据运行统一分析命令可以重建论文所需全部数字。

完成上述条件后，项目实验阶段结束，后续仅进行绘图和论文写作。

## 10. 当前首个工程里程碑

先完成 P0/P1，不立即消耗 API：

1. 自动资产审计；
2. 统一数据 schema；
3. 写正式 manifest；
4. 实现完整跨轮日志；
5. 实现四类 follower 行动与三类信用规则；
6. 本地 mock/smoke 验证后再启动 50 题 pilot。

