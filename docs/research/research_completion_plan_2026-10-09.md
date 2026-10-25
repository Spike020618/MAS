# 研究工作补全与证据链收口计划

> 执行计划更新（2026-10-20）：本文保留为历史计划与成果登记，后续研究工作的阶段顺序、成功/失败分支及停止规则统一以[长期执行计划](long_term_execution_plan_2026-10-20.md)为准。历史实验协议不因此改写。

> 版本：2026-10-09  
> 研究题目：博弈驱动的去中心化语义共识机制及在影视推荐中的应用  
> 研究结构：一个机制创新点 + 一个系统实现  
> 总原则：不新增研究方向；先复算现有数据，再围绕机制识别、博弈响应、留出验证和应用有效性开展新的正式实验。新增实验必须预先登记问题、比较、统计和停止规则。

## 0. 执行状态（2026-10-14）

| 工作包 | 状态 | 已完成证据 | 下一步 |
|---|---|---|---|
| WP1 主张与统计口径 | 基本完成 | claim-evidence registry、主次终点与禁止表述 | 补机器可读 manifest |
| WP2 机制传导链 | 已完成首轮正式分析 | 行动、资源、三轮轨迹、题目级错误流向 | 将核心结果写入中英文论文 |
| WP3 成本收益 | 已完成首轮正式分析 | 10,000次问题级配对Bootstrap、分域成本收益 | 增加调用—准确率二维附图 |
| WP4 报告接口 | 已完成90题×3条件 | AEIC相对自由文本改善，但未证明优于通用结构 | 不扩大“AEIC独立提准”主张 |
| WP4 机制拆解 | 已完成60题×4条件 | 匹配算力未复现合同的行动响应 | 视预算决定是否扩大质量终点样本 |
| WP5 人工盲审 | 未开始 | 已确定审核对象和指标 | 抽样、双人盲审、一致性统计 |
| WP6 既有扩展实验综合 | 数据已完成、叙事待整合 | 奖励/成本、信息环境、Seed、Panel、信号、历史、跨模型 | 形成一张稳健性总表和边界图 |
| WP7 影视系统评测 | 工程与显式请求评测完成 | 30类请求、150条Top-5：实体、约束、证据与Trace均100%；目录检索对照已完成 | 人工审核；网络恢复后补ReDial质量基准 |

目前正式实验不是只有主实验和两个补充实验。冻结资产仍包含8组、5850条问题级记录；加上报告接口270条与机制拆解240条，已形成6360条正式问题级记录。后续重点是综合分析、人工盲审和影视应用评测，不重复采集已经充分的稳健性实验。

## 1. 最终研究问题

本研究只回答一个核心问题：

> 在多智能体审议中，能否通过行动前公开的过程规则，使 Agent 主动选择可核验的高资源行动，并以可量化的计算成本改善集体决策？

完整证据链固定为：

```text
最终聚合无法区分独立推理、验证和廉价模仿
                    ↓
Leader 在行动前提交公开合同 m=(g,c,P,B)
                    ↓
Follower 从 Low / AEIC / Verify / Imitate 中选择行动
                    ↓
行动触发不同的真实 Prompt、Token、调用与验证路径
                    ↓
形成可解析的 AEIC 报告与可观察过程信号
                    ↓
信用—成本结算、信誉更新与确定性集体决策
                    ↓
行为、信息产物、集体正确率和计算成本发生可测变化
```

“博弈驱动”的证据是承诺—响应时序、奖励/成本剂量响应和Follower行为变化，不要求声称已经求得闭式Stackelberg均衡。“去中心化”限定为报告和决策层的逻辑去中心化，不扩展为跨机器P2P部署。

## 2. 最终主张及证据等级

### C1 核心行为主张

行动前公开、并通过AEIC观测接口表达的过程合同能够显著改变Agent的资源行动选择。AEIC在该主张中是报告与结算接口，合同及其执行绑定才是处理变量。

- 主证据：No Mechanism、Outcome Only、Static AEIC的配对比较；
- 关键指标：高资源行动率及Low/AEIC/Verify/Imitate完整分布；
- 已有结果：0.46% → 56.93%；Outcome Only仅为3.08%；
- 机制支持：奖励提高时高资源行动增加，成本提高时减少。

### C2 核心结果主张

在现有联合执行配置和三个QA任务域中，行为变化伴随集体决策改善；该结果尚不能单独归因于AEIC格式、合同文字、Token增加或后端升级中的任一因素。

- 主证据：450道相同问题的问题级配对；
- 已有结果：68.00% → 82.22%，增加14.22个百分点；
- 题目流向：78道错→对，14道对→错；
- 统计：Exact McNemar + Holm校正，报告效应量、原始计数和95% CI。

### C3 成本与机制传导主张

准确率提升不是免费收益，而是由行动、资源执行、信息产物和纠错共同构成的可审计链路。

- 需要补齐：行动→资源→报告→纠错→最终结果的统一分析；
- 需要补齐：等计算量对照，以排除“只因使用更多Token”的解释；
- 需要补齐：准确率—Token—调用次数—延迟的成本收益展示。

### C4 稳健性与边界主张

主方向在额外Seed、不同Panel规模和已测信息环境中保持，但同一合同不自动跨模型迁移。

- 支持证据：Seed、3—5 Agent、信息环境、信号和历史扰动；
- 边界证据：GLM后端不响应同一合同；
- 表述边界：只声称“已测试主后端上的经验有效性”。

## 3. 当前证据资产

现有冻结目录包含8组实验、5850条问题级记录，全部具有验证、provenance和独立复算结果：

| 证据组 | 记录数 | 研究作用 | 最终位置 |
|---|---:|---|---|
| formal_v10 | 1350 | 主实验：行为与准确率 | 正文核心 |
| boundary_v11_dev | 600 | 奖励/成本剂量响应 | 正文机制证据 |
| information_v11_dev | 1200 | 不同同伴信息环境 | 正文次要/附录 |
| seed_robustness_v11_dev | 600 | 随机稳健性 | 正文简表 |
| missing_agent_v11_dev | 450 | Panel规模与缺失Agent | 附录 |
| signal_ablation_v11_dev | 750 | 信号扰动边界 | 附录 |
| history_ablation_v11_dev | 600 | 历史策略与负对照 | 附录 |
| heldout_glm_v11_dev | 300 | 跨模型不迁移边界 | 正文讨论 |

上述资产不重新采集。所有新增分析必须从`experiments/results/mechanism/frozen/`中的question-level数据生成。

## 4. 工作包WP1：冻结论文主张与统计口径

### 目标

先固定“要证明什么”，防止分析过程中根据结果修改问题。

### 工作

1. 建立claim-evidence registry，每个结论记录：主张、主要终点、数据源、统计方法、允许表述和禁止表述。
2. 固定主要比较族：Static AEIC vs No Mechanism；Static AEIC vs Outcome Only。
3. 固定主要终点：高资源行动率、第三轮集体准确率。
4. 固定次要终点：Token、调用次数、延迟、AEIC完整率、证据与纠错指标。
5. 固定分析单位：准确率以问题为单位；行为指标以Agent-round记录，但置信区间按问题聚类。
6. 固定排除规则、缺失值处理和多重校正范围。

### 产出

```text
docs/research/claim_evidence_registry.md
experiments/mechanism/analysis_manifest_v13.json
```

### 验收

- 每一条摘要和结论中的数字都能定位到冻结文件和分析脚本；
- 不使用Agent-round数量虚增准确率样本量；
- 主分析和探索性分析明确分开。

## 5. 工作包WP2：从现有数据补齐机制传导链

### 目标

回答“行动率和准确率为什么会同时变化”，避免两个结果并列但缺少中间联系。

### 分析2.1：完整行动分布

按条件、数据集和轮次报告Low、AEIC、Verify、Imitate的占比及问题聚类95% CI，不只报告合并后的高资源行动率。

### 分析2.2：真实资源执行

按行动和条件报告：

- input/output/total Token；
- 模型调用次数与验证调用率；
- 平均延迟与P95；
- API重试、解析修复和失败率。

检查“声明行动”和“实际路由”是否一致，并给出一致率。

### 分析2.3：报告与证据信号

按条件报告：

- complete_aeic；
- consistency；
- field_A/E/I/C；
- novel_evidence；
- verified_evidence；
- useful_correction；
- unsupported_flip。

这些指标用于描述传导过程，不把相关关系表述为独立因果中介效应。

### 分析2.4：逐轮共识轨迹

对Round 1/2/3计算：

- Agent个体准确率和多数决策准确率；
- 支持分布、投票熵或分歧率；
- 错误多数被纠正的数量；
- 正确多数被破坏的数量；
- 正确少数观点最终被采纳的比例；
- 立场变化中有新证据与无新证据的比例。

### 分析2.5：题目级结果流向

制作No Mechanism → Static AEIC的四格流向：

| No Mechanism | Static AEIC | 含义 |
|---|---|---|
| 错 | 对 | 有效纠正 |
| 对 | 错 | 机制损伤 |
| 对 | 对 | 稳定正确 |
| 错 | 错 | 未解决 |

进一步按任务域、初始分歧和高资源行动变化分层，但不把事后分层写成随机化中介证明。

### 产出

```text
experiments/mechanism/analyze_transmission_chain.py
experiments/results/mechanism/synthesis/transmission_chain.json
experiments/results/mechanism/synthesis/round_trajectory.json
experiments/results/mechanism/synthesis/error_flow.json
docs/papers/en/figures/fig_transmission_chain.pdf
docs/papers/en/figures/fig_round_trajectory.pdf
```

### 验收

- 所有结果可以从冻结question-level文件一键复算；
- 每个图同时保留原始计数、效应量和不确定性；
- 不新增API调用。

## 6. 工作包WP3：成本收益和不确定性

### 目标

正面回答“多花的计算是否值得”，不把资源消耗隐藏在准确率后面。

### 工作

1. 对准确率差、高资源行动率差和主要过程指标计算问题级cluster bootstrap 95% CI。
2. 报告每Agent-round增加的Token、调用次数和延迟。
3. 报告每净纠正一道题增加的Token和调用次数。
4. 绘制Accuracy—Token、Accuracy—Calls二维图；不强行合成单一效用分数。
5. 分别计算三个任务域的成本与收益，避免用总体均值掩盖异质性。
6. 对成本权重只做敏感性分析，不挑选使本方法最好看的权重。

### 产出

```text
experiments/mechanism/analyze_cost_effectiveness.py
experiments/results/mechanism/synthesis/cost_effectiveness.json
docs/papers/en/figures/fig_cost_effectiveness.pdf
```

### 验收

- 主图可以同时看到质量提升和资源代价；
- 论文明确写出Static AEIC比No Mechanism每Agent-round多145.5 Token；
- 不使用“零成本提升”“效率全面占优”等表述。

## 7. 工作包WP4：第一优先新增实验——AEIC接口验证与等计算量分解

### 7.1 现有证据边界

主实验中Static AEIC使用808.8 Token/Agent-round，No Mechanism为663.3。现有Outcome Only和`fixed_backend_control_v12`均不能完全排除“准确率只因更多计算而提高”：

- Outcome Only的计算量仍低于Static AEIC；
- fixed_backend_control_v12固定模型后端且`resource_tiered=false`，回答的是后端混杂，不是等计算量问题。

此外，现有三个主条件都输出AEIC结构，因此没有直接比较自由文本与AEIC。字段遮蔽结果也未发现A、E、I或C中任一字段的独立必要性。新增实验必须把“报告接口价值”和“合同/资源效果”分开。

### 7.2 阶段一：报告接口验证

同一DeepSeek模型、相同问题、相同轮次、相同调用次数和相同max-token预算下比较：

| 条件 | 输出协议 | 过程合同 | 研究问题 |
|---|---|---|---|
| Free Text | 自由文本推理与结论 | 无 | 非结构化基准 |
| Generic Structure | 通用“分析—答案”结构 | 无 | 结构化本身的效果 |
| AEIC Interface | A/E/I/C四字段 | 无 | AEIC语义分解的增量价值 |

主要终点不是只看准确率，而是：自动解析成功率、立场提取准确率、证据边界清晰度、有依据/无依据翻转识别率、复制识别率、与人工审核的一致率。准确率和Token为共同报告的次要终点。若AEIC不提高准确率但显著提高测量可靠性，则其贡献被表述为“可观察语义接口”。

### 7.3 阶段二：合同与资源成分分解

固定使用AEIC接口，比较：

| 条件 | 公开合同 | 行动选择 | 资源随行动变化 | 作用 |
|---|---:|---:|---:|---|
| AEIC No Contract | 否 | 固定/无策略选择 | 否 | AEIC接口基准 |
| Contract Fixed Resource | 是 | 是 | 否 | 合同文字与声明行动 |
| Compute Matched No Contract | 否 | 否 | 固定为相同预算 | 更多计算本身 |
| Full Mechanism | 是 | 是 | 是 | 合同—行动—资源完整链路 |

所有条件使用同一模型端点。预算匹配在运行前设置相同调用上限和max token，并同时报告实际消耗，不能事后筛选Token相近样本。

### 7.4 分阶段规模

#### Stage A：Smoke

- 每域5题；
- 检查Prompt、路由、预算和日志；
- 不用于结论。

#### Stage B：Pilot

- 报告接口实验每域50题、3条件，共450个问题—条件单元；
- 成分分解每域50题、4条件，共600个问题—条件单元；
- 使用3或5 Agent必须在运行前固定；推荐保持与主实验一致的5 Agent、3轮；
- 主要终点：第三轮集体准确率；
- 次要终点：高资源行动、完整AEIC、Token、调用和延迟。

#### Stage C：扩展规则

只有满足以下任一条件才扩展到每域150题：

1. 关键预设比较方向稳定，但95% CI过宽；
2. 三个域中至少两个方向一致，且数据完整率达到100%；
3. 估计新增样本能够实质缩窄CI，而不是只追求显著性。

只有阶段一证明AEIC至少提高测量可靠性，且阶段二能够区分合同或资源效应，才进入奖励×成本完整网格。否则停止扩展并收缩结论。

### 7.5 统计

- 准确率：配对差、Exact McNemar、配对bootstrap 95% CI；
- 行为率：按问题聚类bootstrap；
- 报告接口预设比较为AEIC vs Free Text、AEIC vs Generic Structure；
- 机制分解预设比较为Contract Fixed Resource vs AEIC No Contract、Full Mechanism vs Compute Matched No Contract；
- 各比较族内使用Holm校正；
- 不把Agent-round当作独立准确率样本；
- 正式运行前冻结矩阵和Prompt哈希。

### 7.6 产出

```text
experiments/mechanism/matrices/report_interface_v13.json
experiments/mechanism/matrices/mechanism_decomposition_v13.json
experiments/results/mechanism/frozen/report_interface_v13/
experiments/results/mechanism/frozen/mechanism_decomposition_v13/
docs/research/report_interface_and_decomposition_protocol.md
```

## 8. 工作包WP5：奖励×成本完整响应面与经验合同选择

### 目标

增强“博弈驱动”的研究表达，但不声称求得理论均衡。

### 工作

1. WP4通过进入门后，先使用boundary_v11_dev的单因素数据验证方向，再新跑奖励×成本的3×3完整网格。奖励取0.05/0.2/0.5，成本倍率取0.5/1.0/1.5；三个任务域各50道开发题，共1350个问题—条件单元。
2. 所有网格条件固定DeepSeek模型、5 Agent、3轮、信息策略、Prompt骨架和聚合方式，只有合同参数变化。
3. 拟合Follower经验响应面：

```text
Pr(high_resource) = logistic(β0 + βr·reward + βc·cost
                             + domain interaction)
```

4. 模型加入reward×cost交互项，报告βr、βc、交互项、置信区间和按域异质性；原始9格均值与CI必须与模型结果同时展示。
5. 预先定义Leader效用族，而不是根据结果挑权重：

```text
J_λ(m) = Accuracy(m) - λ_token·NormalizedToken(m)
                    - λ_call·NormalizedCalls(m)
```

6. 至少报告“质量优先、平衡、成本优先”三组预设λ，并绘制有限合同Pareto前沿。
7. 在开发网格上为“平衡”权重选择一个经验合同m*；合同选择完成后锁定参数。
8. 在从未用于选择合同的正式test split上比较m*、Static AEIC中心合同和No Mechanism，每域150题。
9. 只称“有限合同空间中的经验选择及留出验证”，不称闭式均衡、全局最优或一般激励相容。

### 产出

```text
experiments/mechanism/estimate_follower_response.py
experiments/results/mechanism/synthesis/follower_response.json
docs/papers/en/figures/fig_follower_response_surface.pdf
experiments/mechanism/matrices/reward_cost_grid_v13_dev.json
experiments/mechanism/matrices/selected_contract_v13_test.json
```

### 验收

- 奖励系数和成本系数的方向与原始曲线一致；
- 9格数据完整，交互作用是否存在均如实报告；
- 模型预测不替代原始分组结果；
- m*完全由开发集选择，并只在留出test上评估一次；
- 如果m*不优于中心合同，则保留负结果并停止进一步调参。

## 9. 工作包WP6：重复博弈与信誉验证

### 9.1 重复任务与信誉状态

当前主实验验证固定公开合同下的Follower response，但信誉更新的长期作用还不够直观。新增一个小规模顺序实验：

- 每个Panel连续处理30道题，Agent标识保持不变；
- 三个条件：reputation reset、persistent reputation、shuffled reputation；
- 三个任务域各10个Panel序列；
- 每个条件使用相同题序、初始状态和模型配置；
- 主要指标：后10题准确率、高资源行动率、错误Agent权重下降、正确少数观点保留率；
- 次要指标：信誉集中度、恢复速度和历史误差传播。

如果persistent reputation不优于reset，结论应为“当前信誉更新可记录但未证明带来长期性能收益”，不继续增加历史机制复杂度。

### 9.2 产出

```text
experiments/mechanism/matrices/repeated_reputation_v13.json
experiments/results/mechanism/frozen/repeated_reputation_v13/
```

## 10. 工作包WP7：影视应用验证收口

影视系统是毕业设计的重要组成部分，但不是理论论文的第二创新点。

### 已有证据

- 约18,430部MovieLens影片目录；
- 30条多约束请求、150个Top-5结果的工程验收；
- 请求规划、真实目录检索、冻结候选、多Agent AEIC、硬约束门控、共识排序和Trace；
- 同一次真实运行的页面与审计轨迹。

### 最小补全

1. 冻结一套100条影视请求基准，明确硬约束、软偏好和期望相关影片。
2. 比较三个系统条件：字段/BM25检索、单DeepSeek重排、多Agent共识系统。
3. 指标：Hit@5、NDCG@5、硬约束满足率、目录外实体率、证据支持率、Token、调用次数、平均/P95延迟。
4. 人工只审核模糊映射和推荐理由事实性；不要求人工制作完整排序。
5. 系统结论限定为“机制落地、约束可靠和过程可追溯”；只有正式评测完成后才比较推荐质量。

### 与论文的关系

- 英文机制论文不纳入影视实验；
- 毕业论文在系统实现与应用验证章节单独呈现；
- 专利由机制和系统实现共同支撑。

## 11. 论文重组任务

实验部分固定为六段：

1. **研究问题与识别策略**：四个RQ、数据、配对和主要终点；
2. **公开规则改变行为与结果**：唯一主结果表；
3. **机制传导与逐轮纠错**：行动、资源、报告、纠错、共识轨迹；
4. **替代解释与成本**：Outcome Only、等计算量对照、成本收益；
5. **异质性与稳健性**：分域、Seed、Panel规模、信息环境；
6. **失效边界与复现性**：GLM负结果、消融边界、冻结与哈希。

正文图表上限建议：

| 编号 | 内容 | 地位 |
|---|---|---|
| 图1 | 公开合同—行动—资源—报告—结算时序 | 方法核心 |
| 表1 | 三个主条件的行为、准确率、成本、CI | 唯一主结果表 |
| 图2 | 机制传导链 | 解释主结果 |
| 图3 | 奖励/成本Follower响应 | 博弈证据 |
| 图4 | 等计算量对照与成本收益 | 排除替代解释 |
| 图5 | 分域、Seed、Panel与跨模型边界 | 稳健性/边界 |

所有完整矩阵、信号字段变体、历史变体、敏感性权重和复现清单放入附录。

## 12. 执行顺序与时间安排

### 第一阶段：零API补全（3—4天）

| 日程 | 工作 | 完成标志 |
|---|---|---|
| Day 1 | WP1主张登记与分析manifest | 所有论文结论有证据映射 |
| Day 2 | WP2传导链、错误流向、逐轮轨迹 | JSON和初版图生成 |
| Day 3 | WP3 CI与成本收益 | 主结果表和成本图生成 |
| Day 4 | WP5经验响应面与论文图表重排 | 确认可识别范围 |

### 第二阶段：控制实验与机制分解（4—6天，取决于API）

| 日程 | 工作 | 完成标志 |
|---|---|---|
| Day 5 | compute-matched协议、矩阵、单测 | Prompt和预算冻结 |
| Day 6 | 5题/域smoke | 路由与日志100%完整 |
| Day 7 | 50题/域pilot | 450单元完整并冻结 |
| Day 8—10 | 机制成分分解正式采集 | 预设四个相邻比较完成 |

### 第三阶段：完整博弈响应与留出验证（5—8天）

| 日程 | 工作 | 完成标志 |
|---|---|---|
| Day 11—13 | 3×3奖励—成本开发网格 | 1350单元完整并冻结 |
| Day 14 | 拟合响应面、生成Pareto前沿 | m*按预设效用锁定 |
| Day 15—17 | m*、中心合同、无机制的留出检验 | test只评估一次 |
| Day 18 | 重复任务信誉实验 | 完成或按规则停止 |

### 第四阶段：影视应用验证（3—5天）

冻结请求集、运行三个条件、人工审核、统计与截图更新。该阶段可与论文文字重写并行，但不得反向改变理论论文主张。

### 第五阶段：统一写作收口（2—3天）

- 重写摘要、引言、实验与讨论；
- 统一所有数字、图题和结论口径；
- 生成一页claim-evidence核对表；
- 从冻结结果重建论文全部图表；
- 完成答辩用“主结果—机制—成本—边界”四页核心材料。

## 13. 停止规则

满足以下条件后研究实验正式停止：

1. 主实验、传导链、成本收益、等计算量对照、机制分解和合同留出验证均有完整结果或按规则停止；
2. 每个核心主张都有主要证据、反证检查和适用边界；
3. 所有正式数据通过schema、完整性、配对、路由和哈希验证；
4. 论文中的数字均可从冻结question-level数据一键生成；
5. 不再新增QA数据集、模型后端、拓扑路由器、AEIC字段或研究方向；
6. 不因结果不显著而改条件、删题或追加无预设样本；
7. 影视应用完成最小正式评测，或明确限定为工程实现与可追溯验证。

## 14. 人工工作

人工只需要承担不能可靠自动化的审核：

1. 审核claim-evidence registry中的结论是否符合导师口径；
2. 确认等计算量对照的Prompt没有向某一条件泄露答案或额外提示；
3. 审核影视数据中的模糊影片映射；
4. 抽查推荐理由的事实支持与约束违反；
5. 最终确认论文中没有超出证据的强表述。

数据复算、统计检验、绘图、表格、完整性验证和文稿数字同步均由脚本完成。

## 15. 最终完成标准

研究工作完成时，应能用五句话讲清楚：

1. 多智能体系统只聚合最终答案，无法判断Agent是否真正投入了独立推理和验证。
2. 本研究让Leader在行动前公开信用、成本、信息与预算规则，并让Follower行动控制真实资源路径。
3. Static AEIC显著提高高资源行动率，并在主后端的450道配对题上提高集体准确率。
4. 完整响应面、机制分解、传导链和等计算量对照解释效果来自何处，成本分析说明为此付出了什么资源。
5. Seed、Panel和信息环境说明稳健性，GLM负结果明确合同不能未经校准跨模型迁移。

若任何一张图、一个实验或一段文字不能服务上述五句话，应移入附录或删除，而不是继续扩展主线。
