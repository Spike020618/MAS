# MAS 项目深度审查：论文、实验与系统

> 审查日期：2026-09-06  
> 审查口径：结论是否能由现有代码与冻结数据唯一支持，而不是功能是否“看起来完整”。

## 1. 总判断

项目已经具备一套相当扎实的实验资产：8 个冻结 suite、5,850 个问题级记录、完整性校验、SHA-256 provenance 和独立复算均通过；正式主实验也保留了 action、offer、AEIC、token、模型路由、失败重试和纵向信誉。这些比早期仅对固定输出重加权的实验可信得多。

但当前还不能把论文、实验与影视系统作为一条完全闭合的证据链提交。最严重的问题是正式实验存在“机制条件 → 行动选择 → 模型后端”的能力混杂；其次是论文理论章节仍描述旧的 21 参数离线 Stackelberg 优化，与现在的公开契约和真实 follower action 实验不是同一个形式模型；最后，影视系统仍是 8 部影片的演示切片，没有接入论文机制，也没有影视搜推评测结果。

如果不整改，最稳妥的论文结论只能是：**在特定 DeepSeek 决策器、提示合同和分级资源路由组成的联合策略下，系统改变了模型声明的行动并提高了 QA 决策质量。** 目前不能干净地宣称质量增益由 AEIC 激励机制单独造成，更不能宣称已经证明一般性的激励相容或 Stackelberg 最优。

## 2. 阻断性问题（P0）

### P0-1：正式质量增益被模型后端路由混杂

`generate_one()` 在 `resource_tiered=True` 时将 `low_effort` 路由到 `zhipu_flash`，将 `verify` 路由到 `deepseek_reasoner`，其余高努力主要使用 `deepseek`。正式冻结数据的实际执行计数为：

| 条件 | GLM-4-Flash | DeepSeek Chat | DeepSeek Reasoner |
|---|---:|---:|---:|
| No Mechanism | 6,719 | 27 | 16 |
| Outcome Only | 6,542 | 147 | 168 |
| Static AEIC | 2,907 | 3,780 | 436 |

因此条件不仅改变“努力”，也大幅改变回答模型。Static AEIC 的 82.22% 对 No Mechanism 的 68.00% 可能来自合同、更多 token、不同执行 prompt、DeepSeek 替换 GLM、额外验证调用的任意组合。现有 Outcome Only 控制不足以分离这些路径。

必须补做至少一个同后端控制，推荐直接做 2×2 因子设计：

1. 合同：No Mechanism / Static AEIC；
2. 执行资源：固定 DeepSeek / 行动分级路由；
3. 同一问题、同一 choice backend、同一 agent persona、同一轮数配对；
4. 同时报告合同对 action 的效应，以及在固定后端下合同对质量的总效应；
5. 额外做 token-matched 或调用数匹配分析，避免把预算差异误称为机制质量。

通过门槛：固定 DeepSeek 后端时仍出现方向一致的 action 改变；质量若不再显著，论文必须把质量收益解释为“机制诱导的资源分配联合效应”，不能说是 AEIC 报告本身的独立作用。

### P0-2：理论模型与当前实证对象不是同一个机制

论文第 4 节仍定义 21 参数 `theta`、二元 follower 策略、离线 differential evolution 和固定 AEIC 输出；第 7 节实际研究的是四种行动、自然语言公开合同、动作后资源路由、三轮生成和信誉结算。当前理论中的 lower level 甚至写成 `w_i=f(tau_i; theta)`，这是权重计算，不是 follower best response。

现有定理也存在形式缺口：

- 紧致参数集加“目标有界”不足以使用 Weierstrass；还需要目标连续或至少上半连续，而离散 best response 会引入跳变。
- “把 IC/EQ 指数设大即可令 IC=1”没有证明，并且可能因零质量、并列和归一化而失败。
- `min_i w_i` 可能为 0，个体理性条件可能无有限解。
- `lambda -> 0` 只能给出带误差的比较，不能推出存在一个有限小 `lambda` 时准确率必不低于 uniform，除非增加间隔条件。
- 收敛式使用固定 `2 gamma/(n-1)`，文字却称速率由图的代数连通度决定；缺少具体更新方程、步长条件、连通性和谱界证明。

整改建议不是继续补旧证明，而是重写为与实验一致的有限行动公开合同模型：leader 选择合同与资源策略，LLM follower 产生随机行为策略 `pi(a|prompt, model)`，系统观察 action 后执行资源并结算。理论贡献可降为：有限行动下最优经验合同的存在、预算约束下的策略优化定义、以及明确假设下的可识别性命题。未经严格证明的“激励相容保证”和“指数收敛保证”应删除或改为命题/设计目标。

### P0-3：影视系统没有验证论文机制，也没有证明系统优越性

`game_consensus_recsys` 当前使用 8 部手写电影、词面召回、固定线性打分和简单信誉更新。它没有调用 `core/mechanism` 的公开合同、action choice、收益结算或动态信誉控制；DeepSeek Agent 固定标记为 `aeic_effort`，并没有先看到合同再选行动。`research_manifest.json` 声明的 NDCG@K、约束满足率、无效候选率、证据支持率、P95 延迟等指标也没有对应结果集。

这意味着当前系统只能证明“有一个可运行 API vertical slice”，不能证明研究成果在影视搜推中有效。毕业设计要求“两个研究点和一个系统”时，这会形成明显断层。

最低闭环：

1. 将 `core/mechanism` 通过适配层接入影视候选报告，而不是另写一个同名静态共识；
2. 接入可审计的真实影视目录，至少数千部影片；
3. 建立 300–500 条冻结复杂查询，含硬约束、否定、多人偏好、长尾与事实查询；
4. 基线至少包括单 Agent、确定性检索、固定融合、多数投票、Static AEIC、全量 deliberation；
5. 报告 Recall/NDCG、硬约束违反、无效实体、理由事实性、覆盖/多样性、token、调用数和 P95；
6. 做固定质量下的成本比较，或固定预算下的质量比较；
7. trace 必须能显示合同、action、实际资源、证据来源、结算和最终贡献。

## 3. 高风险问题（P1）

### P1-1：“激励”目前主要是自然语言提示，不是具有真实后果的主体收益

模型被提示为“resource-constrained, self-interested follower”，成本和 credit 是文本中的数字。Static AEIC 没有可持续的外部奖励、模型更新或未来权限变化；对无状态 API 模型而言，结算后的 utility 不会反向影响其偏好。因此实验识别的是“对公开规则措辞的 prompt response”，而不是传统意义上稳定偏好主体的经济激励响应。

论文已经部分承认 prompt-mediated costs，但仍应统一使用“announced contract / induced action response”，避免直接声称 incentive compatibility。若要增强博弈解释，可让累计收益真实决定下一轮 token、工具、可见信息、投票权或是否被选中，使奖励成为系统内可兑现后果。

### P1-2：处理效应是一个复合干预，缺少机制拆解

行动一旦选定，会同时改变执行提示、max tokens、后端，有时还增加第二次验证调用。现有 signal ablation 是合同字段奖励消融，不能分解“结构化格式、推理指令、预算、模型、验证调用”各自贡献。

建议补四个消融：仅合同措辞、仅 AEIC schema、仅 token 增加、仅 backend escalation；再加入合同+固定执行 prompt。这样才能回答“AEIC 为什么有效”。

### P1-3：行为统计存在层级依赖和潜在伪重复

高努力 action 来自 question × agent × round 的嵌套重复观测，同一问题内并不独立。当前 agent-round McNemar/秩检验把大量相关观测当作配对单元，会产生极小 p 值。已有 question-level high-effort 汇总是正确方向，但主文应以问题为聚类单元，采用 question-level paired bootstrap、cluster-robust GLM/GEE 或混合效应模型；agent-round 分析只能作为描述性结果。

同时报告 95% CI 和效应量，不要只报告 p 值。三域异质性应报告 domain × treatment interaction 或逐域区间，而不仅是 pooled 结果。

### P1-4：随机种子并不等价于独立复验

相同问题、同一服务端模型别名、同一 prompt 的 seed 7/42/99 只能测试采样扰动；如果 API 不保证严格 seed 复现，seed 甚至只是记录字段。应记录服务端返回的模型版本/指纹（若提供）、采集时间、temperature、top_p 和 seed 是否实际传入 API。真正的复验至少需要新的问题样本或时间批次。

### P1-5：公开过程奖励容易被“填字段”投机

Static AEIC 的奖励主要基于 AEIC 完整度和非 neutral 结论；模型可以填写形式完整但事实错误的内容获得 credit。主结果恰好显示规则有效，但没有证明抗 Goodhart 或抗策略性伪造。应增加：无关但完整证据、重复证据、伪造引用、流畅错误推理等攻击条件，并区分 schema completeness 与 verifiable correctness。

### P1-6：证据验证仍然弱

QA 实验中的 `verified_evidence` 实际通过“最终 stance 是否等于 ground truth”代理，不是逐条证据核验；影视系统中 DeepSeek 证据统一 `verified=False`，只有确定性 Agent 的元数据证据能进入 accepted evidence。这不足以支撑“可验证语义报告”的强表述。

需要明确三种标签：结构完整、结论正确、证据逐条可验证。影视域应以 catalog 字段、来源 ID 和规则校验验证证据，不能用最终推荐是否正确反推证据正确。

### P1-7：数据来源与任务构造仍不足以让第三方复现

冻结 bundle 能复算统计，但无法从公开原始数据重新构造问题缓存。缓存 meta 只记录 dataset/n_agents/n_samples，缺少数据集版本、split、license、原始样本 ID、下载/转换脚本版本；MMLU-Hard 和 TruthfulQA 被改写成“proposed answer 是否正确”的二分类任务，也需要说明 proposed answer 的生成与正负样本平衡规则。

应冻结 dataset card：来源 URI/版本/许可、原始 ID、转换代码 commit、过滤规则、类别分布、去重检查和三个 split 的哈希。

### P1-8：系统的信誉状态会跨用户污染，并且线程不安全

HTTP server 持有一个共享 `ConsensusWorkflow`，其中 `ConsensusEngine.reputation` 是进程级可变字典；所有用户请求共享信誉，且 `ThreadingHTTPServer` 可并发修改它。当前信誉也只按 agent ID 更新，不按用户、会话、任务域或证据反馈区分。结果既可能数据串扰，也可能产生并发竞态。

正式系统应将信誉键至少设为 `(scope, agent_id, mechanism_version)`，明确 global/session/user scope；使用事务存储或锁；匿名请求默认无状态。还要定义只有可观测反馈到来后才更新信誉，不能仅因 Agent 自己声称的 verified evidence 更新。

## 4. 中风险问题（P2）

- 根 README 仍讲旧的 17/21 参数、10 数据集和异构多后端叙事，与新 AEIC-MD 论文和“只提供一个 DeepSeek API”的系统约束冲突。
- `experiment_completion_ledger.md` 仍把多个已冻结 suite 标为 RUNNING/PENDING，与当前 manifest 冲突，容易让答辩材料自相矛盾。
- 资产审计虽总体 pass，但仍记录 3 个旧文件重复、32 个文件含错误；必须明确这些仅为 legacy，不得被论文图表脚本误读。
- 部分冻结分析 JSON 保留 `/Users/spike/Downloads/MAS/...` 和临时隐藏目录路径，降低跨机器可移植性；应在冻结输出中统一使用仓库相对路径。
- 影视 DeepSeek Agent 三个角色串行调用，默认 30 秒超时和重试，2 秒 latency budget 并未真正执行；Planner 的 topology 和 verification budget 目前主要写入 trace，没有控制实际调度。
- DeepSeek 返回的 token 总量被复制到每个候选报告，候选级成本相加时会重复计费；需要 request/agent-call 级成本实体。
- DeepSeek 的“diversity”评估逐候选独立输出，但列表多样性本质上是集合属性；当前共识逐候选打分也没有 MMR/coverage 约束。
- 检索按中文单字 token 命中，8 部电影上的测试不足以说明搜索能力；没有召回索引、画像、交互日志、冷启动或多轮会话。
- HTTP API 缺少认证、限流、请求取消、SSE、持久化、结构化日志和健康检查中的真实下游探测；适合作为演示，不适合称为完整工程系统。
- 当前大量关键代码、测试、系统和文档处于 Git 未跟踪状态；若机器或目录损坏，冻结证据链并不受版本控制保护。

## 5. 现有结论的安全等级

| 结论 | 当前可支持程度 | 建议措辞 |
|---|---|---|
| 公开 Static-AEIC 规则改变声明的行动分布 | 较强 | 可保留，但限定为测试模型与 prompt action setting |
| Static-AEIC 单独导致 +14.22pp 质量提升 | 不成立 | 有模型/预算/提示复合混杂，补固定后端因子实验 |
| Outcome reward 无效，因此 AEIC 激励独特有效 | 中等偏弱 | 只说明该 outcome-only prompt 在当前标度下无效 |
| 奖励升高、成本升高产生单调反应 | 描述性支持 | 保留为 manipulation check，不等同真实经济偏好 |
| 机制激励相容并存在 Stackelberg 均衡 | 理论不足 | 删除强保证或在新模型下重证 |
| 跨模型泛化 | 已被否定 | 保留 GLM 负结果，是可信度加分项 |
| 3–5 Agent 稳健 | 有限支持 | 只对当前题集、同质 DeepSeek choice policy 成立 |
| 系统在影视搜推中优于基线 | 无证据 | 完成影视评测前不得写入结论 |
| 系统支持动态拓扑/预算优化 | 尚未实现 | 当前应称 rule-based plan trace/skeleton |

## 6. 建议的最小补强实验

按投入产出排序：

1. **固定后端主控制（必须）**：No/Static × fixed DeepSeek/resource-tiered，三域各至少 100–150 个配对 test 问题。
2. **预算匹配控制（必须）**：在同 token/调用预算下比较 No vs Static；报告质量—成本 Pareto。
3. **问题级稳健统计（必须）**：cluster bootstrap 或混合模型，补 95% CI、逐域效应和 interaction。
4. **合同拆解（强烈建议）**：contract text、AEIC schema、execution prompt、token、backend escalation 的因子/消融。
5. **策略攻击（强烈建议）**：完整但错误、复制、无关证据、伪造证据、高信誉错误者。
6. **影视离线实验（必须用于毕设系统）**：冻结查询集、真实目录、基线与质量/成本/约束指标。
7. **小型盲评（建议）**：对推荐理由事实性、相关性、分歧保留做双人标注并报告一致性。

## 7. 提交前验收门

只有以下全部通过，才建议使用“结论经得起推敲”的表述：

- 主质量效应不存在未控制的后端、token 或调用数混杂；
- 理论符号、行动空间、时间顺序与实际代码一致；
- 所有定理给出充分假设和完整证明，或降级为经验假设；
- 主统计以 question 为独立单元并报告区间；
- 数据版本、转换、split、模型返回版本和 prompt 哈希可追溯；
- 冻结 bundle 在另一目录/机器上可一键复算且不含绝对路径依赖；
- 影视系统实际运行论文机制，而非同名静态打分器；
- 影视系统至少有一套正式离线评测，且结论包含失败边界；
- README、论文、台账、图表和代码只保留一个权威叙事；
- 关键代码与冻结 manifest 纳入版本控制并打 release/tag。

## 8. 推荐的论文主线调整

保持“博弈 + 多智能体共识”为叙事主线，但将贡献拆成可防守的两层：

1. **研究点一：公开可审计合同如何诱导 LLM follower 的结构化行动响应。** 主证据是固定后端下 action response、质量与成本。
2. **研究点二：观察行动后如何分配模型、验证和通信预算。** 分级后端带来的质量收益应放在这一层，解释为 leader 的资源配置策略，而不是混入研究点一的机制效应。
3. **系统：影视搜推作为联合机制的外部应用验证。** QA 证明机制行为，影视实验验证业务效用与工程成本。

这样不仅保留原有主线，还能把当前最危险的后端混杂转化为第二研究点的正当对象，但前提是补做因子控制，分别估计“合同效应”和“资源配置效应”。
