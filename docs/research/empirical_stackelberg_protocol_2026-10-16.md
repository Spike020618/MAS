# 有限合同空间的经验 Stackelberg 求解协议

> 2026-10-19状态更正：本文件保留为历史采集协议。观测策略下的有限合同择优不等于Follower最佳响应或Stackelberg均衡；相关最佳响应审计未通过。下文“经验Stackelberg解”的命名不再作为现有结果的允许主张。后续研究设计见`mechanism_mainline_redesign_2026-10-19.md`，历史数据与协议不删除。

## 研究目标

本阶段补全“Leader先承诺、Follower后响应、Leader据此选择合同”的双层闭环。研究结论限定为：在预先冻结的有限合同空间中，根据开发集观测到的Follower经验响应选择Leader效用最大的合同，并在一次性留出测试集上验证。

本阶段不声称连续合同空间的闭式均衡、一般激励相容、跨模型通用合同或无约束意义的全局最优。

## 博弈对象

- Leader：公开机制合同的设计者。
- Follower：接收合同后选择资源行动的LLM Agent。
- 合同：奖励倍率、成本倍率、信息策略和验证预算。本阶段固定信息策略与验证预算，只搜索奖励和成本。
- Follower行动：Low、AEIC、Verify、Imitate。
- Follower经验响应：各合同下观测到的行动分布，而非假设的完全理性唯一最优行动。
- Leader结果：集体准确率、Token、执行调用次数、API延迟和失败率。

## 冻结合同空间

奖励倍率为 `0.05 / 0.2 / 0.5`，成本倍率为 `0.5 / 1.0 / 1.5`，构成完整3×3网格。三个任务域各使用开发集50题；所有条件固定5个Agent、3轮、DeepSeek后端、信息策略、聚合方式、随机种子与执行代码。

完整网格包含1,350个问题—合同单元和20,250个Agent-round。矩阵文件为：

`experiments/mechanism/matrices/reward_cost_grid_v13_dev.json`

## Follower响应模型

描述性响应面为：

```text
logit Pr(high_resource) = beta_0 + beta_r reward + beta_c cost
                         + beta_rc reward*cost + domain fixed effects
```

模型用于解释方向和交互；合同选择直接使用九个实测网格点，不使用模型外推值。

## Leader效用

```text
U_L = accuracy
      - lambda_token * normalized_tokens
      - lambda_call * normalized_calls
      - lambda_failure * normalized_failure
```

归一化只在完整开发网格内进行。三组权重在查看完整网格前冻结：

| 偏好 | Token | Calls | Failure |
|---|---:|---:|---:|
| 质量优先 | 0.05 | 0.025 | 0.05 |
| 平衡 | 0.15 | 0.10 | 0.10 |
| 成本优先 | 0.30 | 0.20 | 0.10 |

主合同由“平衡”权重选择。并列时依次选择调用更少、Token更少、合同编号字典序更小者。质量优先和成本优先仅作效用敏感性分析。

### 延迟记录与协议补充（结果分析前登记）

原始Trace逐次记录行动选择、推理、验证及格式修复调用的`latency_s`。分析同时报告：Agent-round平均延迟、Agent-round P95延迟、问题级估算端到端平均延迟和问题级P95延迟。由于同一轮Agent并行执行，问题级延迟按“各轮最慢Agent耗时之和”估算，而不把同轮Agent耗时简单相加。

延迟指标在开发网格结果汇总前补充登记，但不追溯修改已经冻结并启动运行的主Leader效用与合同选择规则，以避免根据运行中结果改变主终点。延迟进入Pareto解释与部署敏感性分析；若后续研究把延迟纳入主效用，必须另行冻结权重并重新选择合同。

## 经验 Stackelberg 解

对冻结的有限合同集合 `M` 完整枚举：

```text
m* = argmax_{m in M} U_L(m, empirical_response(m))
```

因此可称为“冻结有限合同空间内的开发集经验全局最优合同”或“经验Stackelberg解”。不得省略“冻结有限空间”“经验”和“开发集”三个限定。

## 留出验证

开发集选择结束后自动生成测试矩阵，只比较：

1. No Mechanism；
2. Static AEIC中心合同 `(reward=0.2, cost=1.0)`；
3. 开发集锁定的 `m*`。

每个任务域使用150道未参与合同选择的测试题。测试集只运行一次；若 `m*` 未优于中心合同，如实报告负结果，不在测试集继续调参。

## 验收条件

- 27个开发分片全部通过Schema、路由、Trace和完整性校验；
- 原始九格均值与响应模型同时报告；
- Leader效用权重和选择规则可由冻结矩阵复算；
- `m*`生成后写入独立测试矩阵，开发与测试无题目重叠；
- 论文将博弈求解、边界和负结果采用相同口径表述。
