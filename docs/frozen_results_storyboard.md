# AEIC-MD 冻结结果：论文叙事与图表底稿

> 数据口径：`results/mechanism/frozen/manifest.json`（8 个套件，5,850 个 question-level 单元）。
> 本文只使用冻结包中的正式、泛化、边界、消融和稳健性结果；旧的拓扑/评审资产保留为背景和补充材料，不能替代本主实验的因果证据。

## 建议的主结论

**AEIC-MD 的核心贡献不是“选一个更好的拓扑”，而是把 Agent 的努力与报告方式变成可观察、可激励的 follower response；在 DeepSeek 主实验中，公开的 Static-AEIC 规则显著改变这一行为选择，并带来决策质量提升。**

这个措辞有两个边界必须同时写清：

- 现有正式证据支持 **Static AEIC mechanism**，不支持把动态/Stackelberg 自适应机制宣称为主结论；相应候选在 pilot 中没有成为正式比较对象。
- 在 held-out GLM backend 上，质量差异为 -1.33pp（69.33% vs 70.67%，Holm-adjusted *p*=0.791）。因此应该写作“存在后端依赖性、需要进一步机制校准”，而不是泛化优越性。

## 结果链条

1. **真实响应（主证据）**：在正式 450 道配对题目上，Static AEIC 将高努力选择率从 0.46% 提升至 56.93%（+56.47pp），同时决策质量从 68.00% 提升至 82.22%（+14.22pp，Holm-adjusted McNemar *p*=1.23e-11）。仅按结果奖励的质量为 67.33%，与无机制无显著差异（*p*=0.780）。这直接区分了“有激励”与“可验证 AEIC 激励”。
2. **剂量/边界证据**：三数据集的各自响应均符合预测方向：奖励提高时高努力率单调不减，成本提高时单调不增。该曲线使用预实验的同域中档中心点与冻结的低/高边界点，因此应称为“跨域一致的行为剂量响应”，不能与正式 test 主效应混为同一估计量。
3. **信息操纵检查**：在 evidence、social-only、copy-pressure 与 counterevidence 四类可控信息条件下，Static AEIC 都提高高努力选择。该图应作为 AEIC 报告接口可测量/可操作的支持，而不把所有条件都解读为准确率提升。
4. **稳健性边界**：两个额外 seed 上，Static AEIC 的质量为 85.67% vs 无机制 71.00%（+14.67pp，Holm-adjusted *p*=2.40e-09）。3/4/5 Agent 条件质量均在 87.33%–88.67% 范围内，未出现显著质量退化。信号去除和历史处理的结果主要表明部件并非在该样本下可简单地“逐项必要”；避免写成四个字段都被逐一证明不可缺少。
5. **外部边界（必须诚实报告）**：held-out GLM 上静态机制未复现主实验的努力或质量提升。这是论文的重要限制，也是后续工作应从“跨 backend 合同校准”而非“扩展拓扑”切入的原因。

## 主文图表

| 图 | 文件 | 论文中的功能 | 建议标题/图注核心 |
|---|---|---|---|
| Fig. 1 | `results/mechanism/figures/fig1_formal_mechanism.{pdf,png}` | 主效应 | Revealed AEIC incentives change follower effort and improve decision quality. Outcome-only incentives do not. |
| Fig. 2 | `results/mechanism/figures/fig2_incentive_response.{pdf,png}` | 机制响应 | Effort responds monotonically to reward and inversely to effort cost. |
| Fig. 3 | `results/mechanism/figures/fig3_information_conditions.{pdf,png}` | 操纵检查 | Static AEIC changes effort choices across controlled information environments. |
| Fig. 4 | `results/mechanism/figures/fig4_ablations_robustness.{pdf,png}` | 稳健性/消融 | Quality remains stable under panel-size and history variants; component ablations are descriptive, not isolated necessity proofs. |

建议将 held-out GLM 的负向泛化放进主文表格或 Fig. 1 旁的“external validity”小表，而不要藏进附录；它决定了主张的可信范围。

## 写作结构（替换旧的“拓扑主线”）

1. **Introduction**：多智能体系统真正难点不是把回答聚合起来，而是让成本敏感 Agent 在可观察规则下选择值得信赖的努力与报告。AEIC 是报告接口，机制规则是 leader 的公开承诺，拓扑只是可选的信息控制手段。
2. **Method**：定义行动空间（low-effort / AEIC-effort / verify / defer-imitate）、可见合约、声誉状态与决策聚合。仅把“Stackelberg”用于本实验实际实现的 leader-rule/follower-response 顺序；不要扩展为已验证的动态最优解。
3. **Experiment**：先放 formal 主效应，再放边界响应与信息操纵，最后放 seed/panel-size/held-out backend。统计以 question-level paired tests 和 Holm 校正为准。
4. **Discussion**：主发现是“可验证的结构化激励”而非一般性结果奖励；模型依赖性表明 incentive semantics 必须做 backend-specific calibration。拓扑研究作为下一层成本/通信控制，而不是方法身份。

## 现阶段不应写的结论

- “动态 Stackelberg 优于静态机制”；冻结正式实验并没有提供这个比较。
- “跨模型泛化”；GLM held-out 结果不支持。
- “AEIC 四字段逐项不可或缺”；当前信号消融未显示一致的质量损失。
- “拓扑是主要因果变量”；本轮正式设计验证的是机制规则与 follower response。

## 可复现生成

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mas-mpl XDG_CACHE_HOME=/tmp/mas-xdg \
python3 experiments/mechanism/boundary_curves.py \
  results/mechanism/frozen/boundary_v11_dev/question_level.jsonl \
  --center results/mechanism/pilot/pilot_v10_strategyqa_dev50.jsonl \
  --center results/mechanism/pilot/pilot_v10_mmlu_dev50.jsonl \
  --center results/mechanism/pilot/pilot_v10_truthfulqa_dev50.jsonl \
  --out results/mechanism/figures/boundary_curves.json
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mas-mpl XDG_CACHE_HOME=/tmp/mas-xdg \
python3 experiments/mechanism/generate_paper_figures.py
```
