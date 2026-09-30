# AEIC-MD 实验完成台账

> 目的：将“全部实验完成”拆成可审计证据。状态只能是 `COMPLETE`、`RUNNING`、
> `PENDING` 或 `STOPPED_BY_RULE`；没有原始数据、完整性验证和可复算统计时不得标为完成。

## 总体验收

| 要求 | 当前状态 | 完成证据 |
|---|---|---|
| 真实 follower-response 闭环 | COMPLETE | `core/mechanism/`、`experiments/mechanism/collect.py`；行动在看到合约后选择，并按行动路由真实资源与累计声誉 |
| 可恢复采集 | COMPLETE | 原子唯一键存储、文件锁、失败重试、失败声誉回滚、纵向状态恢复；`tests/test_mechanism.py` |
| Pilot 与机制选择 | COMPLETE | `pilot_v10` 三域 750 条、dose calibration、`docs/pilot_decision_log.md` |
| 正式主实验 | COMPLETE | `experiments/results/mechanism/frozen/formal_v10/` 及 `.recomputed/`；1,350 个 question-level 单元 |
| 跨信息结构 | RUNNING | `information_v11_dev.json`，目标 1,200 单元 |
| 跨模型 | RUNNING | `heldout_glm_v11_dev.json`，目标 300 单元；Qwen 欠费失败作为留档，不计入有效复验 |
| 激励/成本边界 | RUNNING | `boundary_v11_dev.json`，目标 600 单元 |
| AEIC 信号消融 | PENDING | `signal_ablation_v11_dev.json`，目标 750 单元 |
| 历史消融/负对照 | PENDING | `history_ablation_v11_dev.json`，目标 600 单元 |
| 随机种子稳健性 | PENDING | `seed_robustness_v11_dev.json`，目标 600 单元；与 seed 42 组成三次复验 |
| 缺失 Agent 稳健性 | PENDING | `missing_agent_v11_dev.json`，目标 450 单元 |
| 统一冻结与一键复算 | PENDING | 每套矩阵须生成 `frozen/<suite>/` 与 `<suite>.recomputed/`，最后生成总 provenance |

## E0–E6 证据映射

### E0 资产审计

- 状态：`RUNNING`（初始审计完成，全部新采集结束后仍须运行终局审计）。
- 权威证据：`experiments/results/audit/asset_inventory.json`、
  `experiments/results/audit/result_consistency.json`、`experiments/results/audit/data_lineage.md`。
- 终局动作：重跑 `experiments/audit_experiment_assets.py`，记录最终哈希与结构完整性。

### E1 行为测量校准

- 状态：`RUNNING`。
- 新因果操纵：`independent`、`evidence_update`、`social_only`、
  `copy_pressure`、`counterevidence`；每个行为分别比较 NoMechanism 与 StaticAEIC。
- 完成门：三域 1,200/1,200；缺失、重复、路由错误均为 0；操纵检查与
  unsupported flip、novelty/copying 的 question-level 统计冻结。

### E2 真实 follower-response 主实验

- 状态：`COMPLETE`。
- 正式 test：三域 × 三条件 × 150 = 1,350；20,250 agent-rounds。
- 主证据：`experiments/results/mechanism/frozen/formal_v10/verification.json`、
  `analysis.json`、`statistics.json`、`sensitivity.json`、`provenance.json`。
- 边界说明：StaticAEIC 进入正式实验；AEIC-Contract 与 Adaptive 候选在 pilot
  中未通过独立 QA，作为负结果保留，不选择性删除。

### E3 策略强度与失效边界

- 状态：`RUNNING`。
- 新证据：奖励 0.05/0.2/0.5 与成本 0.5/1.0/1.5 的三域响应曲线；中心点复用
  pilot_v10，极值由 `boundary_v11_dev` 采集。
- 完成门：奖励和成本响应表、最坏组、质量—努力—成本曲线及哈希冻结。

### E4 信息结构与成本控制

- 状态：`RUNNING`（新机制内信息结构）+ `COMPLETE`（旧拓扑资产）。
- 旧资产：`experiments/results/cache/*_{sequential,star,debate,hierarchical}.json`、
  `experiments/results/configs/router_v10_results.json`、`experiments/results/cross_topology_transfer.json`。
- 证据边界：旧资产只支持“拓扑影响质量/成本及路由可行性”，不支持真实激励响应；
  新 `information_v11_dev` 用于检验信息结构如何调节 AEIC 机制响应。
- 预注册停止规则：不重新训练新的拓扑 router。原因是主线问题由现有旧资产和新信息
  结构矩阵覆盖；额外 router 只会重复拓扑工程，重新造成研究偏航。

### E5 评审域策略性验证

- 状态：`RUNNING`（既有自然域采集完成，尚待纳入终局 provenance；限定结论）。
- 原始数据：`experiments/results/review/iclr26v1.parquet`、`experiments/results/review/panel/*.json`。
- 统计：`study1_analysis.json`、`study1_stats.json`、
  `study2a_soft_calibration.json`、`study2c_routing.json`、
  `regulated_analysis.json`、`cross_model_analysis.json`。
- 证据边界：评审域用于自然群体病理、软校准边界、prompt 调控负结果与成本路由；
  不把旧 regulated prompt 称为 AEIC-MD 或 Stackelberg 因果验证。
- 预注册停止规则：不追加新的评审 API 正式实验。当前论文主因果结论由三 QA 域的真实
  follower-response 给出；评审数据作为外部生态有效性与边界证据保留，避免把缺少
  可核验 truth label 的 agreement 当作机制准确率。

### E6 泛化、消融与稳健性

- 状态：`RUNNING/PENDING`。
- 跨域：StrategyQA、MMLU-Hard、TruthfulQA 的正式与 dev 复验。
- 跨模型：DeepSeek 调参/主实验，GLM held-out follower policy。
- 消融：去 A/E/I/C、隐藏/重置/打乱历史。
- 稳健性：激励与成本极值、seed 7/42/99、3/4/5 Agent、flip threshold 与
  cost-weight sensitivity。
- 完成门：所有预注册矩阵无缺项，方向与最坏组汇总冻结。

## 冻结协议

每套新矩阵只有同时满足以下条件才可标记 `COMPLETE`：

1. `verify_matrix.py`：expected = present = completed，且 missing、incomplete、
   unexpected、route error 全为 0；
2. `freeze_suite.py`：生成不可覆盖 bundle、canonical question-level JSONL、分析、
   配对统计、敏感性结果和 SHA-256 provenance；
3. `recompute_frozen.py`：从冻结数据独立复算并逐字段比对成功；
4. 终局资产审计通过，生成总 manifest，列出正结果、负结果和停止项。
