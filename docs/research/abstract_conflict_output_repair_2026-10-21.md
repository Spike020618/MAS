# 报告输出与计分修复（v2）

## 修复范围

新增独立入口 `experiments/mechanism/pilot_abstract_conflicts_v2.py`。未修改 v1 源码、协议、调用、计分或审计结果。

1. 两分支全部阶段统一使用 1600 输出 Token 上限（旧版 700）；限制文本字段和引用长度，避免结构化分支无限展开。这只能降低截断风险，尚不能宣称真实 API 截断已消除。
2. 仅接受正常结束、完整 AEIC、合法引用列表的报告；结构化分支还要求完整 dispute 字段。引用字符串的客观存在性另记，不等同事实正确性。
3. 无效报告的票为 null，不再替换成 abstain。两个有效且相同的明确票才形成共识；三方各一票也不视为认识性弃答。
4. 初始报告无效时向同伴显式传递失败状态、null 报告和原始证据，不伪造弃答立场。
5. 无有效共识即失败，在全部计划题的分母中计错，包括真实标签为 abstain 的题。不剔除失败题。
6. 不自动续写、修复半截 JSON 或只重跑失败题。成功 HTTP 返回的无效报告保留；网络错误停止，按既有调用日志断点续跑。

## 验证状态

13 项相关单元测试通过，包括无网络的完整 12 题 / 108 作业模拟：两分支输入一致、上限一致、无效输出不误计为弃答正确、完整作业恢复不追加调用。`git diff --check` 通过。

未运行付费 v2 在线复验，未产生新准确率结论。提高上限是否消除真实模型截断，仍需在线验证。

## 复验约束

使用原 12 题做完整工程回归，不将原 heldout_seed 称为新独立测试。固定规则后才启动，108 次新调用；不沿用 v1 报告、不只重跑 8 次失败、不依据输出调整奖励或题目。

```sh
python3 -m experiments.mechanism.pilot_abstract_conflicts_v2 --out-dir experiments/results/mechanism/stackelberg/abstract_conflict_smoke_v2
# 确认冻结协议后，完整在线回归：
python3 -m experiments.mechanism.pilot_abstract_conflicts_v2 --out-dir experiments/results/mechanism/stackelberg/abstract_conflict_smoke_v2 --collect
```

结构化处理的额外输出与 Token 成本仍应计入比较。即便格式可靠，也不能将其视为机制增益、真实任务有效或 Stackelberg 最优性成立。
