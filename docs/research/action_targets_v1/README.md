# 行动目标核查

本目录只含离线开发审计与下一轮方案，不代表新实验完成或新机制有效。

- `轨迹归因.md`：40题逐题表、表达差异、缺失关系、不确定与具体反例。
- `轨迹归因.json`：80条模式轨迹、240阶段输出、支持句匹配、原文件哈希与执行失败。
- `三行动验证方案.md`：停止/取证/推导检查及普通审阅、重新求解控制；还需采集器与题单冻结。
- `本次验收.json`：离线测试、冻结文件哈希与零调用核查。

复核：

```sh
/opt/miniconda3/bin/python -m experiments.mechanism.audit_action_targets
/opt/miniconda3/bin/python -m unittest experiments.tests.test_action_targets experiments.tests.test_sequential_allocation_screen experiments.tests.test_sequential_cost_correction
```

`missing_evidence`不保证检索能纠正，`expression`不改官方评分，`uncertain`不强行计作推导错误。题目与支持句仅在离线审计脚本读取；请求构造器为白名单输入，不导入本审计模块。原40题已经用于方案开发，不能再作为新方法独立确认集。
