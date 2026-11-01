# 当前运行：停止与继续取证

本轮只研究决策空间，不训练博弈。预先固定40题、开发/确认各20题，三个阶段、两个推理模式，最多320次模型请求，费用代理上限5美元。

- [冻结方案](锁定方案.md)
- [评分前成本更正](评分前成本更正.md)：资料新颖度规则不必付中间答案成本，未改模型采集。
- 原始结果：仓库下`experiments/results/sequential_allocation_v1/`，受既有gitignore排除，需另外备份。

## 无新增调用的验收

全部采集完成后，在仓库根目录运行：

```bash
/opt/miniconda3/bin/python -m experiments.mechanism.verify_sequential_allocation_screen
```

验证器禁止网络，检查冻结代码与任务、逐请求白名单和原始响应、检索原文、断点日志、历史169份冻结数据和此前525份验收产物，并复算指标/成本。它不依赖执行日价格，不修改原始响应或操作耗时日志。

`run_sequential_allocation_screen run`是显式联网采集入口，不能作为无网络验收命令。其`replay`入口保留采集日价格校验，因此跨日应使用上面的独立验证器。

分析入口：

```bash
/opt/miniconda3/bin/python -m experiments.mechanism.run_sequential_allocation_screen analyze
/opt/miniconda3/bin/python -m experiments.mechanism.correct_sequential_allocation_costs analyze
```

`result.json`保留原分析；成本解释以`corrected_analysis.json`为准。更正承诺`cost_correction_commitment.json`在整批评分前保存，包含时间、已封存题数及哈希。不能因为结果不好而更改冻结协议或删除attempt记录。

原生资料是数据集提供的未标记检索段落，不是完整金标准事实集。相同API模型的thinking/非thinking是不同资源配置；只在各自模式内比较停止阶段。阶段耗时记录的是成对模式的实验执行时间；单路径HTTP/检索构成项不是另一次真实在线端到端时延。

回答观察oracle与不付观察成本的理想选择oracle都不可部署，二者限制不同。即使数值门槛通过，也必须核查事实差异并与普通控制器比较，不自动启动120题训练。
