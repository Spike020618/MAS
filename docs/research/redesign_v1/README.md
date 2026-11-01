# 新研究环境与验收

先读[一小时研究验收](一小时研究验收_2026-10-25.md)，不要把设计快照当成已成立贡献。正式登记题目不变；本目录不修改论文、专利、系统或PPT。

## 数据和结果

- 原始数据：`experiments/data/raw/redesign_hotpot_v1/`，Parquet旁有固定commit、URL、许可及SHA256。
- 索引：`experiments/results/redesign_v1/retrieval_env/wiki_fts.sqlite`，约2.7GiB。
- 首8题：`open_evidence_screen_v2/`；v1只准备、未调用。
- 新16题：`acquisition_replication_v1/`。
- 路径归因：`acquisition_path_attribution_v1/`。
- 格式/推理控制：`answer_order_control_v1/`、`evidence_thinking_control_v1/`。
- 128题覆盖：`retrieval_coverage_v1/`。
- 全索引校验：`index_parity_v1/`。
- 最终快照：`hour_acceptance_v1/`。

以上结果名均相对`experiments/results/redesign_v1/`。该结果目录和原始大数据受既有gitignore排除；不要以为只提交代码就备份了原始证据。可提交的精简数值与逐批哈希保存在本目录验收JSON。不要将2.7GiB数据库加入Git。

## 无新增付费调用的验收

仓库根目录运行：

```bash
/opt/miniconda3/bin/python -m experiments.mechanism.accept_redesign_hour
```

它禁止网络，验证冻结代码、标签/输入、请求与响应、全部新增取证原文、历史169份冻结文件、相关测试及diff；复跑不会请求模型。依赖PyArrow由`redesign_hotpot_data.parquet_module()`查找，本机位于临时生产依赖目录；换机需要在Python环境提供PyArrow，不能把临时路径当成随仓库携带的安装包。

单批亦可运行`repeat_open_evidence_solver replay`、`replicate_evidence_acquisition replay`、`attribute_acquisition_paths replay`、`control_answer_order replay`、`control_evidence_thinking replay`、`probe_acquired_evidence_pool replay`，均用`python -m experiments.mechanism.<模块> replay`。

`run`是显式联网执行入口，不应作为验收命令。未知请求状态必须人工核对，脚本不自动重发；不是删除attempts记录后重新尝试。已冻结代码或协议不得就地修改后继续同一批，须新版本。
