# Consensus Game 实现验收

## 完成项

本轮完成独立实现与离线执行链路，不再逐项请求确认；没有修改历史原始结果、论文、专利、系统或 PPT，也没有新增付费调用。

- 公式级核心：原文交叉归一化、同步 piKL、固定默认参数、末次/平均策略分开输出。
- 完整文本评分适配：本地 Hugging Face 因果模型 teacher forcing，逐目标 Token 累加 log likelihood；不以单 Token 编码、自报置信度或采样频率替代。
- 同池对照：G、MI、SC、D、ER-G、ER-D，全候选一致；平局弃答，不用候选 ID 决胜。
- 可恢复执行：逐评分槽封存、本地检查点内容哈希、输入/代码哈希锁定、单进程锁、请求文本绑定校验。无远端重试或下载。
- 先封存读出，后读取离线标签；配对 Exact McNemar 与八项 Holm 校正。没有标签进入评分器。
- 开销记录：序列评分槽数、目标 Token 数、实际本地评分耗时及离线求解耗时分开；导入分数文件的上游调用/计费 Token 未知，不冒称为零。
- 软件测试覆盖公式单步、归一化、对称性、候选置换、非法输入、完整评分、背景校正、六臂读出、配对统计、封存重跑和输入篡改拒绝。

## 实际执行结果

对旧的 12 道已开卷题做了零新增调用的更新算法诊断。11 道具备完整编码概率输入，ER-G 与 ER-D 均为 9/12，均有 1 道因输入不完整弃答；相对旧博弈输出各改变 1 道，但正确总数没有提升。

结果：`experiments/results/audit/equilibrium_reference_replay_2026-10-25/result.json`。

这不是原论文复现：仍是旧单 Token 编码输入；更新和初始概率处理也同时改变，不能单独归因于更新算法。它只能说明，没有发现换上 piKL 就能使这批旧输入获得准确率增量。

## 接口与环境边界

核对 DeepSeek 官方 Chat 与 FIM 文档：公开字段提供输出 Token 的 logprobs，没有发现能回传指定完整输入候选各 Token 似然的接口；FIM 的 echo 不能与 logprobs 同用。将候选作为强制前缀也不能直接获得前缀概率。因此不能用这个接口宣称完成完整文本似然复现。此结论为公开文档能力核查，未为验证不受支持的字段增加付费请求。

https://api-docs.deepseek.com/api/create-chat-completion/
https://api-docs.deepseek.com/api/create-completion/

本机可发现 transformers，但没有 torch、可用本地模型检查点、Ollama 或 llama-cli。完整文本评分适配器已实现，但未经过真实模型运行验收；不能把构造评分的端到端测试说成模型实验。不自动安装大体积依赖、下载未知许可模型、调用 Qwen，或把小型模型接口测试包装成研究增益。

## 可运行命令

离线软件验收（不联网、不调用 API）：

```sh
python -m experiments.mechanism.verify_equilibrium_reference
```

完整评分文件模式：

```sh
python -m experiments.mechanism.run_equilibrium_reference \
  --cases /absolute/path/public_cases.json \
  --scores /absolute/path/sequence_scores.json \
  --output /absolute/path/new_run_directory \
  --labels /absolute/path/private_labels.json
```

本地模型模式：用 `--local-model` 替代 `--scores`，同时提供 `--revision`，其值必须为 `sequence_likelihood.checkpoint_digest(model_path)` 的完整内容哈希。模型只从本地读取，禁用远程代码和自动下载。

输入每题仅允许 `id`、`question`、`candidates`（ID 到完整文本的映射）、`background_correction`（布尔）。标签为单独的 ID 到正确候选 ID 映射。评分文件以题 ID、评分槽为键；各槽必须含匹配的 `context`、`target` 和完整 `token_logprobs`。槽由 `requests_for_case` 生成，缺失、额外字段或文本不一致均拒绝。

提示模板、分词边界约定和本地模型设置目前明确属于适配；本地评分跑通也不等于复现原论文数据预处理与全部实验。严格原文复现仍需对齐作者模板、模型与数值记录，本轮未找到并验收可直接对照的作者实现。

## 研究结论

“实现完成”与“博弈有增量”分开验收。当前可交付的是公式级实现、评分执行链路、软件测试与旧轨迹诊断；不是新的有效性结果。真实完整文本实验受限于评分后端，不以旧编码数据充数，也不把旧 450 题收益拼到新求解器。

ECON 的可学习协调作为后续参照保留，不在没有忠实基线和可用反馈链路时仓促追加一个所谓新均衡。普通非博弈控制器仍必须作为必要对照。
