# 影视推荐系统应用评测

## 1. 评测范围

当前应用评测覆盖MovieLens目录中的18,430部影片，以及30类显式影视请求、150条Top-5推荐。请求包括类型包含/排除、年份上下界和复合类型约束。

本评测回答两个问题：

1. 系统能否正确执行显式请求并返回真实目录实体；
2. 多Agent共识层是否形成完整、可审计的证据与执行轨迹。

它不使用隐式用户画像，也不把工程验收结果表述为个性化推荐质量提升。

## 2. 工程验收

| 指标 | 结果 |
|---|---:|
| 请求数 | 30 |
| Top-5推荐数 | 150 |
| 目录实体有效率 | 100% |
| 硬约束满足率 | 100% |
| 证据引用完整率 | 100% |
| 合同先于行动的Trace顺序 | 100% |
| Agent覆盖完整率 | 100% |
| 平均离线延迟 | 27.2 ms |
| P95离线延迟 | 38.4 ms |

这些结果证明系统链路可运行、约束可执行、报告可追溯；不证明相关性优于推荐基线。

## 3. 与目录检索的对照

| 指标 | 目录检索 | 共识工作流 |
|---|---:|---:|
| 推荐数 | 150 | 150 |
| 实体有效率 | 100% | 100% |
| 硬约束满足率 | 100% | 100% |
| 平均Top-5类型覆盖数 | 6.13 | 6.13 |
| 无重复列表率 | 100% | 100% |
| 证据完整率 | 0% | 100% |
| Trace有效率 | 0% | 100% |

在当前显式约束请求和确定性Agent配置下，共识工作流没有改变目录检索生成的Top-5集合，因此不能声称提高了离线相关性或多样性。它增加的是候选级证据、Agent覆盖检查、合同时序和可审计共识记录。这与系统在毕业设计中的定位一致：系统是机制落地和应用验证载体，而不是已经证明优于所有推荐算法的新推荐模型。

## 4. 人工案例审核

系统已生成150行人工审核表，审核人只需填写：

- 相关性（0—2）；
- 推荐理由事实性（0/1）；
- 列表适配度（0/1）；
- 争议备注。

人工审核完成前，不报告人工相关性均值或理由事实正确率。

## 5. ReDial离线基准

### 5.1 数据与防泄漏处理

原始数据来自Hugging Face镜像中的`recwizard/redial`数据快照：训练集8,004段对话，测试集1,342段对话。构建脚本将ReDial影片名称确定性映射到18,430部MovieLens目录；可映射且满足构造条件的对话分别为7,348和1,203段。召回层使用全部1,203条测试样本；需要调用模型的重排实验固定随机种子抽取100条测试样本。

每条样本只保留首个正向目标影片出现之前的对话；目标影片来自其后的正向推荐标注。只有截断点以前实际出现的影片才能进入“已看”和“喜欢”输入，防止使用后续对话标签。验证脚本确认：开发集与测试集对话零重叠、目标影片全部映射到目录、上下文中不含目标影片ID，且已看影片作为硬排除项进入所有方法的共同输入。

### 5.2 对照方法

- `lexical retrieval`：依据对话文本、片名和目录元数据检索；
- `cooccurrence retrieval`：只使用ReDial训练集影片共现信号；
- `hybrid retrieval`：对归一化词法分数与共现分数作加权融合。30条开发对话从共现表中排除后独立选定权重0.85，测试标签不参与参数选择或召回器生成；
- `single-agent rerank`：对混合召回的相同Top-10短名单调用一次偏好Agent重排；
- `consensus rerank`：对相同Top-10短名单并行执行Preference、Diversity、Critic三类Agent评审，再由共识引擎聚合。

召回层在全部1,203条样本上进行配对比较；Agent重排在相同100条样本、相同MovieLens目录、相同混合召回候选池和相同Top-5口径下比较。全部结果由可断点续跑脚本逐条写入。

### 5.3 全量召回层结果

| 方法（N=1,203） | Hit@5 | Recall@5 | NDCG@5 | 已看影片排除满足率 |
|---|---:|---:|---:|---:|
| 词法召回 | 0.75% | 0.26% | 0.24% | 100% |
| 纯训练集共现召回 | 10.47% | 5.83% | 4.88% | 100% |
| 词法—共现混合召回（权重0.85） | **10.81%** | **5.95%** | **5.01%** | 100% |

混合召回相对词法召回的Hit@5提高10.06个百分点，配对Bootstrap 95%置信区间为[8.40, 11.89]个百分点。123条样本只在混合召回中命中，2条只在词法召回中命中，Exact McNemar检验`p=3.70×10^-34`。混合召回又比纯共现高0.33个百分点，说明主要增益来自训练集共现，词法信号提供小幅互补，而不是把增益错误归因于Agent重排。

失败归因进一步显示：1203条样本中130条在Top-5命中；未命中样本的首要限制是截断前没有正向影片锚点（约52%），其次是目标未进入Top-50（约25%）。这说明后续质量上限主要受冷启动与候选覆盖约束，而非共识聚合公式。

### 5.4 固定候选池上的Agent重排

| 方法（N=100） | Hit@5 | Recall@5 | NDCG@5 | 实体有效率 | 已看影片排除满足率 |
|---|---:|---:|---:|---:|---:|
| 混合召回直接排序 | 6.00% | 4.17% | 2.66% | 100% | 100% |
| 单Agent重排 | 7.00% | 5.17% | 3.11% | 100% | 100% |
| 多Agent共识重排 | 7.00% | 5.17% | 2.96% | 100% | 100% |

100条样本上，单Agent和多Agent均比直接排序多命中1条；二者命中集合完全相同。共识与单Agent的Exact McNemar检验`p=1.0`，共识与直接排序也只有1个不一致命中，`p=1.0`。本次结果不支持“多Agent共识提高推荐准确率”的结论，也不足以把1条增益解释为稳定优势。

### 5.5 成本与可审计性

| 方法 | 平均Token | 平均模型调用 | 平均延迟 | P95延迟 | 已验证证据覆盖率 |
|---|---:|---:|---:|---:|---:|
| 混合召回 | 0 | 0 | 191 ms | 302 ms | 0% |
| 单Agent重排 | 5,842 | 1 | 7.44 s | 9.96 s | 0% |
| 多Agent共识重排 | 16,935 | 3 | 12.68 s | 28.69 s | 51.2% |

“已验证证据覆盖率”只统计带有系统可核验目录证据的候选报告。单Agent生成了解释文本，但没有独立验证路径，因此不把该文本记为已验证证据。三个共识角色已改为真正并行执行，冻结权重批次平均延迟为12.68秒；共识仍付出约3倍模型调用和Token成本。

### 5.6 结论边界与模型版本

混合召回后，100条Agent样本中相关目标进入Top-10候选池的比例为9%，进入Top-50的比例为25%。候选覆盖得到改善但仍然偏低，因此重排方法的可提升空间受限。下一阶段若继续提升推荐质量，应补充MovieLens协同过滤或语义向量召回，而不是继续增加Agent数量。

Agent实验通过DeepSeek官方API的`deepseek-chat`别名运行。该别名在不同时期可能指向不同服务端模型快照，因此本文只比较同一采集批次内的配对条件；影视实验不与早期机制实验的绝对性能作跨批次比较。实验清单保留模型别名、采集日期、提示和数据哈希。

通用机制实验回答“公开过程规则能否改变Agent行动并改善集体决策”；影视评测回答机制能否落地，以及检索、约束、重排、证据和成本在真实目录中的表现。两部分共享机制实现，但不混用结论。

## 6. 可防守结论

> 影视系统已完成18,430部真实影片目录上的机制落地与可追溯链路验证。在全部1,203条无标签泄漏的ReDial测试样本上，开发集选定的词法—共现混合召回将Hit@5从0.75%提高到10.81%，且差异通过配对检验；冻结权重后的100条Agent实验中，直接排序、单Agent和多Agent的Hit@5分别为6%、7%和7%，共识保持100%的实体与约束有效率并提供51.2%的已验证证据覆盖，但未显著提高推荐准确率。因此可声称召回改进和审计能力成立，不能声称共识机制提升了推荐相关性。

## 7. 证据文件

- `experiments/results/movie_acceptance/summary.json`
- `experiments/results/movie_acceptance/comparison.json`
- `experiments/results/movie_acceptance/acceptance_results.jsonl`
- `experiments/results/movie_acceptance/manual_review.csv`
- `experiments/run_system_acceptance.py`
- `experiments/evaluate_movie_system.py`
- `experiments/run_movie_benchmark.py`
- `experiments/analyze_movie_benchmark.py`
- `experiments/analyze_retrieval_ablation.py`
- `experiments/analyze_movie_failures.py`
- `experiments/tune_hybrid_retrieval.py`
- `experiments/scripts/data/build_redial_cooccurrence.py`
- `experiments/results/movie/summary.json`
- `experiments/results/movie/summary_hybrid.json`
- `experiments/results/movie/summary_hybrid_w085.json`
- `experiments/results/movie/retrieval_ablation.json`
- `experiments/results/movie/hybrid_tuning.json`
- `experiments/results/movie/failure_analysis.json`
- `experiments/results/movie/full_lexical.jsonl`
- `experiments/results/movie/full_cooccurrence.jsonl`
- `experiments/results/movie/full_hybrid_w085.jsonl`
- `experiments/results/movie/lexical.jsonl`
- `experiments/results/movie/single.jsonl`
- `experiments/results/movie/ours.jsonl`
- `experiments/results/movie/hybrid_lexical.jsonl`
- `experiments/results/movie/hybrid_single.jsonl`
- `experiments/results/movie/hybrid_ours.jsonl`
- `experiments/benchmarks/movie_consensus/manifest.json`
- `experiments/benchmarks/movie_consensus/validation.json`
