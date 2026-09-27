# 最小数据与实验补齐方案

> 日期：2026-09-06  
> 目标：保持“博弈驱动的去中心化语义共识机制及在影视推荐中的应用”不变，只补会影响论证闭环的证据。

## 一、只新增两个实验

### E-Control：同模型控制实验

目的：排除旧正式实验中 GLM / DeepSeek / Reasoner 随 action 改变造成的模型能力混杂。

设计：

- 数据：沿用 StrategyQA、MMLU-Hard、TruthfulQA 的冻结 test split，各抽 50 题；
- 条件：No Mechanism、Static AEIC；
- 规模：3 × 50 × 2 = 300 个问题—条件单元；
- Agent：3 个；
- 轮数：1 轮。该控制只验证合同与独立行动，不研究多轮信息传播；
- choice 和 execution 均使用同一个明确模型名；
- execution 使用相同 thinking 开关、max tokens、temperature 和 prompt 骨架；
- action 只被记录，不在该控制中改变模型、token 或调用数；
- 不把旧实验 seed 当作 API 可复现保证；以问题级配对为统计单位。

主要终点：

1. 高努力行动率；
2. question-level 最终准确率；
3. token 与延迟；
4. 三个域分别的效应与 95% CI。

统计：对问题进行 10,000 次 paired bootstrap；准确率另报 McNemar；不以 agent-round 为独立样本。

解释规则：

- action 改善且 accuracy 改善：支持合同在固定模型资源下的行为与质量效应；
- action 改善、accuracy 不显著：支持行为效应，旧质量增益解释为合同驱动资源配置的联合效应；
- 二者均不改善：限定为旧模型与旧路由环境下的系统效应。

### E-Movie：影视应用验证

目的：证明统一机制能在真实电影集合和真实用户对话上形成可审计 Top-K，而不是只运行 8 部手写电影。

数据来源：

- MovieLens 25M：稳定版本，约 6.2 万部电影、2,500 万评分、100 万 tag application，并包含 tag genome；
- ReDial：CC BY 4.0，11,348 个电影推荐对话，官方 train/test 划分；测试集 1,342 个对话，并提供 movie mentions、suggested、seen、liked 标签。

样本构造：

1. 从 ReDial train 抽 30 个对话作为开发集；
2. 从 ReDial test 冻结 100 个对话作为正式集；
3. 每个正式上下文必须至少有一个能映射到 MovieLens、由推荐者 suggested 且 seeker liked 的电影；
4. title + year 精确匹配，模糊匹配项必须人工确认；
5. 正式集冻结后不得按结果删除失败样本；
6. 英文 ReDial 是主实验；可额外构造 30 条中文硬约束查询作为系统功能测试，但不混入主指标。

三个方法足够：

- B0：BM25/字段检索；
- B1：单 DeepSeek 重排；
- Ours：多个自治角色产生 AEIC 报告，公开合同诱导行动，确定性规则完成共识，不使用中央 LLM Judge。

指标：

- Hit@5、Recall@5、NDCG@5；
- 硬约束满足率；
- catalog 外无效实体率；
- 可追溯证据支持率；
- token、调用次数、平均延迟、P95；
- API 失败后的降级成功率；
- 可选：类型覆盖率和残余异议保留率。

主系统只允许一个 DeepSeek endpoint。资源行动通过同一模型的计算配置实现：低努力使用较短输出/关闭 thinking，AEIC effort 使用完整结构化输出，verify 使用更高 reasoning effort 或第二次核验。模型升级不再作为合同效应的一部分。

## 二、需要下载什么

建议放在不提交 Git 的 `data/raw/`；记录 URL、下载日期和 SHA-256。

```bash
cd /Users/spike/code/MAS
mkdir -p data/raw/movielens data/raw/redial data/processed benchmarks/movie_consensus

curl -L --fail --retry 3 \
  -o data/raw/movielens/ml-25m.zip \
  https://files.grouplens.org/datasets/movielens/ml-25m.zip

curl -L --fail --retry 3 \
  -o data/raw/redial/redial_dataset.zip \
  https://github.com/ReDialData/website/raw/data/redial_dataset.zip

shasum -a 256 data/raw/movielens/ml-25m.zip \
  data/raw/redial/redial_dataset.zip \
  > data/raw/SHA256SUMS

unzip -q data/raw/movielens/ml-25m.zip -d data/raw/movielens
unzip -q data/raw/redial/redial_dataset.zip -d data/raw/redial
```

MovieLens 25M 应保留：

- `movies.csv`：movieId、title、genres；
- `ratings.csv`：用户隐式/显式偏好基线；
- `tags.csv`：自由标签；
- `genome-tags.csv` 与 `genome-scores.csv`：语义检索和理由证据；
- `links.csv`：IMDb/TMDB 外部 ID 映射。

ReDial 应保留官方 train/test 原文件和 license/datasheet 链接。不要下载旧 ReDial 训练代码和 GenSen 权重，本项目不训练 ReDial 原模型。

### 可选 IMDb 数据

只有确实要支持“导演/演员”查询时再下载：

```bash
curl -L --fail --retry 3 -O https://datasets.imdbws.com/title.basics.tsv.gz
curl -L --fail --retry 3 -O https://datasets.imdbws.com/title.crew.tsv.gz
curl -L --fail --retry 3 -O https://datasets.imdbws.com/name.basics.tsv.gz
curl -L --fail --retry 3 -O https://datasets.imdbws.com/title.ratings.tsv.gz
```

通过 MovieLens `links.csv` 的 IMDb ID 连接。IMDb 数据每日更新且限个人/非商业使用；必须冻结下载日期和文件哈希。最小版本可以完全不下载 IMDb，把查询范围限定为片名、年份、类型、评分和语义标签。

### 不建议作为主数据的来源

- TMDB：需要 API key、明确署名，并有缓存期限和 AI/验证用途条款；只为海报或简介接入会增加合规与复现负担。
- Kaggle 镜像：优先从数据原发布方下载，避免版本和许可不清。
- 自动抓取豆瓣：不建议，存在条款、版权、反爬与不可复现问题。

## 三、需要开发什么

### 数据脚本

建议新增：

```text
scripts/data/download_manifest.json
scripts/data/build_movie_catalog.py
scripts/data/build_redial_benchmark.py
scripts/data/validate_movie_benchmark.py
data/processed/movie_catalog.jsonl
benchmarks/movie_consensus/dev.jsonl
benchmarks/movie_consensus/test.jsonl
benchmarks/movie_consensus/dataset_card.md
```

`build_movie_catalog.py`：

- 解析 MovieLens；
- 去除缺年份/类型影片；
- 汇总评分数、均值和高相关 genome tags；
- 保留至少 5,000 部有足够交互或 tag genome 的电影；
- 所有证据带 `source_file + row/movieId`；
- 不把 test 对话标签写进检索特征。

`build_redial_benchmark.py`：

- 解析上下文、movie mentions、liked/suggested；
- title/year 映射 MovieLens；
- 输出 query、relevant_movie_ids、excluded_seen_ids、dialogue_id；
- train 只用于阈值和 prompt 开发，test 只运行一次正式评测。

`validate_movie_benchmark.py`：

- 检查 ID 全在 catalog；
- 检查 dev/test 对话不重叠；
- 检查正式集无空 relevance；
- 检查重复上下文和标签泄漏；
- 输出数量、映射率、类型分布和 SHA-256。

### 系统适配

需要把 `game_consensus_recsys` 的 `InMemoryCatalog` 换成同接口的真实 catalog，并接入 `core/mechanism`：

1. Retrieval/Preference/Diversity/Critic/Verifier 使用局部输入；
2. 每个 LLM Agent 先看到同一公开合同并选择 action；
3. action 控制同一 DeepSeek 模型的 token/thinking/verify 资源；
4. Agent 生成 AEIC candidate report；
5. 逐条 evidence 由 catalog ID/字段验证；
6. 确定性 consensus engine 聚合，保留 dissent；
7. trace 记录 model、prompt hash、action、token、latency、evidence ref 和结算。

不要引入中央 DeepSeek Judge；否则“去中心化共识”会重新变成中央裁决。

## 四、人工需要做什么

最低人工量：

1. 人工确认 ReDial 与 MovieLens 的模糊匹配，预计 1–2 小时；
2. 检查 100 条正式上下文是否真的表达可用偏好，预计 2–3 小时；
3. 抽 100 条系统推荐理由，按“事实支持/相关/违反约束/不可判断”标注；
4. 最好找第二个人独立标其中 50 条，计算 Cohen's kappa；找不到第二人时只能称人工审计，不能称双人一致性实验；
5. 记录任何排除项及排除发生在看结果之前还是之后。

不需要手工给 100 个查询标完整排序。ReDial 的 liked/suggested 电影作为 binary relevance 即可计算 Hit、Recall 和 NDCG；人工只负责映射与事实性审计。

## 五、DeepSeek 配置

新实验必须使用官方当前明确支持的 V4 模型名，不应继续默认旧别名。运行前：

```bash
export DEEPSEEK_MODEL=deepseek-v4-flash
./game_consensus_recsys/run.sh doctor
```

正式实验冻结：

- 完整 model string；
- API 返回 model；
- 采集时间；
- thinking、reasoning effort、temperature、max tokens；
- prompt/config hash；
- finish reason 和 usage；
- API 不支持或不保证的 seed 不得被写成严格复现依据。

## 六、推荐执行顺序

```text
第 1 天：下载 MovieLens/ReDial，生成下载 manifest 和 hash
第 2 天：构建 catalog，完成 ReDial 映射和 dataset card
第 3 天：将真实 catalog 接入系统，完成三个 baseline
第 4 天：实现同模型 E-Control，先跑每域 5 题 smoke test
第 5 天：冻结 E-Control 150 题 × 2 条件
第 6 天：跑 100 条 ReDial 正式影视实验
第 7 天：人工审计、question-level CI、图表和论文改写
```

停止规则：

- 不新增模型后端；
- 不训练推荐神经网络；
- 不新增 QA 数据集；
- 不重新跑旧 5,850 条记录；
- 不追加自适应拓扑大实验；
- 正式 test 结果出来后不修改查询集或按表现删除样本。

## 七、最后需要形成的交付物

```text
数据：movie_catalog.jsonl + dev/test benchmark + dataset_card + hashes
实验：fixed_backend_control + movie_application_eval
统计：question-level estimates + 95% CI + per-domain/query-type results
系统：真实目录 + action + AEIC + evidence + consensus + trace
论文：一个统一机制、两个验证场景、明确负结果和适用边界
```

这套方案补齐后，题目中的“博弈驱动、去中心化、语义共识、影视推荐应用”各自都有可运行实现和对应证据，同时新增 API 成本和人工标注量保持在较小范围。
