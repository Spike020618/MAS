# 补充实验阶段结果（2026-10-12）

## 1. 报告接口实验

设置：StrategyQA、MMLU-Hard、TruthfulQA各30题；同一DeepSeek模型、单次调用、最大800 Token、温度0.2；问题级配对比较Free Text、Generic Structure和AEIC Interface，共270条完成记录，错误与重复均为0。

### 总体结果

| 条件 | 解析成功率 | 立场提取率 | 准确率 | 平均输出Token |
|---|---:|---:|---:|---:|
| Free Text | 100.00% | 100.00% | 72.22% | 95.83 |
| Generic Structure | 98.89% | 98.89% | 82.22% | 91.50 |
| AEIC Interface | 98.89% | 100.00% | 85.56% | 150.02 |

AEIC相对Free Text的准确率差为+13.33个百分点（2道对转错、14道错转对；Holm校正McNemar p=0.00836），但平均输出增加54.19 Token。AEIC相对Generic Structure的准确率差仅+3.33个百分点（2道对转错、5道错转对；Holm校正p=0.453），不显著；两者解析成功率相同。因而不能把结果解释为“四字段格式具有独立准确率优势”。

AEIC的直接可观察优势是提供显式证据槽位；该槽位是否提高证据审核、翻转识别和复制识别的一致性，仍需盲审数据验证。

## 2. 合同与资源执行分解

设置：MMLU-Hard和TruthfulQA各30题，3个Agent、1轮；固定DeepSeek后端；共180条问题级记录和540条Agent轮次。完整性验证结果：180/180完成，路由、Schema、结构和Trace错误均为0。

| 条件 | 高资源行动率 | 准确率 | 平均Token/Agent轮次 |
|---|---:|---:|---:|
| No Contract / Fixed Resource | 14.44% | 85.00% | 731.43 |
| Contract / Fixed Resource | 81.11% | 88.33% | 892.32 |
| No Contract / Matched Compute | 17.78% | 86.67% | 974.83 |
| Full Mechanism | 81.11% | 91.67% | 949.26 |

Contract / Fixed Resource相对No Contract / Fixed Resource的高资源行动率增加66.67个百分点，Agent轮次配对Holm校正p=3.01×10^-36；准确率增加3.33个百分点，但只有2道错转对、0道对转错，Holm校正p=0.5。

Full Mechanism相对No Contract / Fixed Resource的高资源行动率同样增加66.67个百分点，Holm校正p=4.63×10^-35；准确率增加6.67个百分点，4道错转对、0道对转错，Holm校正p=0.25。

Matched Compute逐题、逐Agent复用Full Mechanism的后端、执行调用数和最大Token上限，但保留无合同选择提示。其实际平均Token为974.83，高于Full Mechanism的949.26，差异不显著（Agent轮次p=0.229；问题均值p=0.129），因此不是算力不足的弱基线。Full Mechanism相对Matched Compute仍将高资源行动率从17.78%提高到81.11%（配对p=2.82×10^-33）；准确率从86.67%提高到91.67%，3道错转对、0道对转错，但60题下McNemar p=0.25，方向为正而证据不足。

固定资源条件固定的是可用后端、调用次数和最大Token，而不是实际生成Token。合同诱导了更长的实际输出，因此准确率差仍不能解释为等实际计算量下的合同质量效应。

## 3. 当前可写结论

1. 公开过程合同在资源上限固定时仍显著改变Agent的行动声明和实际生成行为。
2. AEIC相对普通结构化回答未表现出显著的独立准确率优势，其研究价值应继续定位为可结算语义接口。
3. 等算力回放并未复制合同对行动分布的影响；这支持“公开规则改变Follower响应”，而不仅是后台多给了计算资源。
4. Full Mechanism在实际Token不高于Matched Compute时准确率高5个百分点，但仅3个配对纠正、p=0.25；现阶段只能报告方向和边界，不能宣称等算力下质量显著提高。
5. 下一步只需完成人工盲审，以判断显式证据槽位是否改善证据审核、翻转识别和复制识别一致性。

## 4. 证据文件

- `experiments/results/mechanism/interface/report_interface_v1.jsonl`
- `experiments/results/mechanism/interface/report_interface_v1.summary.json`
- `experiments/results/mechanism/decomposition/mechanism_decomposition_v1.jsonl`
- `experiments/results/mechanism/decomposition/mechanism_decomposition_v1.summary.json`
- `experiments/results/mechanism/decomposition/mechanism_decomposition_v1.stats.json`
- `experiments/results/mechanism/decomposition/mechanism_decomposition_v1.verification.json`
- `experiments/results/mechanism/decomposition/matched_compute_schedule_v1.json`
- `experiments/results/mechanism/decomposition/full_vs_matched_compute_v1.stats.json`
