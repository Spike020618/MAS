# 系统证据归档

## 真实在线运行截图

以下四张图片均裁自同一张整页截图 `case_b_fullpage.png`，因此请求、推荐结果和Agent工作台属于同一次真实运行。

| 文件 | SHA-256 | 内容 |
|---|---|---|
| case_b_fullpage.png | `b0e3b5cb808bd4c1ca6b570924f5e8d17f880fab0bcccc70543a3c3334935c72` | 完整页面 |
| 01_request_plan.png | `ab86910cfc9c57551151c1dbe23aeb30dc45943ec94e76ace580c0c460b28c35` | 请求与执行计划 |
| 02_agent_aeic.png | `55ecda753dca78597b01bd6cd2a986534fa9f1a847f3b88d022636e112ea9fef` | 多Agent AEIC工作台 |
| 03_candidate_consensus.png | `000b6441d53e5ca041b7be2da0fdb1bfc50bedbb91998c89ceb8fbecbcc35121` | 候选共识审计 |
| 04_final_recommendations.png | `491127fdb8af661441c599c1746da858aedabf3065d65076f82eafece90ea060` | 最终推荐结果 |

`case_a_trace.json`来自相同配置的另一轮真实复现，SHA-256为
`916bb64d0f0da5fe2e00c42e0503fed17943411fabe50014b51865d64e4b2a36`。
该文件不能表述为截图运行的同一request_id。

## 工程验收

运行命令：

```bash
./system/game_consensus_recsys/run.sh acceptance
```

验收覆盖30条多约束请求和150个Top-5结果，检查目录实体、年份与类型约束、证据引用、合同先于行动的Trace顺序，以及统一和完整Agent覆盖。

验收结果存放在 `experiments/results/movie_acceptance/`：

- `summary.json`：汇总指标
- `acceptance_results.jsonl`：30条请求级记录
- `manual_review.csv`：150行人工审核表
- `REPORT.md`：可引用的结果说明

本验收只支持工程正确性结论。推荐质量优势仍需ReDial配对评测和人工审核。
