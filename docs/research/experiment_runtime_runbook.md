# 实验断点续跑与错误检测手册

## 1. 可靠性边界

- 检查点单位是一个完整的“问题—条件单元”。
- 单元内部包含全部Agent和全部轮次；只有单元完整通过Schema后才原子提交。
- 进程在单元中间退出时，该单元整体重跑，已经完成的单元不会重复调用。
- 纵向信誉实验只允许从连续完成前缀恢复；若发现中间缺题而后续题已完成，程序立即停止，禁止从未来信誉快照继续。
- 每个输出文件有`.run.lock`防止两个Collector同时生成同一单元；写入时另有短时文件锁。
- JSONL通过临时文件、`fsync`和`os.replace`提交；损坏行和重复键会明确报告文件与行号。

## 2. 运行矩阵

顺序运行：

```bash
python3 experiments/mechanism/run_matrix.py \
  experiments/mechanism/matrices/fixed_backend_control_v12.json
```

并行分片运行：

```bash
python3 experiments/mechanism/run_parallel_matrix.py \
  experiments/mechanism/matrices/fixed_backend_control_v12.json \
  --out-dir experiments/results/mechanism/control/fixed_backend_control_v12_shards \
  --max-workers 4
```

正式调用前必须先执行不访问API的矩阵级演练：

```bash
python3 experiments/mechanism/run_parallel_matrix.py \
  experiments/mechanism/matrices/fixed_backend_control_v12.json \
  --out-dir /tmp/fixed_backend_control_v12_dry \
  --max-workers 2 --dry-run
```

只运行精确的第3个Job：

```bash
python3 experiments/mechanism/run_matrix.py \
  experiments/mechanism/matrices/fixed_backend_control_v12.json \
  --job-index 3
```

相同命令可直接重新执行。Collector会读取检查点并跳过成功单元，只重试未完成单元。

## 3. 查看进行中状态

```bash
python3 experiments/mechanism/status_matrix.py \
  experiments/mechanism/matrices/fixed_backend_control_v12.json \
  experiments/results/mechanism/control/fixed_backend_control_v12_shards \
  --out experiments/results/mechanism/control/fixed_backend_control_v12_status.json
```

状态报告包含：

- expected/present/completed/failed/missing；
- 总进度；
- 每个数据集与条件的进度；
- transient/data/protocol/unknown错误分类；
- 非预期实验单元。

## 4. 错误处理规则

| 类别 | 例子 | 行为 |
|---|---|---|
| transient | DNS、ConnectionError、Timeout、HTTP 429/5xx | 指数退避后重试，不计入实验尝试上限 |
| data | AEIC JSON不可恢复、Schema错误、题目缺失 | 写入失败记录，按实验尝试上限重试 |
| protocol | HTTP 400/401/403、模型不支持、API Key缺失 | 立即停止，修复配置后原命令续跑 |
| unknown | 未归类异常 | 保存错误，有限重试后停止 |

不得通过删除失败记录来“恢复”。修复原因后执行同一命令，成功记录会原子替换旧失败记录，并把旧错误保存在`prior_errors`。

## 5. 校验

单文件严格检查：

```bash
python3 experiments/mechanism/validate.py records.jsonl --strict
```

完整矩阵检查：

```bash
python3 experiments/mechanism/verify_matrix.py \
  matrix.json shard1.jsonl shard2.jsonl \
  --out verification.json
```

矩阵检查同时验证：

- 预期单元无缺失、无重复、无意外；
- 每个单元包含完整的`轮次 × Agent`网格；
- Schema、Token、延迟、HTTP状态合法；
- 声明行动与真实执行后端一致；
- 输入文件SHA-256。

只有`pass=true`才能进入冻结和统计阶段。

## 6. 中断与恢复

安全中断使用一次`Ctrl-C`。已提交单元不会丢失，当前未提交单元下次整体重跑。

恢复前依次执行：

```bash
python3 experiments/mechanism/status_matrix.py matrix.json shards/
python3 experiments/mechanism/validate.py shard.jsonl --strict
python3 experiments/mechanism/run_parallel_matrix.py matrix.json --out-dir shards/ --max-workers 4
```

不要手工拼接JSONL、复制完成行或修改`question_id/condition/seed`。需要迁移分片时，先使用`verify_matrix.py`检查是否存在跨文件重复键。
