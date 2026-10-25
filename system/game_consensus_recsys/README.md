# Game Consensus Recommender

绿地影视搜推验证系统。论文主线是博弈驱动的多智能体语义共识；本包提供应用域、工作流和可审计数据契约，并通过适配层使用 `experiments/core/mechanism` 中的研究机制。

当前可运行应用验证切片：

```text
typed request -> request-time planner -> catalog retrieval -> frozen shortlist
              -> uniform heterogeneous review -> gated evidence/reputation consensus
              -> Top-K + full trace
```

本地无外部依赖 smoke test：

```bash
python -m unittest system.test_game_consensus_recsys
python -m system.game_consensus_recsys.demo
```

统一运行脚本：

```bash
./system/game_consensus_recsys/run.sh doctor  # 检查 offline/DeepSeek/API 环境
./system/game_consensus_recsys/run.sh demo    # 运行完整请求
./system/game_consensus_recsys/run.sh test    # 全套机制与系统测试
./system/game_consensus_recsys/run.sh api     # 启动零依赖 HTTP API
./system/game_consensus_recsys/run.sh acceptance  # 运行30条系统工程验收
```

`acceptance`生成请求级结果、汇总报告和150行人工审核表。每次成功的HTTP搜索还会自动归档到`experiments/results/system_traces/`，文件名为request_id。

只使用一个 DeepSeek API。系统会自动复用 MAS 根目录已有的 `.env`；也可以用进程环境变量覆盖：

```bash
export DEEPSEEK_API_KEY="..."
# 可选；未配置时使用默认模型
export DEEPSEEK_MODEL="deepseek-v4-flash"
python -m system.game_consensus_recsys.demo
```

同一客户端承载 preference、diversity、critic 三个逻辑 Agent；它们使用独立角色指令和上下文。检索与硬约束验证保持确定性。DeepSeek 超时、空 JSON 或结构错误时，请求不会失败，而是记录 `agent_degraded` 并用确定性 Agent 继续返回结果。API Key 不写入 trace 或日志。

默认 HTTP 接口完全使用 Python 标准库。FastAPI 与 Uvicorn 仍可作为可选适配器：

```bash
uvicorn system.game_consensus_recsys.api:app --reload
```

系统已通过适配层接入 `experiments/core/mechanism` 的可见合同和真实 follower action。`low_effort`、`aeic_effort`、`verify` 与 `defer_imitate` 使用不同 Prompt、Token 预算与核验路径；行动、Token、延迟和降级事件写入 trace。当前应用切片采用单轮异质审议与透明的证据/信誉聚合，不把它表述为论文三轮协议的完整复刻。

所有进入最终排名的候选使用同一短名单和同一组 Agent 覆盖；硬约束是门控而非统一加分项；信誉在整轮评分结束后批量更新，不让候选输入顺序影响结果。

浏览器工作台展示请求时计划、公开合同、候选级共识分解、多 Agent AEIC 报告和完整审计时间线。正式 MovieLens 目录可由以下命令生成：

```bash
unzip experiments/data/raw/movielens/ml-25m.zip -d experiments/data/raw/movielens
python experiments/scripts/data/build_movie_catalog.py
```

重要边界：Planner 只读取请求时可见的 query、约束、偏好、Top-K 和时延预算，不读取 ground truth、最终相关性或事后收益。
