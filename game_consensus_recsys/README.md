# Game Consensus Recommender

绿地影视搜推验证系统。论文主线是博弈驱动的多智能体语义共识；本包只提供应用域、工作流和可审计数据契约，不替代 `core/mechanism` 中的研究机制。

当前可运行切片（M0 + M1 skeleton）：

```text
typed request -> request-time planner -> catalog retrieval -> heterogeneous review
              -> evidence/reputation consensus -> Top-K + full trace
```

本地无外部依赖 smoke test：

```bash
python -m unittest tests.test_game_consensus_recsys
python -m game_consensus_recsys.demo
```

统一运行脚本：

```bash
./game_consensus_recsys/run.sh doctor  # 检查 offline/DeepSeek/API 环境
./game_consensus_recsys/run.sh demo    # 运行完整请求
./game_consensus_recsys/run.sh test    # 全套机制与系统测试
./game_consensus_recsys/run.sh api     # 启动零依赖 HTTP API
```

只使用一个 DeepSeek API。系统会自动复用 MAS 根目录已有的 `.env`；也可以用进程环境变量覆盖：

```bash
export DEEPSEEK_API_KEY="..."
# 可选；未配置时使用默认模型
export DEEPSEEK_MODEL="deepseek-v4-flash"
python -m game_consensus_recsys.demo
```

同一客户端承载 preference、diversity、critic 三个逻辑 Agent；它们使用独立角色指令和上下文。检索与硬约束验证保持确定性。DeepSeek 超时、空 JSON 或结构错误时，请求不会失败，而是记录 `agent_degraded` 并用确定性 Agent 继续返回结果。API Key 不写入 trace 或日志。

默认 HTTP 接口完全使用 Python 标准库。FastAPI 与 Uvicorn 仍可作为可选适配器：

```bash
uvicorn game_consensus_recsys.api:app --reload
```

当前 `ConsensusEngine` 是透明的首个静态机制基线。下一阶段通过适配层接入 `core/mechanism` 的可见契约、真实 follower action、收益结算和动态 Stackelberg 控制器。

重要边界：Planner 只读取请求时可见的 query、约束、偏好、Top-K 和时延预算，不读取 ground truth、最终相关性或事后收益。
