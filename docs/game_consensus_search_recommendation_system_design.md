# 博弈驱动多智能体共识在影视搜推场景中的系统化方案

> 版本：2026-08-12  
> 定位：技术调研与毕业设计方案沉淀  
> 约束：博弈与多智能体语义共识是唯一叙事主线；影视搜索推荐是验证域；`tvos-agent` 仅作场景参考，系统绿地建设。

> 实施状态（2026-08-20）：已在 `game_consensus_recsys/` 建立绿地系统首个可运行切片，完成版本化研究 manifest、请求/候选/证据/报告/共识/trace schema、请求时 Planner、内存目录检索、异质 Agent、证据—信誉共识排序、可选 FastAPI 接口和回归测试。当前是透明静态机制基线，下一阶段接入 `core/mechanism` 的可见契约与真实 follower response。

## 1. 总体判断

论文不需要改题换主线，也不需要把已有 MAS 成果硬塞进旧系统。更自然的组织方式是：

> 面向影视搜索推荐中“多源候选冲突、证据质量不一、生成式召回不稳定、用户偏好与事实约束并存”的问题，研究博弈驱动的多智能体语义共识机制；通过结构化语义报告、验证反馈、动态信誉、激励契约和通信拓扑，使异质 Agent 在有限预算下形成可靠、可解释、可审计的推荐共识；最终在新建的对话式影视搜推系统中完成应用验证。

系统不是论文的第三条平行主线，而是研究机制的实验装置与应用载体。

## 2. 两个研究点的重新界定

### 研究点一：博弈驱动的可信语义共识形成机制

核心问题：

**在异质 Agent 可能低努力、复制、从众、固执或策略性报告时，如何通过可见激励和验证约束，使高质量、独立且可验证的信息在共识中获得更高影响力？**

方法组成：

1. **AEIC 语义报告空间**：将 Assumption、Evidence、Inference、Conclusion 作为可比较、可审计的策略报告；
2. **局部可信度测量**：IC/EC/EQ/EV、证据新颖度、证据来源可靠度、无依据翻转、复制率与推理连续性；
3. **博弈收益函数**：事实/约束满足、独特证据贡献、边际决策贡献、验证成本、通信成本和策略惩罚；
4. **动态信誉机制**：按多轮、多任务历史更新 Agent 的后续权重、可见信息与验证预算；
5. **真实 follower response**：机制规则在报告前公开，Agent 实际选择低努力、完整报告、追加验证、反驳、保留意见或退出；
6. **共识输出**：不强迫所有 Agent 文本趋同，而形成“最终决策 + 保留分歧 + 证据链 + 信誉变化”的可审计共识。

这里的“共识”应定义为**可执行决策共识**，不是所有 Agent 最后说同一句话。近期研究已经表明，强制多轮一致可能导致正确少数派被错误多数同化；因而保留异议、利用完整轨迹和反从众约束，反而能提高最终可靠性。

### 研究点二：预算约束下的动态共识组织与验证机制

核心问题：

**面对不同复杂度和风险的搜推任务，Leader 如何动态选择参与 Agent、通信拓扑、信息可见范围、验证深度和停止时机，以更低成本获得可靠共识？**

它仍然是博弈与共识研究，而不是独立的推荐算法研究。

Leader 的动作空间：

- 选择 Agent：检索 Agent、生成式推荐 Agent、用户偏好 Agent、批判 Agent、事实验证 Agent；
- 选择拓扑：parallel、targeted-review、star、debate、sequential 或局部 peer-to-peer；
- 选择消息图：只传候选、传候选加证据、匿名报告、隐藏多数立场、公开信誉；
- 分配资源：token、调用次数、事实验证次数和整体时延；
- 选择升级：单 Agent → 异质双 Agent → 局部评审 → 全体审议；
- 选择终止：低风险早停、稳定共识停止、预算耗尽降级。

Follower 的响应：

- 是否投入更多推理/检索成本；
- 是否提供独立证据；
- 是否复制高信誉 Agent；
- 是否接受反证后修正；
- 是否在低收益时保留意见或退出。

研究点二最终回答“共识如何高效形成”，研究点一回答“共识为何可信”。

## 3. 影视搜推场景如何承载共识研究

### 3.1 一个请求不是让多个 Agent 各推荐五部电影

多 Agent 的价值应来自信息和目标的异质性：

- **Intent Agent**：解析显式约束、隐含偏好和否定条件；
- **Retrieval Agent**：基于影视目录、演员、导演、类型、年代等做可验证召回；
- **Generative Retrieval Agent**：基于会话和行为序列生成主题或 Semantic ID 候选；
- **Preference Agent**：根据长期画像和当前会话估计个性化效用；
- **Diversity Agent**：关注列表覆盖、新颖性和同质化风险；
- **Critic/Verifier Agent**：检查实体存在、属性事实、约束满足、推荐理由和供给可用性；
- **Consensus Agent/Mechanism**：执行收益计算、信誉更新和最终候选聚合。

这些角色持有不同信息与局部目标，因此候选冲突和协商具有真实意义。

### 3.2 搜推任务中的“策略行为”

可以构造或观测：

- 检索 Agent 为节省成本只返回热门结果；
- 生成 Agent 产出语义漂亮但目录中不存在的候选；
- 偏好 Agent 过度迎合最近一次点击；
- 后发 Agent 直接复制多数候选；
- 高信誉 Agent 错误导致群体盲从；
- 多样性 Agent 为追求差异牺牲约束满足；
- Verifier 在预算不足时选择浅验证或跳过验证。

这些行为可映射到 effort、truthful reporting、imitation、verification 和 dissent 等博弈策略，而不是人为制造“诚实/恶意”二分类。

### 3.3 共识对象

每个 Agent 对候选影片提交结构化报告：

```json
{
  "candidate_id": "movie:12345",
  "assumption": ["用户接受节奏较慢的科幻片"],
  "evidence": [
    {"claim": "用户过去偏好硬科幻", "source": "profile", "ref": "event:..."},
    {"claim": "影片类型包含科幻", "source": "catalog", "ref": "movie:12345"}
  ],
  "inference": "历史偏好与影片类型匹配，且满足年代约束",
  "conclusion": {"support": true, "utility": 0.81, "rank": 2},
  "confidence": 0.76,
  "cost": {"tokens": 420, "latency_ms": 310}
}
```

最终共识是带证据的 Top-K 列表，并保存未采纳候选、反对理由和信誉更新。

## 4. 生成式推荐在本课题中的位置

生成式推荐不是第二主线，而是一个重要异质 Agent 和冲突来源。

Semantic ID 生成式推荐已有明确研究基础：TIGER 将物品表示为有语义的离散 codeword 序列，并用序列到序列模型预测下一物品；近年的研究继续探索 SID 与语言语义对齐、推理和并行生成。但这一范式也存在解码树导致的概率耦合、左到右误差累积和早期 token 错误传播等问题。

本课题采用三档实现，按资源选择：

1. **最低可行版本**：电影实体使用稳定 MovieID；LLM 只从检索得到的合法候选 ID 中受约束选择，不声称训练了生成式推荐模型。
2. **论文实验版本**：用影片文本 embedding + residual/vector quantization 生成 3–4 层 MovieSID；在公开交互数据上训练轻量 Transformer 预测下一 SID，并使用 trie 约束解码。
3. **扩展版本**：研究 MovieSID Agent 与传统检索 Agent 在不同冷启动、长尾和复杂意图条件下的互补性，但不把模型结构创新作为论文贡献。

推荐使用第二档作为完整毕设目标，第一档作为必须先完成的系统基线。

## 5. 绿地系统总体架构

建议新建独立仓库，例如 `game-consensus-recsys`；MAS 继续作为研究与实验资产库，`tvos-agent` 保持只读参考。

```text
React Web Client
  - 对话 / 搜索 / 推荐卡片
  - 流式状态
  - 研究 Debug 面板
          |
FastAPI Gateway (SSE)
  - auth/session/rate limit
  - request schema
          |
Context Manager
  - short-term dialogue
  - long-term preference
  - experiment assignment
          |
Game-aware Planner / Leader
  - risk & complexity estimation
  - agent/topology/budget selection
          |
Consensus Workflow Runtime
  +-----------------------------+
  | Intent / Retrieval / SID    |
  | Preference / Diversity      |
  | Critic / Verifier Agents    |
  +-----------------------------+
          |
AEIC Report Ledger
  - structured reports
  - evidence graph
  - action/cost trace
          |
Game Mechanism Engine
  - payoff
  - reputation
  - verification allocation
  - topology/stop decision
          |
Evidence-aware Consensus Ranker
          |
Response Grounder + Quality Gate
          |
Streaming answer and cards

Data plane:
PostgreSQL/pgvector | Redis | Movie catalog | Interaction events | Experiment logs
```

## 6. 技术选型

### 6.1 后端与工作流

- **Python 3.12 + FastAPI + Pydantic**：API、SSE 和严格 schema；
- **LangGraph 作为外层工作流运行时**：支持路由、并行、orchestrator-worker、持久状态和 streaming；
- **自研 Game Mechanism Engine**：收益、信誉、AEIC 指标、验证预算和拓扑策略必须是自己的研究代码，不能藏在框架 prompt 中；
- **不采用纯 ReAct 主循环**：搜推链路中的确定性检索、过滤和验证使用普通 DAG 节点，只有规划与开放判断调用 LLM；
- **不以 AutoGen GraphFlow 为首选**：其图工作流能力符合场景，但当前官方仍标记为实验性。可作为多智能体消息机制对照或后续替换，不宜成为毕业系统唯一底座。

### 6.2 数据与检索

- PostgreSQL：影片、用户、会话、实验配置和审计元数据；
- pgvector：影片与 query embedding，规模足够支撑毕设；
- Redis：查询缓存、会话状态、短期信誉和流式事件；
- BM25/字段检索：初期可用 PostgreSQL FTS 或轻量搜索库，数据规模扩大后再使用 OpenSearch；
- MovieLens + TMDB/公开影片元数据：交互评测与影片事实分离；注意数据许可和 ID 对齐。

### 6.3 可观测性

每个请求必须生成不可变 trace：

- 输入、上下文版本、实验条件；
- Planner 状态、动作、预算与停止理由；
- Agent 可见信息和完整 AEIC；
- 候选来源、过滤、合并和重排过程；
- verifier 证据与判定；
- payoff、信誉变化和最终采纳贡献；
- 首 token、端到端延迟、token、调用数和失败/降级路径；
- prompt/model/data/mechanism/schema 版本哈希。

## 7. 博弈模型落地

### 7.1 Leader

Leader 是机制设计者，不是替所有 Agent 直接做答案的中央 Judge。其策略可写为：

```text
M_t = (agent set, topology, message policy, reward rule,
       verification budget, stop threshold)
```

### 7.2 Follower

Agent 在看到 `M_t` 后选择：

```text
a_i = (effort, independent evidence, imitation, verification,
       revise, dissent, abstain)
```

### 7.3 收益

建议从可解释线性形式开始：

```text
u_i = α · verified_contribution
    + β · evidence_novelty
    + γ · marginal_decision_gain
    + η · constraint_satisfaction
    + ρ · reputation_t
    - λ · compute_cost
    - μ · unsupported_flip
    - ν · copying_penalty
    - ξ · invalid_candidate
```

不要一开始追求复杂神经收益函数。先通过消融确认每项能否识别目标行为，再研究 Leader 的动态参数更新。

### 7.4 共识判定

同时报告三类状态：

- `decision_consensus`：Top-K 或关键候选是否达到采纳阈值；
- `evidence_consensus`：支持证据是否相容且经过验证；
- `residual_dissent`：仍存在的高信誉反对意见。

这能避免把“大家都同意”误当作可靠性。

## 8. 实验体系

### 实验 A：语义共识测量有效性

操纵：独立证据、仅多数立场、复制奖励、反证、错误高信誉 Agent。  
验证：AEIC 指标是否区分证据驱动修正与从众翻转。

### 实验 B：真实博弈响应

条件：无机制、仅结果奖励、静态 AEIC、AEIC 契约、动态 Stackelberg。  
验证：高努力、独立证据、有效验证是否提高；复制和无依据翻转是否下降。

### 实验 C：共识搜推效果

基线：单检索 Agent、单生成 Agent、固定融合、多数投票、静态加权、全量 debate、AEIC 博弈共识。  
指标：NDCG@K、Recall@K、约束满足率、无效候选率、理由事实性、列表多样性和用户效用。

### 实验 D：动态组织与成本

基线：固定 parallel、固定 debate、固定 star、规则路由、学习路由、Oracle。  
指标：质量非劣前提下调用数/token/延迟下降、P95、早停比例和最坏组性能。

### 实验 E：鲁棒性与失效边界

条件：错误多数、高信誉错误者、Agent 缺失、检索污染、生成式幻觉、画像噪声、预算压缩。  
验证：信誉恢复速度、正确少数派保留率、错误级联率和安全降级率。

## 9. 数据集与评测集

推荐采用“两层评测”：

1. **通用推理层**：继续使用 MAS 已有 QA/评审数据，证明机制不是影视专用启发式；
2. **影视搜推层**：
   - MovieLens 序列用于离线推荐；
   - 公开影片元数据用于事实和约束验证；
   - 人工构造并双人标注 300–500 条复杂影视查询，用于多约束搜索、解释和冲突评测；
   - 小规模用户研究只作为补充，不替代离线统计。

复杂查询应覆盖：组合偏好、否定约束、多人共同观看、长尾探索、冷启动、多轮偏好修正和事实型检索。

## 10. 实施顺序

### M0：冻结研究定义（1 周）

- 固定两个研究问题、共识定义、博弈动作和主指标；
- 审计 MAS 可复用模块；
- 形成系统 schema 和实验 manifest。

### M1：单链路可靠搜推基线（2 周）

- 新仓库、FastAPI、React、PostgreSQL/pgvector、Redis；
- 传统检索、向量检索、合法候选和 grounded answer；
- 完整 trace 与离线评测。

### M2：多 Agent + AEIC 共识（3 周）

- 实现角色 Agent、结构化报告和证据账本；
- 接入 MAS 指标、信誉和静态共识；
- 完成单 Agent/投票/静态 AEIC 基线。

### M3：真实博弈机制（3–4 周）

- 机制前置可见、Follower 动作、跨任务信誉；
- 实现动态收益和验证预算；
- 完成实验 A/B。

### M4：动态拓扑与成本（3 周）

- 渐进升级、targeted review、匿名/过滤消息图、早停；
- 完成实验 D/E。

### M5：MovieSID 扩展（2–4 周，可并行准备数据）

- 先受约束 MovieID 选择，再训练轻量 MovieSID；
- 作为异质 Agent 接入，不改变论文贡献主线；
- 完成实验 C 的生成式推荐部分。

### M6：系统演示与论文收口（2 周）

- 用户模式和研究 Debug 模式；
- 固定四条演示路径：简单搜索、复杂推荐、Agent 冲突、错误模型降级；
- 所有论文数字由 frozen question-level/request-level 数据一键生成。

## 11. 与开题报告的对应

无需推翻开题：

- “去中心化语义共识”保留为 Agent 间局部交互和非单一 Judge 决策，但工程上允许 Leader 设计机制；
- “演化博弈驱动收敛”落到跨任务信誉、可见收益和真实策略响应；
- “动态验证成本”扩展为验证预算、动态拓扑和渐进升级；
- “影视推荐系统”从旧式 Agent 调用 Demo 升级为共识机制的真实验证域；
- Swarm/LangChain/Django 等开题中的具体框架属于实现候选，不是必须保持的研究承诺。

如需微调题目，建议只做小改：

**博弈驱动的多智能体语义共识机制及其在影视搜索推荐中的应用**

这比改成“生成式检索研究”更符合现有积累和你的原始研究承诺。

## 12. 技术调研依据

1. Rajput et al., *Recommender Systems with Generative Retrieval*, 2023：Semantic ID 生成式推荐的代表性基础工作。  
   https://arxiv.org/abs/2305.05065
2. Pitre et al., *CONSENSAGENT*, Findings of ACL 2025：多 Agent 谄媚会同时损害可靠性和效率，支持将从众治理作为核心问题。  
   https://aclanthology.org/2025.findings-acl.1141/
3. Choi et al., *An Empirical Study of Group Conformity in Multi-Agent Systems*, Findings of ACL 2025：多数规模与 Agent 能力会影响群体从众。  
   https://aclanthology.org/2025.findings-acl.265/
4. Cui et al., *Free-MAD*, Findings of ACL 2026：利用完整审议轨迹和反从众机制，说明可靠决策不等于强制最终文本一致。  
   https://aclanthology.org/2026.findings-acl.1600/
5. LangGraph 官方 workflows 文档：区分确定性 workflow 与动态 agent，支持 routing、parallelization 和 orchestrator-worker。  
   https://langchain-ai.github.io/langgraph/agents/tools/
6. AutoGen GraphFlow 官方文档：支持顺序、并行、条件和循环图，并区分 execution graph 与 message graph；当前仍标记为 experimental。  
   https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/graph-flow.html
