"""
基于 LangGraph 的多智能体辩论框架
================================================================

支持两种辩论模式：
1. 非结构化辩论：自由文本辩论
2. AEIC结构化辩论：遵循Assumptions-Evidence-Inference-Conclusion结构

支持两种共识机制：
1. 单智能体辩论：一个agent内部的自我辩论
2. 多智能体辩论：多个agent间的Stackelberg共识

辩论流程：
- Leader宣布辩论主题和激励参数
- Follower agents进行序贯辩论
- 共识引擎评估辩论质量
- 权重自适应更新
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from .consensus import ConsensusEngine
from .stackelberg import StackelbergConsensusGame, AgentBid


# ─────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────

class DebateState(TypedDict):
    """辩论状态"""
    topic: str
    leader_params: Dict[str, float]  # Stackelberg激励参数
    current_round: int
    max_rounds: int
    agents: List[Dict[str, Any]]  # agent信息
    debate_history: List[Dict[str, Any]]  # 辩论历史
    consensus_scores: Dict[str, float]  # 共识评分
    aeic_structured: bool  # 是否使用AEIC结构
    multi_agent: bool  # 是否多智能体
    current_agent_idx: int  # 当前发言agent
    final_consensus: Optional[Dict[str, Any]]


class AEICStructure(TypedDict):
    """AEIC结构化输出"""
    assumptions: str
    evidence: List[str]
    inference: str
    conclusion: str


# ─────────────────────────────────────────────────────
# 辩论节点
# ─────────────────────────────────────────────────────

class DebateNode:
    """辩论节点基类"""

    def __init__(self, llm: ChatOpenAI, agent_info: Dict[str, Any]):
        self.llm = llm
        self.agent_info = agent_info

    def generate_response(self, state: DebateState) -> str:
        """生成辩论响应"""
        raise NotImplementedError


class UnstructuredDebateNode(DebateNode):
    """非结构化辩论节点"""

    def generate_response(self, state: DebateState) -> str:
        """生成自由文本辩论响应"""

        # 构建上下文
        context = f"辩论主题：{state['topic']}\n\n"

        if state['debate_history']:
            context += "辩论历史：\n"
            for i, entry in enumerate(state['debate_history'][-3:]):  # 只看最近3轮
                context += f"Agent {entry['agent_id']}: {entry['response'][:200]}...\n"

        # 激励参数影响
        theta = state['leader_params'].get(self.agent_info['id'], 0.5)
        creativity_boost = "请发挥创造性，提供新颖观点。" if theta > 0.7 else ""
        conservative_boost = "请基于已有证据进行严谨推理。" if theta < 0.3 else ""

        prompt = f"""{context}

你是 {self.agent_info['name']}，一位{self.agent_info['role']}专家。
{creativity_boost}{conservative_boost}

请针对上述辩论主题发表你的观点。要求：
1. 基于你的专业背景提供见解
2. 考虑其他agent的观点，但不必完全同意
3. 保持建设性和逻辑性
4. 控制在200-400字以内

你的观点："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()


class AEICStructuredDebateNode(DebateNode):
    """AEIC结构化辩论节点"""

    def generate_response(self, state: DebateState) -> AEICStructure:
        """生成AEIC结构化辩论响应"""

        # 构建上下文
        context = f"辩论主题：{state['topic']}\n\n"

        if state['debate_history']:
            context += "辩论历史：\n"
            for entry in state['debate_history'][-2:]:  # 只看最近2轮
                aeic = entry.get('aeic_response', {})
                context += f"Agent {entry['agent_id']} 结论：{aeic.get('conclusion', '')[:100]}...\n"

        # 激励参数影响
        theta = state['leader_params'].get(self.agent_info['id'], 0.5)
        focus_assumptions = "重点关注前提假设的合理性" if theta > 0.6 else ""
        focus_evidence = "重点提供可靠证据支撑" if 0.4 <= theta <= 0.6 else ""
        focus_inference = "重点进行严谨逻辑推理" if theta < 0.4 else ""

        prompt = f"""{context}

你是 {self.agent_info['name']}，一位{self.agent_info['role']}专家。

请以AEIC结构回答辩论主题：
{focus_assumptions}{focus_evidence}{focus_inference}

要求每个部分：
- Assumptions：列出你的核心前提假设（2-3个关键点）
- Evidence：提供支撑证据（3-5个具体事实/数据）
- Inference：基于证据进行逻辑推理
- Conclusion：得出你的结论

请用JSON格式输出，结构如下：
{{
    "assumptions": "前提假设内容",
    "evidence": ["证据1", "证据2", "证据3"],
    "inference": "推理过程内容",
    "conclusion": "结论内容"
}}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        # 解析JSON响应
        try:
            import json
            content = response.content.strip()
            # 清理可能的markdown代码块标记
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            aeic_data = json.loads(content)
            return AEICStructure(**aeic_data)
        except Exception as e:
            # 如果解析失败，返回默认结构
            return AEICStructure(
                assumptions="解析响应失败，使用默认假设",
                evidence=["默认证据1", "默认证据2"],
                inference="解析响应失败，使用默认推理",
                conclusion="解析响应失败，使用默认结论"
            )


# ─────────────────────────────────────────────────────
# 共识评估节点
# ─────────────────────────────────────────────────────

class ConsensusEvaluator:
    """共识评估节点"""

    def __init__(self, consensus_engine: ConsensusEngine):
        self.consensus_engine = consensus_engine

    def evaluate(self, state: DebateState) -> Dict[str, Any]:
        """评估辩论共识"""

        if not state['debate_history']:
            return {"consensus_score": 0.0, "decision": "NO_DEBATE"}

        # 准备节点记录用于共识评估
        node_records = []

        for entry in state['debate_history']:
            if state['aeic_structured']:
                # AEIC结构化记录
                record = {
                    "node_id": entry["agent_id"],
                    "assumptions": entry["aeic_response"]["assumptions"],
                    "evidence": "\n".join(entry["aeic_response"]["evidence"]),
                    "inference": entry["aeic_response"]["inference"],
                    "conclusion": entry["aeic_response"]["conclusion"]
                }
            else:
                # 非结构化记录（模拟AEIC结构）
                response = entry["response"]
                record = {
                    "node_id": entry["agent_id"],
                    "assumptions": f"基于辩论内容：{response[:100]}",
                    "evidence": [f"观点：{response[100:200]}"],
                    "inference": f"推理过程：{response[200:300]}",
                    "conclusion": f"结论：{response[300:400]}"
                }
            node_records.append(record)

        # 评估共识
        result = self.consensus_engine.evaluate_consensus(node_records)

        return {
            "consensus_score": result["avg_similarity"],
            "decision": result["decision"],
            "utility": result["utility"],
            "pairwise": result["pairwise"],
            "n_nodes": result["n_nodes"]
        }


# ─────────────────────────────────────────────────────
# Stackelberg协调器
# ─────────────────────────────────────────────────────

class StackelbergCoordinator:
    """Stackelberg协调器 - 实现真正的序贯决策"""

    def __init__(self, consensus_game: StackelbergConsensusGame):
        self.consensus_game = consensus_game
        self.decision_stage = "leader_commitment"

    def announce_leader_params(self, state: DebateState) -> Dict[str, float]:
        """Leader宣布激励参数（第一阶段）"""

        # 使用真正的序贯Stackelberg决策
        task_context = {
            'complexity': len(state['topic'].split()) / 100,  # 基于主题复杂度
            'urgency': 0.5  # 默认中等紧急程度
        }

        # 调用序贯决策的第一阶段
        leader_params_array = self.consensus_game.leader_commitment(task_context)

        # 转换为agent_id映射
        params = {}
        for i, agent in enumerate(state['agents']):
            if i < len(leader_params_array):
                params[agent['id']] = float(leader_params_array[i])

        return params

    def process_follower_responses(self, state: DebateState) -> Dict[str, Any]:
        """处理Follower响应（第二阶段）"""

        # 构造AgentBid对象
        bids = []
        for agent in state['agents']:
            bid = AgentBid(
                agent_id=agent['id'],
                port=agent.get('port', 8000),
                model=agent.get('model', 'gpt-4'),
                role=agent['role'],
                quality_promise=0.8,  # 默认质量承诺
                cost_per_task=10.0,   # 默认成本
                capacity=0.5,         # 默认容量
                past_performance=agent.get('performance', 0.8)
            )
            bids.append(bid)

        # 调用序贯决策的第二阶段
        task_context = {
            'complexity': len(state['topic'].split()) / 100,
            'urgency': 0.5
        }

        follower_responses = self.consensus_game.follower_best_response(bids, task_context)

        return follower_responses

    def update_params_based_on_consensus(self, state: DebateState, consensus_result: Dict) -> Dict[str, float]:
        """基于共识结果更新参数（第三阶段）"""

        # 获取Follower响应
        follower_responses = self.process_follower_responses(state)

        # 调用序贯决策的第三阶段
        updated_params_array = self.consensus_game.leader_optimization_update(follower_responses, consensus_result)

        # 转换为agent_id映射
        params = {}
        for i, agent in enumerate(state['agents']):
            if i < len(updated_params_array):
                params[agent['id']] = float(updated_params_array[i])

        return params

    def run_sequential_process(self, state: DebateState, consensus_result: Dict = None) -> Dict[str, Any]:
        """运行完整的序贯Stackelberg过程"""

        # 构造AgentBid对象
        bids = []
        for agent in state['agents']:
            bid = AgentBid(
                agent_id=agent['id'],
                port=agent.get('port', 8000),
                model=agent.get('model', 'gpt-4'),
                role=agent['role'],
                quality_promise=0.8,
                cost_per_task=10.0,
                capacity=0.5,
                past_performance=agent.get('performance', 0.8)
            )
            bids.append(bid)

        # 运行完整的序贯过程
        task_context = {
            'complexity': len(state['topic'].split()) / 100,
            'urgency': 0.5
        }

        result = self.consensus_game.run_sequential_stackelberg(bids, task_context, max_iterations=3)

        # 转换为agent_id映射的最终参数
        final_params = {}
        for i, agent in enumerate(state['agents']):
            if i < len(result['final_params']):
                final_params[agent['id']] = float(result['final_params'][i])

        return {
            'final_params': final_params,
            'iterations': result['iterations'],
            'converged': result['converged']
        }


# ─────────────────────────────────────────────────────
# LangGraph辩论流程
# ─────────────────────────────────────────────────────

def create_debate_graph(
    llm: ChatOpenAI,
    consensus_engine: ConsensusEngine,
    consensus_game: StackelbergConsensusGame,
    aeic_structured: bool = True,
    multi_agent: bool = True
) -> StateGraph:
    """创建辩论图"""

    # 初始化组件
    coordinator = StackelbergCoordinator(consensus_game)
    evaluator = ConsensusEvaluator(consensus_engine)

    # 创建辩论节点
    debate_nodes = {}

    def create_debate_node_for_agent(agent_info: Dict):
        """为每个agent创建辩论节点"""
        if aeic_structured:
            return AEICStructuredDebateNode(llm, agent_info)
        else:
            return UnstructuredDebateNode(llm, agent_info)

    # 节点函数
    def leader_announce(state: DebateState) -> DebateState:
        """Leader宣布参数"""
        new_params = coordinator.announce_leader_params(state)
        state['leader_params'] = new_params
        state['current_round'] = 1
        return state

    def agent_debate(state: DebateState) -> DebateState:
        """Agent辩论"""
        current_agent = state['agents'][state['current_agent_idx']]
        node = create_debate_node_for_agent(current_agent)

        if aeic_structured:
            response = node.generate_response(state)
            entry = {
                "agent_id": current_agent['id'],
                "round": state['current_round'],
                "aeic_response": response
            }
        else:
            response = node.generate_response(state)
            entry = {
                "agent_id": current_agent['id'],
                "round": state['current_round'],
                "response": response
            }

        state['debate_history'].append(entry)
        return state

    def evaluate_consensus(state: DebateState) -> DebateState:
        """评估共识"""
        if len(state['debate_history']) >= len(state['agents']):
            consensus_result = evaluator.evaluate(state)
            state['consensus_scores'] = {
                "score": consensus_result["consensus_score"],
                "decision": consensus_result["decision"],
                "utility": consensus_result["utility"]
            }
        return state

    def update_leader_params(state: DebateState) -> DebateState:
        """Leader更新参数（序贯决策）"""
        if state['consensus_scores']:
            new_params = coordinator.update_params_based_on_consensus(
                state, state['consensus_scores']
            )
            state['leader_params'] = new_params
        return state

    def check_termination(state: DebateState) -> str:
        """检查是否终止"""
        if state['current_round'] >= state['max_rounds']:
            state['final_consensus'] = state['consensus_scores']
            return "end"

        # 检查共识质量
        if state['consensus_scores'].get('score', 0) > 0.8:
            state['final_consensus'] = state['consensus_scores']
            return "end"

        # 进入下一轮
        state['current_round'] += 1
        state['current_agent_idx'] = (state['current_agent_idx'] + 1) % len(state['agents'])
        return "continue"

    # 构建图
    workflow = StateGraph(DebateState)

    # 添加节点
    workflow.add_node("leader_announce", leader_announce)
    workflow.add_node("agent_debate", agent_debate)
    workflow.add_node("evaluate_consensus", evaluate_consensus)
    workflow.add_node("update_params", update_leader_params)

    # 添加边
    workflow.add_edge("leader_announce", "agent_debate")
    workflow.add_edge("agent_debate", "evaluate_consensus")
    workflow.add_edge("evaluate_consensus", "update_params")
    workflow.add_conditional_edges(
        "update_params",
        check_termination,
        {
            "continue": "agent_debate",
            "end": END
        }
    )

    # 设置入口
    workflow.set_entry_point("leader_announce")

    return workflow


# ─────────────────────────────────────────────────────
# 辩论执行器
# ─────────────────────────────────────────────────────

class DebateExecutor:
    """辩论执行器"""

    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0.7)
        self.consensus_engine = ConsensusEngine()
        self.consensus_game = StackelbergConsensusGame(leader_port=8000)

    def run_debate(
        self,
        topic: str,
        agents: List[Dict[str, Any]],
        aeic_structured: bool = True,
        multi_agent: bool = True,
        max_rounds: int = 3
    ) -> Dict[str, Any]:
        """运行辩论"""

        # 创建图
        workflow = create_debate_graph(
            self.llm,
            self.consensus_engine,
            self.consensus_game,
            aeic_structured=aeic_structured,
            multi_agent=multi_agent
        )

        # 编译图
        app = workflow.compile()

        # 初始状态
        initial_state = DebateState(
            topic=topic,
            leader_params={},
            current_round=0,
            max_rounds=max_rounds,
            agents=agents,
            debate_history=[],
            consensus_scores={},
            aeic_structured=aeic_structured,
            multi_agent=multi_agent,
            current_agent_idx=0,
            final_consensus=None
        )

        # 运行辩论
        try:
            result = app.invoke(initial_state)
            return {
                "success": True,
                "final_consensus": result.get('final_consensus'),
                "debate_history": result.get('debate_history', []),
                "leader_params_history": result.get('leader_params', {}),
                "total_rounds": result.get('current_round', 0)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "debate_history": initial_state.get('debate_history', [])
            }


# ─────────────────────────────────────────────────────
# 实验对比框架
# ─────────────────────────────────────────────────────

class DebateExperimentRunner:
    """辩论实验运行器 - 4种实验配置"""

    def __init__(self):
        self.executor = DebateExecutor()

    def run_single_experiment(
        self,
        topic: str,
        aeic_structured: bool,
        multi_agent: bool,
        agents: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行单个实验"""

        # 默认agents
        if agents is None:
            if multi_agent:
                agents = [
                    {"id": "agent_1", "name": "张教授", "role": "经济学家"},
                    {"id": "agent_2", "name": "李博士", "role": "社会学家"},
                    {"id": "agent_3", "name": "王研究员", "role": "政策分析师"}
                ]
            else:
                agents = [
                    {"id": "agent_1", "name": "辩论者", "role": "综合分析师"}
                ]

        # 运行辩论
        result = self.executor.run_debate(
            topic=topic,
            agents=agents,
            aeic_structured=aeic_structured,
            multi_agent=multi_agent,
            max_rounds=3
        )

        return {
            "config": {
                "aeic_structured": aeic_structured,
                "multi_agent": multi_agent,
                "topic": topic
            },
            "result": result,
            "metrics": self._calculate_metrics(result)
        }

    def run_all_experiments(self, topic: str) -> Dict[str, Any]:
        """运行4种实验配置"""

        configs = [
            {"name": "非结构化_单智能体", "aeic": False, "multi": False},
            {"name": "非结构化_多智能体", "aeic": False, "multi": True},
            {"name": "AEIC结构化_单智能体", "aeic": True, "multi": False},
            {"name": "AEIC结构化_多智能体", "aeic": True, "multi": True}
        ]

        results = {}
        for config in configs:
            print(f"运行实验：{config['name']}")
            result = self.run_single_experiment(
                topic=topic,
                aeic_structured=config['aeic'],
                multi_agent=config['multi']
            )
            results[config['name']] = result

        return results

    def _calculate_metrics(self, result: Dict) -> Dict[str, Any]:
        """计算实验指标"""

        if not result.get('success', False):
            return {"error": result.get('error', 'Unknown error')}

        consensus = result.get('final_consensus', {})
        history = result.get('debate_history', [])

        return {
            "consensus_score": consensus.get('score', 0),
            "decision": consensus.get('decision', 'UNKNOWN'),
            "utility": consensus.get('utility', 0),
            "total_rounds": result.get('total_rounds', 0),
            "total_utterances": len(history),
            "avg_response_length": self._avg_response_length(history)
        }

    def _avg_response_length(self, history: List[Dict]) -> float:
        """计算平均响应长度"""
        if not history:
            return 0

        total_length = 0
        count = 0

        for entry in history:
            if 'response' in entry:
                total_length += len(entry['response'])
                count += 1
            elif 'aeic_response' in entry:
                aeic = entry['aeic_response']
                total_length += (
                    len(aeic.get('assumptions', '')) +
                    len(str(aeic.get('evidence', []))) +
                    len(aeic.get('inference', '')) +
                    len(aeic.get('conclusion', ''))
                )
                count += 1

        return total_length / count if count > 0 else 0


if __name__ == "__main__":
    # 测试运行
    runner = DebateExperimentRunner()

    # 运行单实验
    result = runner.run_single_experiment(
        topic="人工智能是否会取代人类工作？",
        aeic_structured=True,
        multi_agent=True
    )

    print("单实验结果：")
    print(result)

    # 运行全对比实验
    all_results = runner.run_all_experiments(
        topic="人工智能是否会取代人类工作？"
    )

    print("\n全对比实验结果：")
    for config_name, result in all_results.items():
        print(f"{config_name}: {result['metrics']}")