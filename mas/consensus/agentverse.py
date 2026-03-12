"""
AgentVerse 框架 - 多节点协作编排

四个阶段：招募 → 决策 → 执行 → 评估

架构说明：
  每个节点是独立进程（通过 agent_node.py 启动），通过 HTTP 互相通信。
  此编排器运行在发起任务的节点内部，负责协调其他节点参与共识。
  共识评估使用 ConsensusEngine.evaluate_consensus()，支持任意 N 个节点。
"""

from typing import Dict, List
from dataclasses import dataclass
from .consensus import ConsensusEngine
from .stackelberg import StackelbergConsensusGame, AgentBid


@dataclass
class AgentVerseState:
    """单次协作任务的运行状态"""
    session_id: str
    round_idx: int
    task_description: str
    current_proposals: List[Dict]          # 所有参与节点的当前提案
    utility_history: List[float]
    participants: List[str]                # 参与节点的 node_id 列表
    leader_id: str
    budget_limit: int = 5


class NodeRecruiter:
    """Stage 1: 招募参与节点"""

    def __init__(self):
        self.trust_scores: Dict[str, float] = {}

    def recruit_nodes(self, task: Dict, available_nodes: Dict[str, Dict]) -> List[str]:
        """从可用节点中选出最合适的参与者"""
        required_roles = task.get("requirements", {}).get("roles_needed", [])

        candidates = []
        for node_id, info in available_nodes.items():
            if info["role"] in required_roles or not required_roles:
                trust = self.trust_scores.get(node_id, 50.0)
                candidates.append({"node_id": node_id, "trust": trust, "role": info["role"]})

        candidates.sort(key=lambda x: x["trust"], reverse=True)
        return [c["node_id"] for c in candidates[:4]]   # 最多招募 4 个节点

    def update_trust(self, node_id: str, utility: float):
        current = self.trust_scores.get(node_id, 50.0)
        delta   = 5 if utility > 55 else (2 if utility > 0 else -3)
        self.trust_scores[node_id] = max(0, min(100, current + delta))


class CollaborativeDecisionMaker:
    """Stage 2: 多节点协作决策"""

    def __init__(self):
        self.consensus_engine = ConsensusEngine()

    def evaluate_proposals(self, node_proposals: Dict[str, Dict]) -> Dict:
        """
        评估多个节点的提案，计算全网共识

        Args:
            node_proposals: { node_id: aeic_record_dict, ... }

        Returns:
            共识评估结果（含 pairwise 相似度矩阵、avg_similarity、decision）
        """
        if len(node_proposals) < 2:
            only = list(node_proposals.values())
            return {"decision": "SINGLE_NODE", "result": only[0] if only else {}}

        # 构建节点记录列表（带 node_id 字段）
        node_records = [
            {"node_id": nid, **rec}
            for nid, rec in node_proposals.items()
        ]

        result = self.consensus_engine.evaluate_consensus(node_records)

        return {
            "decision": result["decision"],
            "utility":  result["utility"],
            "avg_similarity": result["avg_similarity"],
            "pairwise": result["pairwise"],
            "n_nodes":  result["n_nodes"],
            "result":   result,
        }


class ActionExecutor:
    """Stage 3: 执行 Stackelberg 任务分配"""

    def execute_allocation(
        self,
        game: StackelbergConsensusGame,
        bids: List[AgentBid],
    ) -> Dict:
        return game.execute_stackelberg_game(bids, total_workload=1.0)


class EvaluationReporter:
    """Stage 4: 评估并记录"""

    def evaluate(self, state: AgentVerseState, result: Dict) -> Dict:
        state.utility_history.append(result.get("leader_utility", 0))
        return {
            "session_id":  state.session_id,
            "round":       state.round_idx,
            "utility":     result.get("leader_utility", 0),
            "avg_utility": sum(state.utility_history) / len(state.utility_history),
            "n_nodes":     len(state.participants),
        }


class AgentVerseOrchestrator:
    """编排器 - 协调四个阶段的多节点协作流程"""

    def __init__(self):
        self.recruiter      = NodeRecruiter()
        self.decision_maker = CollaborativeDecisionMaker()
        self.executor       = ActionExecutor()
        self.evaluator      = EvaluationReporter()

    def run_workflow(
        self,
        task: Dict,
        available_nodes: Dict[str, Dict],
        node_proposals: Dict[str, Dict],   # { node_id: aeic_record }
    ) -> Dict:
        """
        运行完整的多节点共识工作流

        Args:
            task:             任务描述字典
            available_nodes:  可用节点信息 { node_id: { role, ... } }
            node_proposals:   各节点提交的 AEIC 提案 { node_id: aeic_dict }
        """

        # Stage 1: 招募节点
        selected = self.recruiter.recruit_nodes(task, available_nodes)

        # 只保留被选中节点的提案
        active_proposals = {
            nid: rec for nid, rec in node_proposals.items()
            if nid in selected
        }

        # Stage 2: 多节点协作决策
        decision_result = self.decision_maker.evaluate_proposals(active_proposals)

        # Stage 3: 执行 Stackelberg 分配
        bids = [
            AgentBid(nid, 8000 + i, "deepseek", "solver", 0.85, 20, 0.6)
            for i, nid in enumerate(selected)
        ]
        game             = StackelbergConsensusGame(leader_port=8001)
        execution_result = self.executor.execute_allocation(game, bids)

        # Stage 4: 评估
        state = AgentVerseState(
            session_id="demo",
            round_idx=1,
            task_description=task.get("description", ""),
            current_proposals=list(active_proposals.values()),
            utility_history=[],
            participants=selected,
            leader_id=selected[0] if selected else "unknown",
        )
        evaluation = self.evaluator.evaluate(state, execution_result)

        # 更新信任分
        for nid in selected:
            self.recruiter.update_trust(nid, decision_result.get("utility", 0))

        return {
            "selected_nodes":  selected,
            "n_nodes":         len(selected),
            "decision":        decision_result["decision"],
            "avg_similarity":  decision_result.get("avg_similarity", 0),
            "pairwise":        decision_result.get("pairwise", {}),
            "execution":       execution_result,
            "evaluation":      evaluation,
        }


if __name__ == "__main__":
    orchestrator = AgentVerseOrchestrator()

    task = {
        "description": "多节点贷款风险共识评估",
        "requirements": {"roles_needed": ["solver", "reviewer"]},
    }

    # 模拟 4 个节点各自提交 AEIC 提案
    available_nodes = {
        "node_0": {"role": "solver"},
        "node_1": {"role": "reviewer"},
        "node_2": {"role": "solver"},
        "node_3": {"role": "reviewer"},
    }

    node_proposals = {
        "node_0": {
            "assumptions": "申请人信用评分高，历史还款记录良好",
            "evidence": ["征信报告A+", "流水单12个月", "资产证明"],
            "inference": "偿债能力充足，风险可控",
            "conclusion": "批准贷款",
        },
        "node_1": {
            "assumptions": "客户资质优秀，财务状况稳定",
            "evidence": ["信用记录优良", "银行流水", "不动产证明"],
            "inference": "还款能力有保障，符合放贷标准",
            "conclusion": "同意发放贷款",
        },
        "node_2": {
            "assumptions": "标准风控模型下申请人评分达标",
            "evidence": ["央行征信", "收入证明", "社保记录"],
            "inference": "各维度指标均满足准入条件",
            "conclusion": "核准贷款申请",
        },
        "node_3": {
            "assumptions": "申请人整体资质符合授信要求",
            "evidence": ["征信报告", "工资流水", "住房证明"],
            "inference": "综合评估通过，建议放款",
            "conclusion": "批准",
        },
    }

    result = orchestrator.run_workflow(task, available_nodes, node_proposals)

    print(f"\n✅ 多节点共识工作流完成")
    print(f"   参与节点: {result['selected_nodes']}")
    print(f"   共识决策: {result['decision']}")
    print(f"   平均相似度: {result['avg_similarity']:.4f}")
    print(f"\n   两两相似度:")
    for pair, sims in result["pairwise"].items():
        print(f"     {pair}: {sims['total']:.4f}")
