"""
AgentVerse框架 - 多智能体协作的完整流程

实现AgentVerse论文中的四个阶段：
1. Expert Recruitment - 专家招募
2. Collaborative Decision-Making - 协作决策
3. Action Execution - 行动执行
4. Evaluation - 评估反馈

与Stackelberg博弈的集成：
- 外层：Stackelberg博弈（任务分配）
- 内层：语义共识博弈（质量验证）
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

from consensus.consensus import ConsensusEngine, Operators
from consensus.stackelberg import StackelbergScheduler, AgentBid

class CommunicationStructure(Enum):
    """通信结构"""
    HORIZONTAL = "horizontal"  # 水平沟通：民主协作
    VERTICAL = "vertical"      # 垂直沟通：求解者-评审者
    DYNAMIC = "dynamic"        # 动态切换

@dataclass
class AgentVerseState:
    """AgentVerse状态（类似LangGraph的State）"""
    session_id: str
    round_idx: int
    
    # 任务信息
    task_description: str
    task_requirements: Dict
    
    # 当前提案
    current_proposal: Dict
    
    # 历史
    utility_history: List[float]
    feedback_history: List[str]
    
    # 决策
    decision: str  # 'ESS_CONSENSUS', 'AUDIT_REQUIRED', 'REFINE_REQUIRED'
    
    # 参与者
    participants: List[str]
    leader_id: str
    
    # 配置
    budget_limit: int = 5
    ess_threshold: float = 55.0

class ExpertRecruiter:
    """
    专家招募模块（AgentVerse Stage 1）
    
    根据任务需求和历史表现，动态选择专家
    """
    
    def __init__(self):
        self.trust_scores = {}  # 信任分数
    
    def recruit_experts(
        self, 
        task: Dict, 
        available_agents: Dict[str, Dict],
        history: List[Dict] = None
    ) -> List[str]:
        """
        招募专家
        
        策略：
        1. 根据任务需求筛选角色
        2. 根据历史表现排序
        3. 选择top-k个Agent
        """
        required_roles = task.get('requirements', {}).get('roles_needed', [])
        
        # 筛选角色匹配的Agent
        candidates = []
        for agent_id, agent_info in available_agents.items():
            if agent_info['role'] in required_roles or not required_roles:
                trust = self.trust_scores.get(agent_id, 50.0)  # 默认50分
                candidates.append({
                    'agent_id': agent_id,
                    'trust': trust,
                    'role': agent_info['role']
                })
        
        # 按信任分排序
        candidates.sort(key=lambda x: x['trust'], reverse=True)
        
        # 选择专家（至少1个solver + 1个reviewer）
        selected = []
        solver_count = 0
        reviewer_count = 0
        
        for candidate in candidates:
            if candidate['role'] == 'solver' and solver_count < 1:
                selected.append(candidate['agent_id'])
                solver_count += 1
            elif candidate['role'] == 'reviewer' and reviewer_count < 2:
                selected.append(candidate['agent_id'])
                reviewer_count += 1
            
            if solver_count >= 1 and reviewer_count >= 2:
                break
        
        print(f"\n👥 专家招募完成:")
        print(f"   需求角色: {required_roles}")
        print(f"   候选人数: {len(candidates)}")
        print(f"   选中: {len(selected)} 个Agent")
        for agent_id in selected:
            role = next(c['role'] for c in candidates if c['agent_id'] == agent_id)
            trust = self.trust_scores.get(agent_id, 50.0)
            print(f"     - {agent_id} ({role}, 信任分: {trust:.1f})")
        
        return selected
    
    def update_trust(self, agent_id: str, utility: float, ess_threshold: float):
        """
        更新信任分
        
        规则：
        - 达成ESS（U > threshold）：+5分
        - 有贡献但未达ESS（0 < U < threshold）：+2分
        - 负贡献（U < 0）：-3分
        """
        current_trust = self.trust_scores.get(agent_id, 50.0)
        
        if utility > ess_threshold:
            delta = 5
        elif utility > 0:
            delta = 2
        else:
            delta = -3
        
        self.trust_scores[agent_id] = max(0, min(100, current_trust + delta))
        print(f"   {agent_id} 信任分: {current_trust:.1f} → {self.trust_scores[agent_id]:.1f} ({delta:+.1f})")

class CollaborativeDecisionMaker:
    """
    协作决策模块（AgentVerse Stage 2）
    
    支持两种通信结构：
    - Horizontal：水平沟通（头脑风暴）
    - Vertical：垂直沟通（求解者-评审者迭代）
    """
    
    def __init__(self, structure: CommunicationStructure = CommunicationStructure.VERTICAL):
        self.structure = structure
        self.consensus_engine = ConsensusEngine()
    
    async def vertical_decision(
        self, 
        state: AgentVerseState, 
        solver_proposal: Dict,
        reviewer_feedbacks: List[Dict]
    ) -> Dict:
        """
        垂直沟通：求解者提案，评审者反馈，迭代优化
        """
        # 计算语义共识收益
        utilities = []
        for feedback in reviewer_feedbacks:
            result = self.consensus_engine.evaluate_game(
                solver_proposal,
                feedback
            )
            utilities.append(result['utility'])
        
        avg_utility = sum(utilities) / len(utilities) if utilities else 0
        
        # 生成改进建议
        critique = self.generate_critique(solver_proposal, reviewer_feedbacks, avg_utility)
        
        return {
            'utility': avg_utility,
            'critique': critique,
            'individual_utilities': utilities
        }
    
    def generate_critique(
        self, 
        proposal: Dict, 
        feedbacks: List[Dict],
        utility: float
    ) -> str:
        """
        生成结构化的改进建议
        """
        critiques = []
        
        # 检查各个维度
        if utility < 20:
            critiques.append("整体一致性严重不足，建议重新评估前提假设")
        elif utility < 40:
            critiques.append("证据支持不足，建议补充更多证据")
        elif utility < 55:
            critiques.append("逻辑推理存在漏洞，建议加强推理链条")
        else:
            critiques.append("已达成ESS共识，建议finalize")
        
        # 具体维度反馈
        for i, feedback in enumerate(feedbacks, 1):
            if 'missing_evidence' in str(feedback):
                critiques.append(f"Reviewer {i}: 缺少关键证据")
        
        return " | ".join(critiques)

class AgentVerseFramework:
    """
    完整的AgentVerse框架
    
    集成：
    1. 专家招募
    2. 协作决策（垂直/水平）
    3. 行动执行
    4. 评估反馈
    5. Stackelberg调度（外层）
    """
    
    def __init__(self):
        self.recruiter = ExpertRecruiter()
        self.decision_maker = CollaborativeDecisionMaker()
        self.stackelberg = None  # 动态创建
    
    async def execute_agentverse_loop(
        self,
        task: Dict,
        available_agents: Dict[str, Dict],
        leader_port: int,
        enable_stackelberg: bool = True
    ) -> AgentVerseState:
        """
        执行完整的AgentVerse循环
        
        流程：
        1. 专家招募
        2. (可选) Stackelberg分配
        3. 协作决策（迭代）
        4. 评估与反馈
        """
        print(f"\n{'='*60}")
        print("🚀 AgentVerse框架启动")
        print(f"{'='*60}")
        
        # 初始化状态
        state = AgentVerseState(
            session_id=f"session-{leader_port}-{task['id']}",
            round_idx=0,
            task_description=task['description'],
            task_requirements=task.get('requirements', {}),
            current_proposal={},
            utility_history=[],
            feedback_history=[],
            decision="REFINE_REQUIRED",
            participants=[],
            leader_id=f"127.0.0.1:{leader_port}"
        )
        
        # ===== Stage 1: 专家招募 =====
        print(f"\n📋 Stage 1: 专家招募")
        recruited = self.recruiter.recruit_experts(
            task, 
            available_agents,
            history=None
        )
        state.participants = recruited
        
        # ===== Stage 1.5: Stackelberg分配（可选）=====
        if enable_stackelberg and len(recruited) > 1:
            print(f"\n📋 Stage 1.5: Stackelberg任务分配")
            self.stackelberg = StackelbergScheduler(leader_port)
            
            # 构建竞标（简化：假设参与者都提交了竞标）
            bids = []
            for agent_id in recruited:
                agent_info = available_agents[agent_id]
                port = int(agent_id.split(':')[1])
                
                # 模拟竞标参数（实际应该从Agent获取）
                bid = AgentBid(
                    agent_id=agent_id,
                    port=port,
                    model=agent_info['model'],
                    role=agent_info['role'],
                    quality_promise=0.85,
                    cost_per_task=20,
                    capacity=0.5,
                    past_performance=self.recruiter.trust_scores.get(agent_id, 50) / 100
                )
                bids.append(bid)
            
            # 执行Stackelberg博弈
            stackelberg_result = self.stackelberg.execute_stackelberg_game(bids)
            allocation = stackelberg_result['allocation']
        else:
            allocation = {agent: 1.0 / len(recruited) for agent in recruited}
        
        # ===== Stage 2: 协作决策（迭代）=====
        print(f"\n📋 Stage 2: 协作决策（迭代）")
        
        max_rounds = state.budget_limit
        for round_idx in range(max_rounds):
            state.round_idx = round_idx + 1
            print(f"\n🔄 Round {state.round_idx}/{max_rounds}")
            
            # 模拟提案和反馈（实际应该调用Agent API）
            solver_proposal = self.generate_mock_proposal(state, round_idx)
            reviewer_feedbacks = self.generate_mock_feedbacks(state, round_idx)
            
            state.current_proposal = solver_proposal
            
            # 垂直决策
            decision_result = await self.decision_maker.vertical_decision(
                state,
                solver_proposal,
                reviewer_feedbacks
            )
            
            utility = decision_result['utility']
            critique = decision_result['critique']
            
            state.utility_history.append(utility)
            state.feedback_history.append(critique)
            
            print(f"   收益 U = {utility:.2f}")
            print(f"   反馈: {critique}")
            
            # ===== Stage 4: 评估 =====
            if utility > state.ess_threshold:
                state.decision = "ESS_CONSENSUS"
                print(f"   ✅ 达成ESS共识！")
                break
            elif state.round_idx >= max_rounds:
                state.decision = "BUDGET_EXHAUSTED"
                print(f"   ⏱️  预算耗尽，强制结束")
                break
            else:
                state.decision = "REFINE_REQUIRED"
                print(f"   🔧 需要继续优化")
        
        # ===== 更新信任分 =====
        final_utility = state.utility_history[-1] if state.utility_history else 0
        print(f"\n📊 更新信任分:")
        for agent_id in state.participants:
            self.recruiter.update_trust(agent_id, final_utility, state.ess_threshold)
        
        print(f"\n{'='*60}")
        print("✅ AgentVerse框架执行完成")
        print(f"{'='*60}")
        print(f"  最终决策: {state.decision}")
        print(f"  总轮次: {state.round_idx}")
        print(f"  最终收益: {final_utility:.2f}")
        print(f"  参与者: {len(state.participants)} 个Agent")
        print(f"{'='*60}\n")
        
        return state
    
    def generate_mock_proposal(self, state: AgentVerseState, round_idx: int) -> Dict:
        """生成模拟提案（实际应该调用Solver Agent）"""
        # 随着轮次增加，提案质量提升
        quality_factor = min(1.0, 0.5 + round_idx * 0.15)
        
        return {
            'id': f"proposal-{round_idx}",
            'assumptions': f'高信用模型V{3 + round_idx}',
            'evidence': ['资产证明', '流水单', '纳税记录'][:1 + round_idx],
            'inference': '多维交叉验证' if round_idx > 1 else '初步验证',
            'conclusion': '批准' if quality_factor > 0.7 else '待定'
        }
    
    def generate_mock_feedbacks(self, state: AgentVerseState, round_idx: int) -> List[Dict]:
        """生成模拟反馈（实际应该调用Reviewer Agent）"""
        # 标准反馈模板
        return [
            {
                'id': 'reviewer-1',
                'assumptions': '高信用模型V3',
                'evidence': ['资产证明', '流水单', '纳税记录', '征信报告'],
                'inference': '多维交叉验证',
                'conclusion': '批准'
            },
            {
                'id': 'reviewer-2',
                'assumptions': '高信用模型V3',
                'evidence': ['资产证明', '流水单', '纳税记录'],
                'inference': '多维交叉验证',
                'conclusion': '批准'
            }
        ]

# ============ 示例使用 ============

if __name__ == "__main__":
    # 模拟可用Agent
    available_agents = {
        "127.0.0.1:8002": {
            "model": "qwen-plus",
            "role": "solver",
            "status": "online"
        },
        "127.0.0.1:8003": {
            "model": "deepseek-chat",
            "role": "reviewer",
            "status": "online"
        },
        "127.0.0.1:8004": {
            "model": "qwen-plus",
            "role": "reviewer",
            "status": "online"
        }
    }
    
    # 模拟任务
    task = {
        'id': 1,
        'description': '审核企业贷款申请',
        'requirements': {
            'roles_needed': ['solver', 'reviewer'],
            'min_agents': 2
        }
    }
    
    # 执行框架
    framework = AgentVerseFramework()
    
    async def run():
        result = await framework.execute_agentverse_loop(
            task=task,
            available_agents=available_agents,
            leader_port=8001,
            enable_stackelberg=True
        )
        
        print("\n📈 最终状态:")
        print(f"  Session ID: {result.session_id}")
        print(f"  决策: {result.decision}")
        print(f"  收益历史: {result.utility_history}")
    
    asyncio.run(run())
