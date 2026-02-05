"""
Coordination Engine - 分布式多智能体协调引擎

职责：
1. 编排整个AgentVerse流程
2. 协调专家招募、任务规划、共识博弈
3. 管理Agent生命周期
4. 同步博弈状态
"""

import httpx
import asyncio
from typing import Dict, List, Optional
import time

from task_planner import TaskPlanner
from expert_recruiter import ExpertRecruiter
from memory import MemoryManager


class CoordinationEngine:
    """协调引擎（在任务发起者的Agent中运行）"""
    
    def __init__(
        self, 
        agent_port: int,
        registry_url: str,
        memory_manager: Optional[MemoryManager] = None,
        llm=None
    ):
        self.agent_port = agent_port
        self.registry_url = registry_url
        self.llm = llm
        
        # 子模块
        self.planner = TaskPlanner(llm)
        self.recruiter = ExpertRecruiter(registry_url, llm)
        self.memory = memory_manager or MemoryManager()
        
        # 当前任务状态
        self.current_task_id = None
        self.task_state = {}
    
    async def orchestrate(self, user_request: Dict) -> Dict:
        """
        主编排流程 - AgentVerse四阶段
        
        Args:
            user_request: {
                "description": "任务描述",
                "goal": "目标",
                "data": {...}
            }
        
        Returns:
            Dict: 最终结果
        """
        
        print(f"\n{'='*70}")
        print(f"🚀 协调引擎启动")
        print(f"   发起者: Agent {self.agent_port}")
        print(f"   任务: {user_request['description']}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # ===== Stage 1: 任务规划 (Task Planning) =====
        task_plan = await self.planner.decompose(user_request)
        
        # ===== Stage 2: 专家招募 (Expert Recruitment) =====
        trust_scores = self.memory.consensus_memory.trust_scores
        recruited_experts = await self.recruiter.recruit_experts(
            user_request['description'],
            current_utility=None,
            trust_scores=trust_scores
        )
        
        if not recruited_experts:
            print("❌ 招募失败：没有可用的专家")
            return {"status": "failed", "reason": "No experts available"}
        
        # ===== 发布任务到Registry =====
        task_id = await self._publish_task_to_registry(user_request)
        self.current_task_id = task_id
        
        # ===== Stage 3: 协作决策 (Collaborative Decision-Making) =====
        print(f"\n{'='*60}")
        print(f"🤝 开始协作决策（垂直结构）")
        print(f"{'='*60}")
        
        # 选择Solver和Reviewers
        solver = None
        reviewers = []
        
        for expert in recruited_experts:
            if 'solver' in expert['assigned_role'].lower() or '求解' in expert['assigned_role']:
                solver = expert
            else:
                reviewers.append(expert)
        
        if not solver:
            # 如果没有明确的Solver，第一个专家作为Solver
            solver = recruited_experts[0]
            reviewers = recruited_experts[1:]
        
        # 执行协作决策循环
        consensus_result = await self._collaborative_decision(
            solver=solver,
            reviewers=reviewers,
            task_data=user_request.get('data', {})
        )
        
        # ===== Stage 4: 评估与反馈 (Evaluation) =====
        feedback = await self._evaluate_result(
            consensus_result, 
            user_request['goal']
        )
        
        # ===== 存储到Registry =====
        await self._store_consensus_to_registry(
            task_id=task_id,
            result=consensus_result,
            participants=[e['port'] for e in recruited_experts]
        )
        
        # ===== 更新记忆 =====
        for expert in recruited_experts:
            self.memory.record_task_result(
                agent_id=expert['port'],
                task_desc=user_request['description'],
                result=consensus_result,
                utility=consensus_result.get('utility', 0)
            )
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✓ 任务完成")
        print(f"   决策: {consensus_result.get('decision', 'unknown')}")
        print(f"   收益U: {consensus_result.get('utility', 0):.2f}")
        print(f"   轮次: {consensus_result.get('rounds', 0)}")
        print(f"   耗时: {elapsed:.2f}秒")
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "task_id": task_id,
            "result": consensus_result,
            "feedback": feedback,
            "elapsed_time": elapsed
        }
    
    async def _publish_task_to_registry(self, user_request: Dict) -> int:
        """发布任务到Registry"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.registry_url}/publish_task",
                    json={
                        "description": user_request['description'],
                        "initiator": self.agent_port,
                        "requirements": {
                            "goal": user_request.get('goal', ''),
                            "data": user_request.get('data', {})
                        }
                    },
                    timeout=5.0
                )
                result = response.json()
                return result['task_id']
        except Exception as e:
            print(f"⚠️  无法发布任务到Registry: {e}")
            return -1
    
    async def _collaborative_decision(
        self,
        solver: Dict,
        reviewers: List[Dict],
        task_data: Dict
    ) -> Dict:
        """
        协作决策阶段（垂直结构：Solver-Reviewer循环）
        
        流程：
        1. Solver提出初始方案
        2. Reviewers评审并反馈
        3. 计算语义共识收益
        4. 如果未达成ESS，Solver根据反馈修正
        5. 重复直到达成共识或超过最大轮次
        """
        
        MAX_ROUNDS = 5
        ESS_THRESHOLD = 55
        
        state = {
            "current_proposal": None,
            "round_idx": 0,
            "utility_history": [],
            "decision": "in_progress"
        }
        
        for round_num in range(1, MAX_ROUNDS + 1):
            print(f"\n--- 第 {round_num} 轮协作 ---")
            
            state['round_idx'] = round_num
            
            # Step 1: Solver提出/修正方案
            if round_num == 1:
                # 初始方案
                proposal = await self._request_solver_proposal(
                    solver, 
                    task_data
                )
            else:
                # 根据反馈修正
                proposal = await self._request_solver_refinement(
                    solver,
                    state['current_proposal'],
                    state.get('critique_feedback', '')
                )
            
            state['current_proposal'] = proposal
            
            # Step 2: Reviewers评审
            critiques = []
            for reviewer in reviewers:
                critique = await self._request_reviewer_feedback(
                    reviewer,
                    proposal
                )
                critiques.append(critique)
            
            # 合并评审意见
            combined_critique = "\n".join(critiques) if critiques else "无反馈"
            state['critique_feedback'] = combined_critique
            
            # Step 3: 计算语义共识收益（使用consensus.py）
            utility = await self._compute_semantic_utility(proposal, critiques)
            state['utility_history'].append(utility)
            
            print(f"   收益U = {utility:.2f}")
            
            # Step 4: 检查是否达成ESS
            if utility > ESS_THRESHOLD:
                state['decision'] = "ESS_CONSENSUS"
                print(f"   ✓ 达成ESS共识！")
                break
            elif round_num >= MAX_ROUNDS:
                state['decision'] = "MAX_ROUNDS_STOP"
                print(f"   ⚠️  达到最大轮次")
                break
            else:
                print(f"   → 继续优化...")
        
        return {
            "proposal": state['current_proposal'],
            "utility": state['utility_history'][-1] if state['utility_history'] else 0,
            "rounds": state['round_idx'],
            "decision": state['decision'],
            "history": state['utility_history']
        }
    
    async def _request_solver_proposal(
        self, 
        solver: Dict, 
        task_data: Dict
    ) -> Dict:
        """请求Solver提出初始方案"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{solver['port']}/solve",
                    json={"task_data": task_data},
                    timeout=10.0
                )
                return response.json()
        except Exception as e:
            print(f"⚠️  Solver {solver['port']} 无响应: {e}")
            # 返回默认方案
            return {
                "assumptions": task_data.get('assumptions', '未知'),
                "evidence": task_data.get('evidence', []),
                "inference": "初步分析",
                "conclusion": "待定"
            }
    
    async def _request_solver_refinement(
        self,
        solver: Dict,
        current_proposal: Dict,
        feedback: str
    ) -> Dict:
        """请求Solver根据反馈修正方案"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{solver['port']}/refine",
                    json={
                        "current_proposal": current_proposal,
                        "feedback": feedback
                    },
                    timeout=10.0
                )
                return response.json()
        except Exception as e:
            print(f"⚠️  Solver修正失败: {e}")
            return current_proposal  # 保持原方案
    
    async def _request_reviewer_feedback(
        self,
        reviewer: Dict,
        proposal: Dict
    ) -> str:
        """请求Reviewer评审方案"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{reviewer['port']}/review",
                    json={"proposal": proposal},
                    timeout=10.0
                )
                result = response.json()
                return result.get('feedback', '无具体意见')
        except Exception as e:
            print(f"⚠️  Reviewer {reviewer['port']} 无响应: {e}")
            return "评审未完成"
    
    async def _compute_semantic_utility(
        self, 
        proposal: Dict, 
        critiques: List[str]
    ) -> float:
        """
        计算语义共识收益
        这里应该调用 consensus.py 的逻辑
        """
        # TODO: 集成真实的共识引擎
        # 目前使用简化版本
        
        # 基础分数
        base_score = 50
        
        # 如果proposal有完整的AEIC结构，加分
        if all(k in proposal for k in ['assumptions', 'evidence', 'inference', 'conclusion']):
            base_score += 10
        
        # 如果有证据，加分
        evidence = proposal.get('evidence', [])
        if isinstance(evidence, list) and len(evidence) > 0:
            base_score += len(evidence) * 3
        
        # 如果critique是正面的，加分
        for critique in critiques:
            if any(word in critique for word in ['充分', '合理', '完善', '通过']):
                base_score += 5
            if any(word in critique for word in ['不足', '缺少', '问题', '不合理']):
                base_score -= 5
        
        return min(100, max(0, base_score))
    
    async def _evaluate_result(self, result: Dict, goal: str) -> Dict:
        """评估最终结果"""
        return {
            "goal_achieved": result.get('utility', 0) > 55,
            "quality_score": result.get('utility', 0),
            "recommendations": "继续保持" if result.get('utility', 0) > 55 else "需要改进"
        }
    
    async def _store_consensus_to_registry(
        self,
        task_id: int,
        result: Dict,
        participants: List[int]
    ):
        """存储共识结果到Registry"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.registry_url}/store_consensus",
                    json={
                        "task_id": task_id,
                        "initiator": self.agent_port,
                        "result": result['proposal'],
                        "participants": participants,
                        "utility": result['utility'],
                        "rounds": result['rounds']
                    },
                    timeout=5.0
                )
        except Exception as e:
            print(f"⚠️  无法存储共识到Registry: {e}")
