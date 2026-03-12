"""
Agent Node - 分布式多智能体节点

功能：
1. 可以加入/退出网络（注册到Registry）
2. 可以发起任务（成为临时协调者）
3. 可以响应任务（作为Solver或Reviewer参与）
4. P2P通信进行博弈
"""

import os
import sys
import asyncio
import argparse
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import httpx
from typing import Dict, Optional

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordination_engine import CoordinationEngine
from memory import MemoryManager
from consensus.consensus import ConsensusEngine, Operators

app = FastAPI()

# ================= 数据模型 =================

class TaskInvitation(BaseModel):
    task_id: int
    description: str
    initiator: int

class ProposalRequest(BaseModel):
    task_data: Dict

class RefinementRequest(BaseModel):
    current_proposal: Dict
    feedback: str

class ReviewRequest(BaseModel):
    proposal: Dict

# ================= Agent类 =================

class DistributedAgent:
    """分布式Agent节点"""
    
    def __init__(
        self, 
        port: int, 
        model: str, 
        role: str, 
        registry_url: str,
        llm=None
    ):
        self.port = port
        self.model = model
        self.role = role  # 'solver', 'reviewer', 'initiator'
        self.registry_url = registry_url
        self.llm = llm
        
        # 子系统
        self.memory_manager = MemoryManager()
        self.consensus_engine = ConsensusEngine()
        self.coordination_engine = CoordinationEngine(
            agent_port=port,
            registry_url=registry_url,
            memory_manager=self.memory_manager,
            llm=llm
        )
        
        # 状态
        self.is_leader = False  # 是否是当前任务的Leader
        self.current_tasks = {}  # 正在参与的任务
        
        print(f"\n{'='*60}")
        print(f"🤖 Agent初始化完成")
        print(f"   端口: {self.port}")
        print(f"   模型: {self.model}")
        print(f"   角色: {self.role}")
        print(f"{'='*60}\n")
    
    async def join_network(self):
        """加入网络（注册到Registry）"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.registry_url}/register",
                    json={
                        "host": "127.0.0.1",
                        "port": self.port,
                        "model": self.model,
                        "role": self.role,
                        "capabilities": self.get_capabilities()
                    },
                    timeout=5.0
                )
                result = response.json()
                
                print(f"✓ 成功加入网络")
                print(f"  网络规模: {len(result['network'])} 个Agent")
                print(f"  其他Agent: {[a for a in result['network'] if str(self.port) not in a]}")
                
                return result
        except Exception as e:
            print(f"❌ 无法连接到Registry: {e}")
            print(f"   请确保Registry运行在 {self.registry_url}")
            return None
    
    def get_capabilities(self) -> Dict:
        """声明自己的能力"""
        capabilities = {}
        
        if self.role in ['solver', 'initiator']:
            capabilities['problem_solving'] = True
            capabilities['semantic_analysis'] = True
            capabilities['evidence_collection'] = True
        
        if self.role in ['reviewer', 'initiator']:
            capabilities['critical_thinking'] = True
            capabilities['logic_verification'] = True
            capabilities['semantic_analysis'] = True
        
        return capabilities
    
    async def initiate_task(self, task_desc: str, task_data: Dict, goal: str = None):
        """
        发起任务（成为临时Leader）
        
        Args:
            task_desc: 任务描述
            task_data: 任务数据（AEIC格式）
            goal: 任务目标
        """
        self.is_leader = True
        
        user_request = {
            "description": task_desc,
            "goal": goal or "达成ESS共识",
            "data": task_data
        }
        
        try:
            # 通过协调引擎执行完整的AgentVerse流程
            result = await self.coordination_engine.orchestrate(user_request)
            return result
        finally:
            self.is_leader = False
    
    async def solve(self, task_data: Dict) -> Dict:
        """
        作为Solver提出方案
        
        Args:
            task_data: {
                "assumptions": "...",
                "evidence": [...],
                "inference": "...",
                "conclusion": "..."
            }
        
        Returns:
            提案（AEIC格式）
        """
        print(f"\n[Solver {self.port}] 正在分析任务...")
        
        # 如果有LLM，使用智能分析
        if self.llm:
            proposal = await self._llm_solve(task_data)
        else:
            # 使用规则基础方法
            proposal = self._rule_based_solve(task_data)
        
        # 记忆：添加到短期记忆
        memory = self.memory_manager.get_agent_memory(self.port)
        memory.add_to_short_term(
            f"提出方案: {proposal.get('conclusion', 'unknown')}",
            role="solver"
        )
        
        return proposal
    
    async def _llm_solve(self, task_data: Dict) -> Dict:
        """使用LLM生成方案"""
        # TODO: 实现LLM调用
        return self._rule_based_solve(task_data)
    
    def _rule_based_solve(self, task_data: Dict) -> Dict:
        """基于规则生成方案"""
        # 保留原有数据，进行一些增强
        proposal = {
            "assumptions": task_data.get('assumptions', '基础评估模型'),
            "evidence": task_data.get('evidence', []),
            "inference": task_data.get('inference', '初步验证'),
            "conclusion": task_data.get('conclusion', '待定')
        }
        
        # 如果证据太少，添加建议
        if isinstance(proposal['evidence'], list) and len(proposal['evidence']) < 3:
            proposal['inference'] = "证据不足，建议补充更多材料"
        
        return proposal
    
    async def refine(self, current_proposal: Dict, feedback: str) -> Dict:
        """
        根据Reviewer反馈修正方案
        
        Args:
            current_proposal: 当前方案
            feedback: 反馈意见
        
        Returns:
            修正后的方案
        """
        print(f"\n[Solver {self.port}] 根据反馈修正方案...")
        print(f"   反馈: {feedback[:100]}...")
        
        refined = current_proposal.copy()
        
        # 解析反馈，进行相应修正
        if "证据不足" in feedback or "补充" in feedback:
            # 增加证据
            current_evidence = refined.get('evidence', [])
            if isinstance(current_evidence, list):
                refined['evidence'] = current_evidence + ["补充材料1", "补充材料2"]
        
        if "逻辑" in feedback or "推理" in feedback:
            refined['inference'] = "多维交叉验证逻辑"
        
        if "前提" in feedback or "假设" in feedback:
            refined['assumptions'] = "高信用模型V3"
        
        # 记忆
        memory = self.memory_manager.get_agent_memory(self.port)
        memory.add_to_short_term(
            f"修正方案（基于反馈）",
            role="solver",
            metadata={"feedback_length": len(feedback)}
        )
        
        return refined
    
    async def review(self, proposal: Dict) -> Dict:
        """
        作为Reviewer评审方案
        
        Args:
            proposal: 待评审的方案
        
        Returns:
            {"feedback": "评审意见", "score": 0-100}
        """
        print(f"\n[Reviewer {self.port}] 正在评审方案...")
        
        # 使用共识引擎计算语义相似度
        # 这里需要一个参考标准，我们使用"理想方案"作为对比
        ideal_proposal = {
            "assumptions": "高信用模型V3",
            "evidence": ["资产证明", "纳税记录", "流水单", "征信报告", "社保缴纳"],
            "inference": "多维交叉验证逻辑",
            "conclusion": "批准"
        }
        
        # 计算各层相似度
        result = self.consensus_engine.evaluate_game(proposal, ideal_proposal)
        
        # 生成反馈
        feedback_parts = []
        
        if result['sim_a'] < 0.7:
            feedback_parts.append(f"前提假设不够完善（相似度{result['sim_a']:.2f}），建议使用更高级的评估模型")
        
        if result['sim_e'] < 0.5:
            feedback_parts.append(f"证据严重不足（重叠率{result['sim_e']:.2f}），请补充银行流水、纳税记录等")
        
        if result['sim_i'] < 0.6:
            feedback_parts.append(f"推理逻辑有待加强（一致性{result['sim_i']:.2f}），建议采用多维交叉验证")
        
        if result['sim_c'] < 0.5:
            feedback_parts.append(f"结论对齐度较低（{result['sim_c']:.2f}），请重新检查推理链条")
        
        if not feedback_parts:
            feedback_parts.append("方案质量良好，各项指标均达标")
        
        feedback_text = "；".join(feedback_parts)
        
        # 记忆
        memory = self.memory_manager.get_agent_memory(self.port)
        memory.add_to_short_term(
            f"评审完成，收益U={result['utility']:.2f}",
            role="reviewer"
        )
        
        print(f"   收益U = {result['utility']:.2f}")
        print(f"   反馈: {feedback_text}")
        
        return {
            "feedback": feedback_text,
            "score": result['total_score'] * 100,
            "utility": result['utility'],
            "details": result
        }


# ================= 全局Agent实例 =================
agent_instance: Optional[DistributedAgent] = None

# ================= FastAPI路由 =================

@app.post("/task_invitation")
async def on_task_invitation(invitation: TaskInvitation):
    """收到任务邀请（P2P通信）"""
    print(f"\n📬 收到任务邀请")
    print(f"   任务ID: {invitation.task_id}")
    print(f"   描述: {invitation.description}")
    print(f"   发起者: Agent {invitation.initiator}")
    
    # 决定是否参与（目前总是接受）
    return {
        "status": "accept",
        "agent": agent_instance.port,
        "model": agent_instance.model,
        "role": agent_instance.role
    }

@app.post("/solve")
async def solve_endpoint(request: ProposalRequest):
    """Solver端点：提出初始方案"""
    proposal = await agent_instance.solve(request.task_data)
    return proposal

@app.post("/refine")
async def refine_endpoint(request: RefinementRequest):
    """Solver端点：修正方案"""
    refined = await agent_instance.refine(
        request.current_proposal,
        request.feedback
    )
    return refined

@app.post("/review")
async def review_endpoint(request: ReviewRequest):
    """Reviewer端点：评审方案"""
    review_result = await agent_instance.review(request.proposal)
    return review_result

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "port": agent_instance.port,
        "model": agent_instance.model,
        "role": agent_instance.role,
        "is_leader": agent_instance.is_leader
    }

@app.get("/stats")
async def stats():
    """统计信息"""
    memory = agent_instance.memory_manager.get_agent_memory(agent_instance.port)
    trust = agent_instance.memory_manager.consensus_memory.get_trust(agent_instance.port)
    
    return {
        "agent_id": agent_instance.port,
        "model": agent_instance.model,
        "role": agent_instance.role,
        "trust_score": trust,
        "memory_stats": memory.get_stats()
    }

# ================= 启动 =================

@app.on_event("startup")
async def startup_event():
    """启动时注册到网络"""
    await agent_instance.join_network()

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时注销"""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{agent_instance.registry_url}/unregister/{agent_instance.port}",
                timeout=2.0
            )
        print(f"\n✓ Agent {agent_instance.port} 已退出网络")
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Agent Node")
    parser.add_argument("--port", type=int, required=True, help="Agent端口")
    parser.add_argument("--model", type=str, required=True, help="模型名称 (qwen/deepseek)")
    parser.add_argument("--role", type=str, required=True, help="角色 (solver/reviewer/initiator)")
    parser.add_argument("--registry", type=str, default="http://127.0.0.1:9000", help="Registry地址")
    args = parser.parse_args()
    
    # 创建Agent实例
    agent_instance = DistributedAgent(
        port=args.port,
        model=args.model,
        role=args.role,
        registry_url=args.registry,
        llm=None  # TODO: 集成LLM
    )
    
    print(f"\n{'='*70}")
    print(f"🚀 Agent节点启动")
    print(f"{'='*70}")
    print(f"   端口: {args.port}")
    print(f"   模型: {args.model}")
    print(f"   角色: {args.role}")
    print(f"   Registry: {args.registry}")
    print(f"{'='*70}\n")
    
    # 启动FastAPI服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning"  # 减少日志输出
    )
