"""
Expert Recruiter - AgentVerse专家招募模块

职责：
1. 根据任务需求动态生成专家角色
2. 从Registry发现匹配的Agent
3. 基于信任分优先选择高质量Agent
"""

import httpx
from typing import List, Dict, Optional
import json

class ExpertRecruiter:
    """专家招募器"""
    
    def __init__(self, registry_url: str, llm=None):
        self.registry_url = registry_url
        self.llm = llm
        
        # 预定义的角色模板
        self.role_templates = {
            "assumption_validator": {
                "name": "前提验证专家",
                "description": "验证逻辑前提的合理性",
                "required_capability": "semantic_analysis"
            },
            "evidence_gatherer": {
                "name": "证据收集专家",
                "description": "收集和验证支撑证据",
                "required_capability": "evidence_collection"
            },
            "inference_analyst": {
                "name": "推理分析专家",
                "description": "分析逻辑推理路径",
                "required_capability": "logic_verification"
            },
            "conclusion_reviewer": {
                "name": "结论审查专家",
                "description": "审查最终结论的合理性",
                "required_capability": "semantic_analysis"
            },
            "solver": {
                "name": "求解者",
                "description": "提出初始解决方案",
                "required_capability": "problem_solving"
            },
            "reviewer": {
                "name": "评审者",
                "description": "评审和反馈解决方案",
                "required_capability": "critical_thinking"
            }
        }
    
    async def recruit_experts(
        self, 
        task_desc: str, 
        current_utility: Optional[float] = None,
        trust_scores: Optional[Dict[int, float]] = None
    ) -> List[Dict]:
        """
        根据任务动态招募专家
        
        Args:
            task_desc: 任务描述
            current_utility: 当前博弈收益（用于动态调整）
            trust_scores: Agent信任分
        
        Returns:
            List[Dict]: 招募的专家列表
        """
        
        # Step 1: 分析任务需求
        required_roles = await self._analyze_task_requirements(
            task_desc, 
            current_utility
        )
        
        print(f"\n{'='*60}")
        print(f"🔍 专家招募分析")
        print(f"  任务: {task_desc}")
        print(f"  需要角色: {[r['name'] for r in required_roles]}")
        print(f"{'='*60}")
        
        # Step 2: 从Registry发现可用Agent
        available_agents = await self._discover_available_agents()
        
        if not available_agents:
            print("⚠️  警告: 没有可用的Agent")
            return []
        
        # Step 3: 匹配Agent到角色
        recruited = await self._match_agents_to_roles(
            required_roles,
            available_agents,
            trust_scores
        )
        
        print(f"\n✓ 成功招募 {len(recruited)} 个专家:")
        for agent in recruited:
            print(f"  - Agent {agent['port']}: {agent['assigned_role']}")
        
        return recruited
    
    async def _analyze_task_requirements(
        self, 
        task_desc: str, 
        current_utility: Optional[float]
    ) -> List[Dict]:
        """分析任务需求，生成所需角色"""
        
        # 如果有LLM，使用智能分析
        if self.llm:
            return await self._llm_based_analysis(task_desc, current_utility)
        
        # 否则使用规则基础分析
        return self._rule_based_analysis(task_desc, current_utility)
    
    async def _llm_based_analysis(
        self, 
        task_desc: str, 
        current_utility: Optional[float]
    ) -> List[Dict]:
        """基于LLM的智能分析"""
        
        prompt = f"""
        任务描述: {task_desc}
        当前收益: {current_utility if current_utility else '首次分析'}
        
        请分析此任务需要哪些专家角色。可选角色包括：
        1. 前提验证专家 (assumption_validator)
        2. 证据收集专家 (evidence_gatherer)
        3. 推理分析专家 (inference_analyst)
        4. 结论审查专家 (conclusion_reviewer)
        5. 求解者 (solver)
        6. 评审者 (reviewer)
        
        请输出JSON格式：
        {{
            "roles": ["role_id1", "role_id2", ...],
            "reason": "选择这些角色的原因"
        }}
        """
        
        try:
            response = await self.llm.agenerate(prompt)
            result = json.loads(response)
            
            return [
                self.role_templates[role_id] 
                for role_id in result['roles']
                if role_id in self.role_templates
            ]
        except Exception as e:
            print(f"⚠️  LLM分析失败: {e}，使用规则基础方法")
            return self._rule_based_analysis(task_desc, current_utility)
    
    def _rule_based_analysis(
        self, 
        task_desc: str, 
        current_utility: Optional[float]
    ) -> List[Dict]:
        """基于规则的分析"""
        
        # 默认配置：1个Solver + 2个Reviewer
        roles = [
            self.role_templates["solver"],
            self.role_templates["reviewer"]
        ]
        
        # 如果是信用审核任务，增加证据收集专家
        if "信用" in task_desc or "审核" in task_desc or "贷款" in task_desc:
            roles.append(self.role_templates["evidence_gatherer"])
        
        # 如果当前收益低，增加更多审查角色
        if current_utility and current_utility < 30:
            roles.append(self.role_templates["inference_analyst"])
            roles.append(self.role_templates["conclusion_reviewer"])
        
        return roles
    
    async def _discover_available_agents(self) -> Dict:
        """从Registry发现可用Agent"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.registry_url}/discover",
                    timeout=5.0
                )
                result = response.json()
                return result['agents']
        except Exception as e:
            print(f"⚠️  无法连接到Registry: {e}")
            return {}
    
    async def _match_agents_to_roles(
        self,
        required_roles: List[Dict],
        available_agents: Dict,
        trust_scores: Optional[Dict[int, float]]
    ) -> List[Dict]:
        """匹配Agent到角色"""
        
        recruited = []
        
        for role in required_roles:
            # 查找能够胜任此角色的Agent
            candidates = []
            
            for agent_addr, agent_info in available_agents.items():
                # 检查能力匹配
                required_cap = role.get('required_capability')
                if required_cap and required_cap in agent_info.get('capabilities', {}):
                    
                    # 提取端口
                    port = int(agent_addr.split(':')[1])
                    
                    # 获取信任分
                    trust = trust_scores.get(port, 50) if trust_scores else 50
                    
                    candidates.append({
                        "port": port,
                        "agent_addr": agent_addr,
                        "agent_info": agent_info,
                        "trust": trust,
                        "role": role
                    })
            
            # 按信任分排序，选择最优
            if candidates:
                candidates.sort(key=lambda x: x['trust'], reverse=True)
                best_candidate = candidates[0]
                
                recruited.append({
                    "port": best_candidate['port'],
                    "agent_addr": best_candidate['agent_addr'],
                    "model": best_candidate['agent_info']['model'],
                    "assigned_role": role['name'],
                    "trust_score": best_candidate['trust']
                })
        
        return recruited
    
    def get_role_description(self, role_id: str) -> str:
        """获取角色描述"""
        if role_id in self.role_templates:
            return self.role_templates[role_id]['description']
        return "未知角色"
