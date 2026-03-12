"""
Task Planner - 任务分解与规划

职责：
1. 将复杂任务分解为子任务
2. 构建任务依赖图
3. 生成执行计划
"""

import json
from typing import List, Dict, Optional
import networkx as nx

class TaskPlanner:
    """任务规划器"""
    
    def __init__(self, llm=None):
        self.llm = llm
    
    async def decompose(self, user_request: Dict) -> 'TaskPlan':
        """
        将用户请求分解为可执行的子任务
        
        Args:
            user_request: {
                "description": "任务描述",
                "goal": "目标",
                "data": {...}  # 任务相关数据
            }
        
        Returns:
            TaskPlan: 任务计划对象
        """
        
        print(f"\n{'='*60}")
        print(f"📋 任务分解中...")
        print(f"  描述: {user_request['description']}")
        print(f"  目标: {user_request.get('goal', '未指定')}")
        print(f"{'='*60}")
        
        # 如果有LLM，使用智能分解
        if self.llm:
            plan = await self._llm_based_decomposition(user_request)
        else:
            plan = self._rule_based_decomposition(user_request)
        
        print(f"\n✓ 任务分解完成:")
        print(f"  子任务数: {len(plan.subtasks)}")
        print(f"  复杂度: {plan.complexity}")
        for i, subtask in enumerate(plan.subtasks, 1):
            print(f"  {i}. {subtask['description']}")
        
        return plan
    
    async def _llm_based_decomposition(self, user_request: Dict) -> 'TaskPlan':
        """基于LLM的智能分解"""
        
        prompt = f"""
        用户请求: {user_request['description']}
        目标: {user_request.get('goal', '未指定')}
        
        请将此任务分解为可独立执行的子任务，每个子任务需要：
        1. 明确的输入和输出
        2. 可验证的完成标准
        3. 所需的专家角色
        
        输出JSON格式：
        {{
            "subtasks": [
                {{
                    "id": 1, 
                    "description": "子任务描述", 
                    "required_role": "solver/reviewer", 
                    "priority": 1
                }},
                ...
            ],
            "dependencies": [[1,2], ...],  // 任务依赖关系 [前置任务, 后续任务]
            "estimated_complexity": "high/medium/low"
        }}
        """
        
        try:
            response = await self.llm.agenerate(prompt)
            plan_data = json.loads(response)
            return self._build_task_plan(plan_data, user_request)
        except Exception as e:
            print(f"⚠️  LLM分解失败: {e}，使用规则基础方法")
            return self._rule_based_decomposition(user_request)
    
    def _rule_based_decomposition(self, user_request: Dict) -> 'TaskPlan':
        """基于规则的分解"""
        
        description = user_request['description']
        
        # 默认分解为3个阶段
        subtasks = [
            {
                "id": 1,
                "description": f"分析任务需求: {description}",
                "required_role": "solver",
                "priority": 1,
                "input": user_request.get('data', {}),
                "output": "initial_proposal"
            },
            {
                "id": 2,
                "description": "评审初始方案",
                "required_role": "reviewer",
                "priority": 2,
                "input": "initial_proposal",
                "output": "feedback"
            },
            {
                "id": 3,
                "description": "根据反馈优化方案",
                "required_role": "solver",
                "priority": 3,
                "input": "feedback",
                "output": "final_solution"
            }
        ]
        
        # 任务依赖：1->2->3
        dependencies = [[1, 2], [2, 3]]
        
        # 估计复杂度
        complexity = "medium"
        if "复杂" in description or "困难" in description:
            complexity = "high"
        elif "简单" in description or "基础" in description:
            complexity = "low"
        
        plan_data = {
            "subtasks": subtasks,
            "dependencies": dependencies,
            "estimated_complexity": complexity
        }
        
        return self._build_task_plan(plan_data, user_request)
    
    def _build_task_plan(self, plan_data: Dict, user_request: Dict) -> 'TaskPlan':
        """构建任务计划对象"""
        
        # 构建任务依赖图
        task_graph = nx.DiGraph()
        
        for task in plan_data['subtasks']:
            task_graph.add_node(
                task['id'],
                description=task['description'],
                role=task['required_role'],
                priority=task.get('priority', 1)
            )
        
        for dep in plan_data.get('dependencies', []):
            task_graph.add_edge(dep[0], dep[1])
        
        return TaskPlan(
            subtasks=plan_data['subtasks'],
            graph=task_graph,
            complexity=plan_data['estimated_complexity'],
            original_request=user_request
        )


class TaskPlan:
    """任务计划对象"""
    
    def __init__(
        self, 
        subtasks: List[Dict], 
        graph: nx.DiGraph, 
        complexity: str,
        original_request: Dict
    ):
        self.subtasks = subtasks
        self.graph = graph
        self.complexity = complexity
        self.original_request = original_request
    
    def get_executable_tasks(self) -> List[Dict]:
        """获取当前可执行的任务（无前置依赖的任务）"""
        executable = []
        
        for task in self.subtasks:
            task_id = task['id']
            # 检查是否有未完成的前置任务
            predecessors = list(self.graph.predecessors(task_id))
            
            if not predecessors:
                executable.append(task)
        
        return executable
    
    def mark_task_completed(self, task_id: int):
        """标记任务完成"""
        # 从图中移除已完成的任务
        if task_id in self.graph:
            self.graph.remove_node(task_id)
    
    def is_completed(self) -> bool:
        """检查是否所有任务都完成"""
        return len(self.graph.nodes) == 0
    
    def get_progress(self) -> float:
        """获取完成进度"""
        total = len(self.subtasks)
        remaining = len(self.graph.nodes)
        return (total - remaining) / total if total > 0 else 0
    
    def visualize(self) -> str:
        """可视化任务依赖图"""
        lines = ["任务依赖图:"]
        
        for task in self.subtasks:
            task_id = task['id']
            if task_id in self.graph:
                predecessors = list(self.graph.predecessors(task_id))
                successors = list(self.graph.successors(task_id))
                
                line = f"  [{task_id}] {task['description']}"
                if predecessors:
                    line += f" (依赖: {predecessors})"
                if successors:
                    line += f" (后续: {successors})"
                
                lines.append(line)
        
        return "\n".join(lines)
