"""
Stackelberg博弈调度器

实现层级博弈机制：
- Leader（任务发起者）：优化流量/任务分配
- Follower（其他Agent）：根据分配选择最优策略

理论基础：
Von Stackelberg (1934) 的序贯博弈模型
通过逆向归纳法求解均衡
"""

from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class AgentBid:
    """Agent的竞标信息"""
    agent_id: str
    port: int
    model: str
    role: str
    
    # 竞标参数
    quality_promise: float  # 承诺的质量 [0, 1]
    cost_per_task: float    # 每个任务的成本
    capacity: float         # 容量 [0, 1]
    
    # 历史表现（用于信任评估）
    past_performance: float = 0.8  # 默认0.8

class StackelbergScheduler:
    """
    Stackelberg博弈调度器
    
    任务发起者（Leader）使用此类来优化任务分配
    """
    
    def __init__(self, leader_port: int):
        self.leader_port = leader_port
        self.history = []
    
    def predict_follower_response(self, bid: AgentBid, allocated_workload: float) -> Dict:
        """
        预测Follower的最优反应（Stackelberg第二步）
        
        Follower收到workload后，会根据自身能力选择投入策略
        """
        # 如果分配的workload超过容量，质量会下降
        if allocated_workload > bid.capacity:
            # 过载惩罚
            overload_penalty = bid.capacity / allocated_workload
            actual_quality = bid.quality_promise * overload_penalty
        else:
            # 正常情况
            actual_quality = bid.quality_promise
        
        # 考虑历史表现（信任因子）
        actual_quality *= bid.past_performance
        
        # Follower的成本
        actual_cost = bid.cost_per_task * allocated_workload
        
        # Follower的收益（质量×工作量 - 成本）
        follower_utility = actual_quality * allocated_workload * 100 - actual_cost
        
        return {
            'actual_quality': actual_quality,
            'actual_cost': actual_cost,
            'follower_utility': follower_utility,
            'overloaded': allocated_workload > bid.capacity
        }
    
    def optimize_allocation(self, bids: List[AgentBid], total_workload: float = 1.0) -> Dict[str, float]:
        """
        Leader的优化问题（Stackelberg第一步）
        
        目标：最大化总体质量，同时最小化成本
        约束：
        1. 总workload = total_workload
        2. 每个Agent的workload ≤ capacity
        3. workload ≥ 0
        
        使用贪心算法求解（简化版）
        """
        if not bids:
            return {}
        
        # 计算每个Agent的"性价比"（质量/成本比）
        efficiency = []
        for bid in bids:
            # 预测在最优workload下的表现
            optimal_load = min(bid.capacity, total_workload)
            response = self.predict_follower_response(bid, optimal_load)
            
            # 性价比 = 质量 / 成本
            if response['actual_cost'] > 0:
                eff = response['actual_quality'] / response['actual_cost']
            else:
                eff = response['actual_quality']  # 避免除零
            
            efficiency.append({
                'agent_id': bid.agent_id,
                'efficiency': eff,
                'capacity': bid.capacity,
                'bid': bid
            })
        
        # 按性价比排序（从高到低）
        efficiency.sort(key=lambda x: x['efficiency'], reverse=True)
        
        # 贪心分配
        allocation = {}
        remaining_workload = total_workload
        
        for item in efficiency:
            if remaining_workload <= 0:
                break
            
            # 分配workload
            assigned = min(item['capacity'], remaining_workload)
            allocation[item['agent_id']] = assigned
            remaining_workload -= assigned
        
        # 如果还有剩余（说明总容量不足），按比例增加
        if remaining_workload > 0 and allocation:
            # 警告：容量不足
            print(f"⚠️  警告：总容量不足，还有 {remaining_workload:.2%} 的workload未分配")
            # 按比例分摊到已分配的Agent（会导致过载）
            for agent_id in allocation:
                allocation[agent_id] += remaining_workload / len(allocation)
        
        return allocation
    
    def evaluate_allocation(self, bids: List[AgentBid], allocation: Dict[str, float]) -> Dict:
        """
        评估分配方案的效果
        
        返回：
        - 总质量
        - 总成本
        - Leader收益
        - 各Agent的预期表现
        """
        total_quality = 0
        total_cost = 0
        agent_details = {}
        
        bid_map = {bid.agent_id: bid for bid in bids}
        
        for agent_id, workload in allocation.items():
            if agent_id not in bid_map:
                continue
            
            bid = bid_map[agent_id]
            response = self.predict_follower_response(bid, workload)
            
            total_quality += response['actual_quality'] * workload
            total_cost += response['actual_cost']
            
            agent_details[agent_id] = {
                'workload': workload,
                'quality': response['actual_quality'],
                'cost': response['actual_cost'],
                'utility': response['follower_utility'],
                'overloaded': response['overloaded']
            }
        
        # Leader的收益 = 总质量 × 权重 - 总成本
        leader_utility = total_quality * 100 - total_cost
        
        return {
            'total_quality': total_quality,
            'total_cost': total_cost,
            'leader_utility': leader_utility,
            'agent_details': agent_details,
            'allocation': allocation
        }
    
    def execute_stackelberg_game(self, bids: List[AgentBid], total_workload: float = 1.0) -> Dict:
        """
        执行完整的Stackelberg博弈
        
        流程：
        1. Leader收集Follower的竞标
        2. Leader优化分配（考虑Follower的最优反应）
        3. 计算均衡结果
        """
        print(f"\n{'='*60}")
        print("🎮 Stackelberg博弈开始")
        print(f"{'='*60}")
        print(f"  Leader: Agent:{self.leader_port}")
        print(f"  Follower数量: {len(bids)}")
        print(f"  总工作量: {total_workload:.2%}\n")
        
        # Step 1: 显示所有竞标
        print("📬 收到的竞标:")
        for i, bid in enumerate(bids, 1):
            print(f"  {i}. Agent:{bid.port} ({bid.model})")
            print(f"     质量承诺: {bid.quality_promise:.2f}")
            print(f"     成本: {bid.cost_per_task:.2f}")
            print(f"     容量: {bid.capacity:.2%}")
            print(f"     历史表现: {bid.past_performance:.2f}")
        
        # Step 2: 优化分配
        print(f"\n🧮 Leader优化分配中...")
        allocation = self.optimize_allocation(bids, total_workload)
        
        # Step 3: 评估结果
        result = self.evaluate_allocation(bids, allocation)
        
        # Step 4: 显示结果
        print(f"\n📊 Stackelberg均衡分配:")
        for agent_id, details in result['agent_details'].items():
            overload_mark = "⚠️ 过载" if details['overloaded'] else "✓"
            print(f"  Agent:{agent_id.split(':')[1]}: {details['workload']:.2%} workload {overload_mark}")
            print(f"    预期质量: {details['quality']:.2f}")
            print(f"    成本: {details['cost']:.2f}")
            print(f"    Agent收益: {details['utility']:.2f}")
        
        print(f"\n💰 博弈结果:")
        print(f"  总质量: {result['total_quality']:.2f}")
        print(f"  总成本: {result['total_cost']:.2f}")
        print(f"  Leader收益: {result['leader_utility']:.2f}")
        print(f"{'='*60}\n")
        
        # 记录历史
        self.history.append(result)
        
        return result

def calculate_nash_equilibrium_bounds(leader_utility: float, follower_utilities: List[float]) -> Dict:
    """
    计算纳什均衡的理论边界
    
    虽然Stackelberg是序贯博弈，但可以和同时博弈的纳什均衡对比
    """
    total_utility = leader_utility + sum(follower_utilities)
    
    return {
        'total_social_welfare': total_utility,
        'leader_share': leader_utility / total_utility if total_utility > 0 else 0,
        'followers_share': sum(follower_utilities) / total_utility if total_utility > 0 else 0,
        'efficiency': total_utility  # 社会总福利
    }

# ============ 示例使用 ============

if __name__ == "__main__":
    # 创建调度器
    scheduler = StackelbergScheduler(leader_port=8001)
    
    # 模拟竞标
    bids = [
        AgentBid(
            agent_id="127.0.0.1:8002",
            port=8002,
            model="qwen-plus",
            role="solver",
            quality_promise=0.85,
            cost_per_task=20,
            capacity=0.6,
            past_performance=0.9
        ),
        AgentBid(
            agent_id="127.0.0.1:8003",
            port=8003,
            model="deepseek-chat",
            role="reviewer",
            quality_promise=0.90,
            cost_per_task=25,
            capacity=0.4,
            past_performance=0.85
        ),
        AgentBid(
            agent_id="127.0.0.1:8004",
            port=8004,
            model="qwen-plus",
            role="solver",
            quality_promise=0.75,
            cost_per_task=15,
            capacity=0.5,
            past_performance=0.8
        )
    ]
    
    # 执行博弈
    result = scheduler.execute_stackelberg_game(bids, total_workload=1.0)
    
    # 计算理论边界
    follower_utils = [d['utility'] for d in result['agent_details'].values()]
    bounds = calculate_nash_equilibrium_bounds(result['leader_utility'], follower_utils)
    
    print(f"\n📈 理论分析:")
    print(f"  社会总福利: {bounds['total_social_welfare']:.2f}")
    print(f"  Leader占比: {bounds['leader_share']:.2%}")
    print(f"  Followers占比: {bounds['followers_share']:.2%}")
