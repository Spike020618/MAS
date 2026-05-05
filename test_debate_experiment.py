#!/usr/bin/env python3
"""
简化的辩论实验测试框架
用于验证4种实验配置的Stackelberg序贯决策
"""

from mas.consensus.stackelberg import StackelbergConsensusGame, AgentBid
from mas.consensus.consensus import ConsensusEngine
import numpy as np


class SimplifiedDebateExperiment:
    """简化的辩论实验 - 重点验证Stackelberg序贯决策"""

    def __init__(self):
        self.consensus_engine = ConsensusEngine()
        self.consensus_game = StackelbergConsensusGame(leader_port=8000, num_agents=3)

    def simulate_debate_response(self, agent_info: dict, leader_params: dict, aeic_structured: bool, round_num: int) -> str:
        """模拟辩论响应"""
        agent_id = agent_info['id']
        theta = leader_params.get(agent_id, 0.5)

        if aeic_structured:
            # AEIC结构化响应
            response = {
                "assumptions": f"基于{agent_info['role']}的专业视角，",
                "evidence": [f"证据{round_num}: θ={theta:.2f}影响辩论质量"],
                "inference": f"推断：激励参数θ={theta:.2f}会{'提高' if theta > 0.7 else '降低'}辩论效果",
                "conclusion": f"结论：{'支持' if theta > 0.5 else '反对'}当前立场"
            }
            return str(response)
        else:
            # 非结构化响应
            return f"Agent {agent_id} ({agent_info['role']}): 基于θ={theta:.2f}的激励，我的观点是..."

    def run_single_experiment(self, topic: str, aeic_structured: bool, multi_agent: bool) -> dict:
        """运行单个实验"""

        # 设置agents
        if multi_agent:
            agents = [
                {"id": "agent_1", "name": "张教授", "role": "经济学家"},
                {"id": "agent_2", "name": "李博士", "role": "社会学家"},
                {"id": "agent_3", "name": "王研究员", "role": "政策分析师"}
            ]
        else:
            agents = [{"id": "agent_1", "name": "辩论者", "role": "综合分析师"}]

        # 创建bids用于Stackelberg
        bids = []
        for agent in agents:
            bid = AgentBid(
                agent_id=agent['id'],
                port=8000 + len(bids),
                model='gpt-4',
                role=agent['role'],
                quality_promise=0.8,
                cost_per_task=10.0,
                capacity=0.5,
                past_performance=0.85
            )
            bids.append(bid)

        # 运行序贯Stackelberg
        stackelberg_result = self.consensus_game.run_sequential_stackelberg(
            bids,
            task_context={'complexity': 0.6, 'urgency': 0.4},
            max_iterations=3
        )

        # 模拟辩论过程
        debate_history = []
        leader_params_history = []

        for round_num in range(1, 4):  # 3轮辩论
            # Leader宣布参数
            current_params = stackelberg_result['final_params'] if round_num > 1 else stackelberg_result['iterations'][0]['leader_params']
            leader_params_history.append(current_params.copy())

            # 每个agent生成响应
            for agent in agents:
                response = self.simulate_debate_response(agent, dict(zip([a['id'] for a in agents], current_params)), aeic_structured, round_num)

                entry = {
                    "agent_id": agent['id'],
                    "round": round_num,
                    "response": response if not aeic_structured else None,
                    "aeic_response": response if aeic_structured else None
                }
                debate_history.append(entry)

        # 计算共识
        responses = [entry['response'] or str(entry['aeic_response']) for entry in debate_history[-len(agents):]]
        if len(responses) >= 2:
            # 构造节点记录格式
            node_records = []
            for i, response in enumerate(responses):
                node_records.append({
                    "node_id": f"agent_{i+1}",
                    "response": response,
                    "aeic_structured": aeic_structured
                })
            consensus_score = self.consensus_engine.evaluate_consensus(node_records)["avg_similarity"]
        else:
            # 单智能体情况下的模拟共识分数
            consensus_score = 0.8 + np.random.normal(0, 0.1)  # 基础分数加噪声

        return {
            "config": {
                "aeic_structured": aeic_structured,
                "multi_agent": multi_agent,
                "topic": topic
            },
            "stackelberg_result": stackelberg_result,
            "debate_history": debate_history,
            "leader_params_history": leader_params_history,
            "consensus_score": consensus_score,
            "metrics": {
                "consensus_score": consensus_score,
                "stackelberg_converged": stackelberg_result['converged'],
                "stackelberg_iterations": len(stackelberg_result['iterations']),
                "final_leader_params": stackelberg_result['final_params'].tolist(),
                "total_utterances": len(debate_history)
            }
        }

    def run_all_experiments(self, topic: str) -> dict:
        """运行4种实验配置"""

        configs = [
            {"name": "非结构化_单智能体", "aeic": False, "multi": False},
            {"name": "非结构化_多智能体", "aeic": False, "multi": True},
            {"name": "AEIC结构化_单智能体", "aeic": True, "multi": False},
            {"name": "AEIC结构化_多智能体", "aeic": True, "multi": True}
        ]

        results = {}
        for config in configs:
            print(f"🏃 运行实验：{config['name']}")
            result = self.run_single_experiment(
                topic=topic,
                aeic_structured=config['aeic'],
                multi_agent=config['multi']
            )
            results[config['name']] = result

            # 打印关键指标
            metrics = result['metrics']
            print(f"   共识分数: {metrics['consensus_score']:.3f}")
            print(f"   Stackelberg收敛: {metrics['stackelberg_converged']}")
            print(f"   最终参数: {[f'{x:.2f}' for x in metrics['final_leader_params']]}")
            print()

        return results


if __name__ == "__main__":
    # 测试实验框架
    experiment = SimplifiedDebateExperiment()

    print("🧪 4种实验配置对比测试")
    print("=" * 50)

    # 运行全对比实验
    results = experiment.run_all_experiments("人工智能是否应该被监管？")

    print("📊 实验结果汇总：")
    print("-" * 50)
    for config_name, result in results.items():
        metrics = result['metrics']
        print(f"{config_name}:")
        print(f"  共识分数: {metrics['consensus_score']:.3f}")
        print(f"  Stackelberg收敛: {metrics['stackelberg_converged']}")
        print(f"  迭代次数: {metrics['stackelberg_iterations']}")
        print(f"  最终参数: {[f'{x:.2f}' for x in metrics['final_leader_params']]}")
        print()