#!/usr/bin/env python3
"""
辩论实验框架 - 两层结构
======================================================================

DatasetLoader: 负责数据集加载和管理
ExperimentRunner: 负责实验执行和结果收集
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

from mas.consensus.stackelberg import StackelbergConsensusGame, AgentBid
from mas.consensus.consensus import ConsensusEngine
import numpy as np

# 直接导入数据生成器模块，避免 experiments 包初始化时依赖额外库
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
import dataset_generator
DatasetGenerator = dataset_generator.DatasetGenerator


class DatasetLoader:
    """数据集加载器 - 负责数据集管理和预处理"""

    def __init__(self):
        self.dataset = None
        self.tasks = []

    def load_from_generator(self, num_tasks: int = 7, num_nodes: int = 3) -> List[Dict]:
        """从 DatasetGenerator 加载数据集"""
        self.dataset = DatasetGenerator.create_simple_dataset(num_tasks, num_nodes)
        self.tasks = self.dataset
        return self.tasks

    def get_task_by_index(self, index: int) -> Dict:
        """按索引获取任务"""
        if 0 <= index < len(self.tasks):
            return self.tasks[index]
        raise IndexError(f"Task index {index} out of range")

    def get_all_tasks(self) -> List[Dict]:
        """获取所有任务"""
        return self.tasks

    def prepare_task_for_experiment(self, task: Dict, multi_agent: bool) -> Dict:
        """为实验准备任务数据"""
        if multi_agent:
            agents = [
                {"id": f"agent_{i+1}", "name": f"Agent-{i+1}", "role": f"{task['domain']}_node_{i+1}"}
                for i in range(len(task['nodes']))
            ]
            node_records = task['nodes']
        else:
            agents = [{"id": "agent_1", "name": "单智能体", "role": f"{task['domain']}_merged"}]
            # 单智能体时复制节点以满足 ConsensusEngine 最小节点数要求
            node_records = [task['nodes'][0], task['nodes'][0].copy()]

        # 生成对应的 bids
        bids = []
        for agent in agents:
            bids.append(AgentBid(
                agent_id=agent['id'],
                port=8000 + len(bids),
                model='gpt-4',
                role=agent['role'],
                quality_promise=0.8,
                cost_per_task=10.0,
                capacity=0.5,
                past_performance=0.85
            ))

        return {
            'agents': agents,
            'bids': bids,
            'node_records': node_records,
            'task_info': {
                'domain': task['domain'],
                'consistency_type': task.get('consistency_type', 'unknown')
            }
        }


class ExperimentRunner:
    """实验运行器 - 负责实验执行和结果收集"""

    def __init__(self):
        self.consensus_engine = ConsensusEngine()
        self.consensus_game = StackelbergConsensusGame(leader_port=8000, num_agents=3)

    def _calculate_stackelberg_metrics(self, stackelberg_result: Dict) -> Dict:
        """计算Stackelberg博弈的性能指标"""
        final_params = stackelberg_result['final_params']
        iterations = stackelberg_result['iterations']
        
        # 1. 参数收敛度：最后两次迭代的参数差异
        if len(iterations) >= 2:
            last_params = iterations[-1]['updated_params']
            prev_params = iterations[-2]['updated_params']
            param_diff = np.abs(last_params - prev_params)
            param_convergence = 1.0 - np.mean(param_diff)
        else:
            param_convergence = 1.0 if stackelberg_result['converged'] else 0.5
        
        # 2. 共识能量：最终参数的共识能量值（从最后一次迭代中取）
        if len(iterations) > 0:
            consensus_energy = iterations[-1].get('consensus_energy', 0.0)
        else:
            consensus_energy = self.consensus_game.consensus_energy(final_params)
        
        # 3. 参数方差：参数分布的一致性
        param_variance = float(np.var(final_params))
        
        return {
            "param_convergence": float(param_convergence),
            "consensus_energy": float(consensus_energy),
            "param_variance": param_variance,
            "param_std": float(np.std(final_params)),
            "final_params": final_params.tolist()
        }

    def run_single_experiment(self, prepared_task: Dict, aeic_structured: bool) -> Dict:
        """运行单个实验配置"""

        bids = prepared_task['bids']
        node_records = prepared_task['node_records']

        # 运行序贯Stackelberg
        stackelberg_result = self.consensus_game.run_sequential_stackelberg(
            bids,
            task_context={'complexity': 0.6, 'urgency': 0.4},
            max_iterations=3
        )

        # 计算共识分数（输入节点的语义相似度）
        if aeic_structured:
            semantic_similarity = self.consensus_engine.evaluate_consensus(node_records)["avg_similarity"]
        else:
            # 非结构化时，使用节点文本组合后构建AEIC格式
            node_texts = []
            for i, node in enumerate(node_records):
                combined = ' '.join(str(v) for v in node.values())
                node_texts.append({
                    'node_id': f'node_{i+1}',
                    'assumptions': combined,
                    'evidence': combined,
                    'inference': combined,
                    'conclusion': combined
                })
            semantic_similarity = self.consensus_engine.evaluate_consensus(node_texts)["avg_similarity"]

        # 计算Stackelberg性能指标
        stackelberg_metrics = self._calculate_stackelberg_metrics(stackelberg_result)

        return {
            "stackelberg_result": stackelberg_result,
            "semantic_similarity": semantic_similarity,  # 输入节点的相似度
            "stackelberg_metrics": stackelberg_metrics,  # Stackelberg博弈指标
            "metrics": {
                "semantic_similarity": semantic_similarity,
                "param_convergence": stackelberg_metrics["param_convergence"],
                "consensus_energy": stackelberg_metrics["consensus_energy"],
                "param_variance": stackelberg_metrics["param_variance"],
                "stackelberg_converged": stackelberg_result['converged'],
                "stackelberg_iterations": len(stackelberg_result['iterations']),
                "final_leader_params": stackelberg_result['final_params'].tolist()
            }
        }

    def run_task_experiments(self, prepared_task: Dict) -> Dict[str, Dict]:
        """对单个任务运行所有实验配置"""

        results = {}
        for aeic in [False, True]:
            for multi in [False, True]:
                config_name = f"{'AEIC' if aeic else '非结构化'}_{'多智能体' if multi else '单智能体'}"

                # 重新准备任务数据（因为单/多智能体有不同处理）
                task = {'nodes': prepared_task['node_records'], 'domain': prepared_task['task_info']['domain']}
                task_prepared = self._prepare_task_for_experiment(task, multi)

                result = self.run_single_experiment(task_prepared, aeic)
                results[config_name] = result

        return results

    def _prepare_task_for_experiment(self, task: Dict, multi_agent: bool) -> Dict:
        """内部方法：准备任务数据"""
        if multi_agent:
            agents = [
                {"id": f"agent_{i+1}", "name": f"Agent-{i+1}", "role": f"{task['domain']}_node_{i+1}"}
                for i in range(len(task['nodes']))
            ]
            node_records = task['nodes']
        else:
            agents = [{"id": "agent_1", "name": "单智能体", "role": f"{task['domain']}_merged"}]
            # 单智能体时复制节点以满足 ConsensusEngine 最小节点数要求
            node_records = [task['nodes'][0], task['nodes'][0].copy()]

        bids = []
        for agent in agents:
            bids.append(AgentBid(
                agent_id=agent['id'],
                port=8000 + len(bids),
                model='gpt-4',
                role=agent['role'],
                quality_promise=0.8,
                cost_per_task=10.0,
                capacity=0.5,
                past_performance=0.85
            ))

        return {
            'agents': agents,
            'bids': bids,
            'node_records': node_records,
            'task_info': {
                'domain': task['domain'],
                'consistency_type': task.get('consistency_type', 'unknown')
            }
        }


class DebateExperimentFramework:
    """辩论实验框架 - 整合 DatasetLoader 和 ExperimentRunner"""

    def __init__(self):
        self.dataset_loader = DatasetLoader()
        self.experiment_runner = ExperimentRunner()

    def run_architecture_validation(self) -> Dict[str, Dict]:
        """架构验证 - 使用模拟数据验证流程"""

        print("🧪 架构验证：4种实验配置对比测试")
        print("=" * 50)

        # 这里使用简化的模拟验证（不依赖数据集）
        configs = [
            {"name": "非结构化_单智能体", "aeic": False, "multi": False},
            {"name": "非结构化_多智能体", "aeic": False, "multi": True},
            {"name": "AEIC结构化_单智能体", "aeic": True, "multi": False},
            {"name": "AEIC结构化_多智能体", "aeic": True, "multi": True}
        ]

        results = {}
        for config in configs:
            print(f"🏃 运行实验：{config['name']}")

            # 模拟任务数据
            mock_task = self._create_mock_task(config['multi'])
            prepared_task = self.dataset_loader.prepare_task_for_experiment(mock_task, config['multi'])
            result = self.experiment_runner.run_single_experiment(prepared_task, config['aeic'])

            results[config['name']] = result

            # 打印关键指标
            metrics = result['metrics']
            print(f"   语义相似度(输入): {metrics['semantic_similarity']:.3f}")
            print(f"   参数收敛度: {metrics['param_convergence']:.3f}")
            print(f"   共识能量: {metrics['consensus_energy']:.4f}")
            print(f"   Stackelberg收敛: {metrics['stackelberg_converged']}")
            print(f"   最终参数: {[f'{x:.2f}' for x in metrics['final_leader_params']]}")
            print()

        return results

    def run_dataset_experiments(self, num_tasks: int = 7) -> Dict[str, Dict]:
        """数据集驱动实验"""

        print(f"\n📚 数据集驱动实验：{num_tasks} 个任务")
        print("=" * 50)

        # 加载数据集
        tasks = self.dataset_loader.load_from_generator(num_tasks)

        results = {}
        for idx, task in enumerate(tasks, start=1):
            print(f"🏷 任务 {idx}/{len(tasks)}: domain={task['domain']} consistency={task.get('consistency_type')}")

            prepared_task = self.dataset_loader.prepare_task_for_experiment(task, multi_agent=True)
            task_results = self.experiment_runner.run_task_experiments(prepared_task)

            results[f"task_{idx}"] = task_results

            # 打印结果摘要
            for config_name, result in task_results.items():
                metrics = result['metrics']
                semantic_sim = metrics['semantic_similarity']
                converged = metrics['stackelberg_converged']
                print(f"   {config_name} => semantic_sim={semantic_sim:.3f} converged={converged}")

            print()

        return results

    def _create_mock_task(self, multi_agent: bool) -> Dict:
        """创建模拟任务用于架构验证"""
        if multi_agent:
            nodes = [
                {
                    'assumptions': '假设市场处于上升趋势',
                    'evidence': ['上市公司盈利增长15%', '机构投资者净买入增加'],
                    'inference': '通过数据推断市场向好',
                    'conclusion': '建议增加股票配置'
                },
                {
                    'assumptions': '假设经济数据可靠',
                    'evidence': ['GDP增速下降到5.5%', '融资规模创新高'],
                    'inference': '经济放缓但仍保持增长',
                    'conclusion': '建议控制风险敞口'
                },
                {
                    'assumptions': '假设政策环境稳定',
                    'evidence': ['央行公布存款准备金率下降', '市场情绪积极'],
                    'inference': '流动性充足促进投资',
                    'conclusion': '看好市场长期表现'
                }
            ]
        else:
            nodes = [{
                'assumptions': '假设综合市场分析',
                'evidence': ['多维度数据综合分析'],
                'inference': '综合判断市场状况',
                'conclusion': '综合投资建议'
            }]

        return {
            'nodes': nodes,
            'domain': 'finance',
            'consistency_type': 'mock_validation'
        }


if __name__ == "__main__":
    # 创建实验框架
    framework = DebateExperimentFramework()

    # 第一阶段：架构验证
    validation_results = framework.run_architecture_validation()

    print("📊 架构验证结果汇总：")
    print("-" * 50)
    for config_name, result in validation_results.items():
        metrics = result['metrics']
        print(f"{config_name}:")
        print(f"  语义相似度: {metrics['semantic_similarity']:.3f}")
        print(f"  参数收敛度: {metrics['param_convergence']:.3f}")
        print(f"  共识能量: {metrics['consensus_energy']:.4f}")
        print(f"  Stackelberg收敛: {metrics['stackelberg_converged']}")
        print(f"  迭代次数: {metrics['stackelberg_iterations']}")
        print(f"  最终参数: {[f'{x:.2f}' for x in metrics['final_leader_params']]}")
        print()

    # 第二阶段：数据集驱动实验
    dataset_results = framework.run_dataset_experiments(num_tasks=3)

    print("📊 数据集驱动实验结果示例：")
    print("-" * 50)
    for task_name, task_result in list(dataset_results.items())[:2]:  # 只显示前2个任务
        print(f"{task_name}:")
        for config_name, result in task_result.items():
            metrics = result['metrics']
            print(f"  {config_name}: semantic_sim={metrics['semantic_similarity']:.3f} converged={metrics['stackelberg_converged']}")
        print()