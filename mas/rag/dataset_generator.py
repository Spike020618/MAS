"""
数据集生成器 - 为对比实验生成任务和反馈数据

功能：
- 生成合成任务数据集
- 生成Agent配置
- 生成反馈样本
"""

import logging
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TaskSample:
    """任务样本"""
    task_id: str
    task_type: str
    description: str
    expected_agent_id: int
    baseline_difficulty: float  # 难度（0-1）


@dataclass
class AgentConfig:
    """Agent配置"""
    agent_id: int
    name: str
    task_types: List[str]
    base_success_rate: float


class DatasetGenerator:
    """数据集生成器"""

    def __init__(self, seed: int = 42):
        """
        初始化数据集生成器

        Args:
            seed: 随机种子
        """
        random.seed(seed)
        self.seed = seed
        logger.info(f"✓ DatasetGenerator initialized with seed={seed}")

    def generate_agents(self, num_agents: int = 5) -> List[AgentConfig]:
        """
        生成Agent配置

        Args:
            num_agents: Agent数量

        Returns:
            Agent配置列表
        """
        try:
            agents = []
            task_types_pool = ["review", "planning", "development", "testing"]

            for i in range(num_agents):
                agent_id = i + 1
                name = f"Agent_{agent_id:02d}"

                # 随机选择2-3个任务类型
                num_types = random.randint(2, 3)
                task_types = random.sample(task_types_pool, num_types)

                # 生成基础成功率 (0.6-0.95)
                base_rate = random.uniform(0.6, 0.95)

                agents.append(
                    AgentConfig(
                        agent_id=agent_id,
                        name=name,
                        task_types=task_types,
                        base_success_rate=base_rate,
                    )
                )

            logger.info(f"✓ Generated {len(agents)} agents")
            return agents

        except Exception as e:
            logger.error(f"✗ Failed to generate agents: {e}")
            raise

    def generate_tasks(
        self,
        num_tasks: int = 50,
        agents: Optional[List[AgentConfig]] = None,
    ) -> List[TaskSample]:
        """
        生成任务数据集

        Args:
            num_tasks: 任务数量
            agents: Agent配置列表

        Returns:
            任务样本列表
        """
        try:
            if agents is None:
                agents = self.generate_agents()

            tasks = []
            task_types = set()

            # 收集所有任务类型
            for agent in agents:
                task_types.update(agent.task_types)

            task_types_list = list(task_types)

            for i in range(num_tasks):
                task_id = f"task_{i:04d}"

                # 随机选择任务类型
                task_type = random.choice(task_types_list)

                # 找到支持此任务的Agent
                supporting_agents = [
                    a for a in agents if task_type in a.task_types
                ]

                if not supporting_agents:
                    continue

                # 随机选择一个Agent作为"最佳"Agent
                expected_agent = random.choice(supporting_agents)

                # 生成任务描述
                description = self._generate_task_description(task_type, i)

                # 生成难度
                difficulty = random.uniform(0.1, 0.9)

                tasks.append(
                    TaskSample(
                        task_id=task_id,
                        task_type=task_type,
                        description=description,
                        expected_agent_id=expected_agent.agent_id,
                        baseline_difficulty=difficulty,
                    )
                )

            logger.info(f"✓ Generated {len(tasks)} tasks")
            return tasks

        except Exception as e:
            logger.error(f"✗ Failed to generate tasks: {e}")
            raise

    def generate_feedback_samples(
        self,
        tasks: List[TaskSample],
        agents: List[AgentConfig],
        num_samples_per_task: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        生成反馈样本

        Args:
            tasks: 任务列表
            agents: Agent列表
            num_samples_per_task: 每个任务的样本数

        Returns:
            反馈样本列表
        """
        try:
            samples = []

            for task in tasks:
                for sample_idx in range(num_samples_per_task):
                    # 随机选择一个支持此任务的Agent
                    supporting_agents = [
                        a for a in agents if task.task_type in a.task_types
                    ]

                    if not supporting_agents:
                        continue

                    allocated_agent = random.choice(supporting_agents)

                    # 基于Agent的基础成功率生成反馈
                    # 如果是最佳Agent，成功率更高
                    if allocated_agent.agent_id == task.expected_agent_id:
                        # 最佳Agent成功率较高
                        base_rate = allocated_agent.base_success_rate
                        noise = random.gauss(0, 0.05)
                    else:
                        # 其他Agent成功率较低
                        base_rate = allocated_agent.base_success_rate - 0.1
                        noise = random.gauss(0, 0.08)

                    # 考虑任务难度
                    difficulty_factor = 1.0 - task.baseline_difficulty * 0.3
                    success_score = max(0.0, min(1.0, base_rate * difficulty_factor + noise))

                    samples.append({
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "allocated_agent_id": allocated_agent.agent_id,
                        "allocated_agent_name": allocated_agent.name,
                        "success_score": success_score,
                        "is_optimal": allocated_agent.agent_id == task.expected_agent_id,
                        "difficulty": task.baseline_difficulty,
                    })

            logger.info(f"✓ Generated {len(samples)} feedback samples")
            return samples

        except Exception as e:
            logger.error(f"✗ Failed to generate feedback samples: {e}")
            raise

    def _generate_task_description(self, task_type: str, idx: int) -> str:
        """生成任务描述"""
        descriptions = {
            "review": [
                "代码审查：检查代码质量",
                "文档审查：验证文档准确性",
                "安全审查：检查安全漏洞",
                "性能审查：优化代码性能",
            ],
            "planning": [
                "项目规划：制定时间表",
                "资源规划：分配资源",
                "风险规划：评估风险",
                "流程规划：设计工作流",
            ],
            "development": [
                "功能开发：实现新功能",
                "模块开发：开发新模块",
                "集成开发：进行系统集成",
                "部署开发：部署到生产环境",
            ],
            "testing": [
                "单元测试：测试单个组件",
                "集成测试：测试系统集成",
                "性能测试：测试性能",
                "安全测试：测试安全性",
            ],
        }

        candidates = descriptions.get(task_type, ["Generic task"])
        return random.choice(candidates)

    def split_dataset(
        self,
        tasks: List[TaskSample],
        train_ratio: float = 0.7,
    ) -> Tuple[List[TaskSample], List[TaskSample]]:
        """
        分割数据集为训练集和测试集

        Args:
            tasks: 任务列表
            train_ratio: 训练集比例

        Returns:
            (训练集, 测试集)
        """
        try:
            random.shuffle(tasks)
            split_idx = int(len(tasks) * train_ratio)

            train_tasks = tasks[:split_idx]
            test_tasks = tasks[split_idx:]

            logger.info(
                f"✓ Split dataset: {len(train_tasks)} train, {len(test_tasks)} test"
            )

            return train_tasks, test_tasks

        except Exception as e:
            logger.error(f"✗ Failed to split dataset: {e}")
            raise
