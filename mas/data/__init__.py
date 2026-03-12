"""
数据模块 - LLM 驱动的多节点共识数据生成
================================================================

核心入口：
  load_or_generate()   → 有缓存就加载，否则调用 DeepSeek 生成

架构说明：
  每个 ConsensusRound 包含 N 个节点（node_0 ... node_N-1）的 AEIC 记录，
  而非二元 agent_a / agent_b 对。
  共识引擎对所有节点做全量两两相似度计算。

用法：
  from mas.data import load_or_generate

  # 首次运行：调用 DeepSeek 生成，自动缓存
  dataset = load_or_generate(n_nodes=3)

  # 后续运行：直接读缓存
  dataset = load_or_generate()

  # 获取节点记录（供共识引擎使用）
  node_recs_list = dataset.get_node_records()
  # → [ [node_0_dict, node_1_dict, node_2_dict], [...], ... ]

  # 强制重新生成
  dataset = load_or_generate(force_regenerate=True)

  # 只生成特定场景
  from mas.data import TASK_SCENARIOS
  finance_only = [s for s in TASK_SCENARIOS if s.domain == "finance"]
  dataset = load_or_generate(scenarios=finance_only, n_per_scenario=2)
"""

from .generator import (
    DataGenerator,
    GeneratedDataset,
    ConsensusRound,
    AEICRecord,
    AEICPair,          # 向后兼容别名
    TaskScenario,
    TASK_SCENARIOS,
    load_or_generate,
)

# 向后兼容别名
BenchmarkDataset = GeneratedDataset


class DataLoader:
    """向后兼容接口"""

    @staticmethod
    def load_builtin(cache_path=None, force_regenerate=False):
        return load_or_generate(
            cache_path=cache_path,
            force_regenerate=force_regenerate,
        )


__all__ = [
    "DataGenerator",
    "GeneratedDataset",
    "BenchmarkDataset",
    "DataLoader",
    "ConsensusRound",
    "AEICPair",
    "AEICRecord",
    "TaskScenario",
    "TASK_SCENARIOS",
    "load_or_generate",
]
