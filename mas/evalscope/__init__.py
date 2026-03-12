"""
EvalScope 评测集成模块
================================================================

将 GeneratedDataset（多节点 AEIC 共识数据）导出为 EvalScope 格式，
利用 EvalScope 框架对模型的共识理解能力进行标准化评测。

两个评测任务：
  consensus_mcq   共识等级分类（选择题 MCQ，指标：accuracy）
                  输入：N 个节点的 AEIC 记录
                  输出：high / medium / low
                  
  consensus_qa    相似度估计（开放题 QA，LLM Judge 打分）
                  输入：场景描述 + N-1 个节点记录
                  输出：0-1 浮点相似度

快速使用：
  from mas.evalscope import export_datasets, run_evaluation
  
  # 1. 从已生成的数据集导出 EvalScope 格式
  export_datasets()
  
  # 2. 运行评测（DeepSeek 作为被测模型）
  run_evaluation(task="mcq")
  run_evaluation(task="qa")
  run_evaluation(task="all")
"""

from .exporter import DatasetExporter, export_datasets
from .run_eval  import run_evaluation, EvalConfig

__all__ = [
    "DatasetExporter",
    "export_datasets",
    "run_evaluation",
    "EvalConfig",
]
