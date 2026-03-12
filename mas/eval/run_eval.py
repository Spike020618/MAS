"""
evalscope 评测入口
================================================================

两个评测任务：

  Task 1: consensus_mcq  共识等级三分类（多选一）
    给模型 N 份 AEIC 记录，让它判断 high / medium / low。
    自动计算 Accuracy，无需 judge 模型。
    适合快速验证模型对"共识程度"的判断能力。

  Task 2: consensus_qa  完整共识分析（开放问答）
    给模型完整 AEIC 记录，要求输出：
      - 共识等级 + 相似度估计
      - 共识点与分歧点
      - 引擎决策建议
    用 DeepSeek 作为 judge，从准确性/专业性/结构完整性三维打分（1-5分）。

用法：
  # 一键运行两个评测任务
  python mas/eval/run_eval.py

  # 只跑 MCQ
  python mas/eval/run_eval.py --task mcq

  # 只跑 QA（需要 judge）
  python mas/eval/run_eval.py --task qa

  # 强制重新导出数据（数据集有更新时用）
  python mas/eval/run_eval.py --reexport

  # 评测指定模型 API（默认 DeepSeek）
  python mas/eval/run_eval.py --model deepseek-chat --api-url https://api.deepseek.com/v1/chat/completions

环境要求：
  pip install evalscope
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ── 项目路径 ──────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mas.data import load_or_generate
from mas.eval.export_to_evalscope import EvalScopeExporter

try:
    import config as _cfg
    DEEPSEEK_API_KEY  = getattr(_cfg, "DEEPSEEK_API_KEY",  None)
    DEEPSEEK_API_BASE = getattr(_cfg, "DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL    = getattr(_cfg, "DEEPSEEK_MODEL",    "deepseek-chat")
except Exception:
    DEEPSEEK_API_KEY  = None
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL    = "deepseek-chat"

# evalscope 输出 / 数据目录
_EVAL_DATA_DIR    = os.path.join(_ROOT, "evalscope_data")
_EVAL_RESULTS_DIR = os.path.join(_ROOT, "results", "evalscope")
_DATASET_CACHE    = os.path.join(_ROOT, "results", "generated_dataset.json")

os.makedirs(_EVAL_RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# 数据准备
# ─────────────────────────────────────────────────────

def prepare_data(force_reexport: bool = False) -> dict:
    """
    1. 加载/生成 AEIC 数据集
    2. 导出为 evalscope 格式（有缓存时跳过）
    """
    manifest_path = os.path.join(_EVAL_DATA_DIR, "manifest.json")

    # 检查是否需要重新导出
    if not force_reexport and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"  📂 evalscope 数据已存在，跳过导出（{manifest_path}）")
        print(f"     数据集统计: {manifest.get('stats', {})}")
        return manifest.get("evalscope", {})

    print("  📡 加载/生成 AEIC 数据集...")
    ds = load_or_generate(cache_path=_DATASET_CACHE)

    exporter = EvalScopeExporter(ds, output_dir=_EVAL_DATA_DIR)
    paths    = exporter.export_all()

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    return manifest.get("evalscope", {})


# ─────────────────────────────────────────────────────
# Task 1: MCQ 评测
# ─────────────────────────────────────────────────────

def run_mcq_eval(
    model: str = None,
    api_url: str = None,
    api_key: str = None,
    limit: int = None,
) -> dict:
    """
    general_mcq 共识等级分类评测。

    评测流程：
      模型读取「N 份 AEIC 记录」→ 判断 A/B/C（high/medium/low）
      evalscope 自动计算 AverageAccuracy

    以 eval-type service 模式调用模型 API。
    """
    try:
        from evalscope import TaskConfig, run_task
    except ImportError:
        print("❌ evalscope 未安装，请运行: pip install evalscope")
        return {}

    _model   = model   or DEEPSEEK_MODEL
    _url     = api_url or (DEEPSEEK_API_BASE.rstrip("/") + "/chat/completions")
    _key     = api_key or DEEPSEEK_API_KEY or "EMPTY"
    _mcq_dir = os.path.join(_EVAL_DATA_DIR, "consensus_mcq")

    print(f"\n{'='*65}")
    print(f"  📊 Task 1: MCQ 共识等级分类评测")
    print(f"  模型: {_model}  |  API: {_url}")
    print(f"{'='*65}\n")

    task_cfg = TaskConfig(
        model=_model,
        api_url=_url,
        api_key=_key,
        eval_type="service",
        datasets=["general_mcq"],
        dataset_args={
            "general_mcq": {
                "local_path":   _mcq_dir,
                "subset_list":  ["consensus"],
            }
        },
        work_dir=_EVAL_RESULTS_DIR,
        outputs=_EVAL_RESULTS_DIR,
        **({"limit": limit} if limit else {}),
    )

    result = run_task(task_cfg=task_cfg)
    _print_result("MCQ", result)
    return result or {}


# ─────────────────────────────────────────────────────
# Task 2: QA 评测（LLM judge）
# ─────────────────────────────────────────────────────

def run_qa_eval(
    model: str = None,
    api_url: str = None,
    api_key: str = None,
    judge_model: str = None,
    judge_api_url: str = None,
    judge_api_key: str = None,
    limit: int = None,
) -> dict:
    """
    general_qa 完整共识分析评测，使用 DeepSeek 作为 LLM judge。

    judge 使用 numeric 模式（1-5 分），从以下维度评估模型输出：
      - 准确性：共识等级和相似度估计是否正确
      - 专业性：分析是否使用了正确的 AEIC / 共识概念
      - 结构完整性：是否覆盖了等级/估计/共识点/决策四个方面
    """
    try:
        from evalscope import TaskConfig, run_task
        from evalscope.constants import JudgeStrategy
    except ImportError:
        print("❌ evalscope 未安装，请运行: pip install evalscope")
        return {}

    _model     = model        or DEEPSEEK_MODEL
    _url       = api_url      or (DEEPSEEK_API_BASE.rstrip("/") + "/chat/completions")
    _key       = api_key      or DEEPSEEK_API_KEY or "EMPTY"
    _j_model   = judge_model  or DEEPSEEK_MODEL
    _j_url     = judge_api_url or (DEEPSEEK_API_BASE.rstrip("/") + "/chat/completions")
    _j_key     = judge_api_key or DEEPSEEK_API_KEY or "EMPTY"
    _qa_dir    = os.path.join(_EVAL_DATA_DIR, "consensus_qa")

    print(f"\n{'='*65}")
    print(f"  📊 Task 2: QA 共识分析评测（LLM judge）")
    print(f"  模型: {_model}  |  judge: {_j_model}")
    print(f"{'='*65}\n")

    _judge_prompt = (
        "你是一位严格的多智能体共识评估专家。\n"
        "请对以下模型输出从三个维度打分（每项 1-5 分）：\n"
        "  1. 准确性：共识等级（high/medium/low）和相似度估计是否与参考答案一致\n"
        "  2. 专业性：是否正确使用了 AEIC 共识相关术语，分析是否专业\n"
        "  3. 完整性：是否覆盖了共识等级、相似度估计、共识/分歧点、决策建议四个方面\n\n"
        "请以 JSON 格式输出：{\"accuracy\": N, \"professionalism\": N, \"completeness\": N, \"overall\": N}\n"
        "overall 为三项均分，overall 即为最终分数。"
    )

    task_cfg = TaskConfig(
        model=_model,
        api_url=_url,
        api_key=_key,
        eval_type="service",
        datasets=["general_qa"],
        dataset_args={
            "general_qa": {
                "dataset_id":  _qa_dir,
                "subset_list": ["consensus_test"],
            }
        },
        judge_model_args={
            "model_id":    _j_model,
            "api_url":     _j_url,
            "api_key":     _j_key,
            "judge_strategy": JudgeStrategy.NUMERIC,
            "generation_config": {
                "temperature": 0.0,
                "max_tokens":  256,
            },
            "custom_prompt_template": _judge_prompt,
        },
        work_dir=_EVAL_RESULTS_DIR,
        outputs=_EVAL_RESULTS_DIR,
        **({"limit": limit} if limit else {}),
    )

    result = run_task(task_cfg=task_cfg)
    _print_result("QA", result)
    return result or {}


# ─────────────────────────────────────────────────────
# 结果打印
# ─────────────────────────────────────────────────────

def _print_result(task_name: str, result):
    print(f"\n{'='*65}")
    print(f"  ✅ {task_name} 评测结果")
    print(f"{'='*65}")
    if result is None:
        print("  （无结果，请检查 evalscope 输出日志）")
        return
    if isinstance(result, dict):
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print(f"  {result}")
    print(f"  详细结果已保存至: {_EVAL_RESULTS_DIR}")
    print()


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MAS evalscope 评测入口（AEIC 共识理解能力评测）"
    )
    parser.add_argument("--task",       type=str, default="all",
                        choices=["mcq", "qa", "all"],
                        help="评测任务（默认 all）")
    parser.add_argument("--model",      type=str, default=None,
                        help=f"被评测模型名（默认 {DEEPSEEK_MODEL}）")
    parser.add_argument("--api-url",    type=str, default=None,
                        help="模型 API URL（默认 DeepSeek）")
    parser.add_argument("--api-key",    type=str, default=None,
                        help="模型 API Key（默认读 config.py）")
    parser.add_argument("--judge",      type=str, default=None,
                        help="judge 模型名（QA 任务，默认同 --model）")
    parser.add_argument("--limit",      type=int, default=None,
                        help="每个任务最多评测 N 条（调试用）")
    parser.add_argument("--reexport",   action="store_true",
                        help="强制重新导出 evalscope 数据格式")
    args = parser.parse_args()

    print()
    print("╔" + "═"*63 + "╗")
    print("║   MAS · evalscope 共识理解能力评测                         ║")
    print("╚" + "═"*63 + "╝")

    # 准备数据
    prepare_data(force_reexport=args.reexport)

    # 运行评测
    if args.task in ("mcq", "all"):
        run_mcq_eval(
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
            limit=args.limit,
        )

    if args.task in ("qa", "all"):
        run_qa_eval(
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
            judge_model=args.judge or args.model,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
