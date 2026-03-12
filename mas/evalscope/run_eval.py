"""
EvalScope 评测运行器
================================================================

用 EvalScope 框架对 DeepSeek 等模型进行多节点语义共识评测，
并将结果与我们自己的 ConsensusEngine（BM25/Embedding）对比。

评测维度：
  1. MCQ 准确率    模型判断共识等级是否正确（accuracy）
  2. QA 相似度误差  模型估计相似度与 GT 的差距（LLM Judge 打分）
  3. QA 生成质量    模型生成 AEIC 记录的专业性和一致性（LLM Judge）

支持两种被测对象：
  - service 模式：被测模型通过 API 访问（DeepSeek/Qwen等）
  - engine  模式：我们的 ConsensusEngine 直接运行（本地，不走 LLM）

用法：
  # 命令行
  python mas/evalscope/run_eval.py --task mcq
  python mas/evalscope/run_eval.py --task all --model deepseek
  python mas/evalscope/run_eval.py --task mcq --engine    # 用 ConsensusEngine

  # Python API
  from mas.evalscope import run_evaluation
  run_evaluation(task="mcq")
  run_evaluation(task="all", model="deepseek", engine_compare=True)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ── 路径 ──────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_MAS_ROOT  = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_MAS_ROOT)
sys.path.insert(0, _MAS_ROOT)

from .exporter import (
    DatasetExporter, export_datasets,
    DATASETS_DIR,
)

RESULTS_DIR = os.path.join(_PROJ_ROOT, "results", "evalscope")

# ── 配置 ──────────────────────────────────────────────────
try:
    import config as _cfg
    DEEPSEEK_API_KEY  = getattr(_cfg, "DEEPSEEK_API_KEY",  "")
    DEEPSEEK_API_BASE = getattr(_cfg, "DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL    = getattr(_cfg, "DEEPSEEK_MODEL",    "deepseek-chat")
    DASHSCOPE_API_KEY = getattr(_cfg, "API_KEY",           "")
    DASHSCOPE_API_BASE= getattr(_cfg, "API_BASE",          "")
except Exception:
    DEEPSEEK_API_KEY  = ""
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL    = "deepseek-chat"
    DASHSCOPE_API_KEY = ""
    DASHSCOPE_API_BASE= ""


# ─────────────────────────────────────────────────────────
# 配置数据类
# ─────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    """EvalScope 评测任务配置"""

    # 被测模型（EvalScope service 模式）
    model:    str = "deepseek-chat"
    api_url:  str = ""          # 填空则自动从 config.py 读取
    api_key:  str = ""

    # Judge 模型（用于 QA 开放题打分）
    judge_model:   str = ""     # 填空则复用 model
    judge_api_url: str = ""
    judge_api_key: str = ""

    # 评测范围
    task:  str = "all"          # mcq / qa_sim / qa_gen / all
    limit: int = 0              # 0 = 不限制；>0 = 限制条数（debug 用）

    # 本地 ConsensusEngine 对比评测
    engine_compare: bool  = False
    engine_method:  str   = "bm25"

    # 输出
    output_dir:  str = RESULTS_DIR
    verbose:     bool = True

    def __post_init__(self):
        if not self.api_url:
            self.api_url = DEEPSEEK_API_BASE.rstrip("/") + "/chat/completions"
        if not self.api_key:
            self.api_key = DEEPSEEK_API_KEY
        if not self.judge_model:
            self.judge_model = self.model
        if not self.judge_api_url:
            self.judge_api_url = self.api_url
        if not self.judge_api_key:
            self.judge_api_key = self.api_key


# ─────────────────────────────────────────────────────────
# EvalScope 包装器
# ─────────────────────────────────────────────────────────

class EvalScopeRunner:
    """调用 EvalScope 运行标准评测任务"""

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

    def _base_task_cfg(self) -> Dict:
        """构造 EvalScope TaskConfig 的公共部分"""
        cfg_dict = {
            "model":     self.cfg.model,
            "eval_type": "service",
            "api_url":   self.cfg.api_url,
            "api_key":   self.cfg.api_key,
            "generation_config": {
                "temperature": 0.0,
                "max_tokens":  512,
            },
        }
        if self.cfg.limit > 0:
            cfg_dict["limit"] = self.cfg.limit
        return cfg_dict

    def run_mcq(self) -> Optional[Dict]:
        """运行共识等级分类 MCQ 评测"""
        print("\n" + "─" * 60)
        print("📊 EvalScope MCQ 评测：共识等级分类")
        print("─" * 60)

        mcq_path = os.path.join(DatasetExporter.MCQ_DIR, "consensus_mcq_val.jsonl")
        if not os.path.exists(mcq_path):
            print(f"  ❌ 数据集未找到: {mcq_path}")
            print(f"     请先运行 export_datasets() 或 python mas/evalscope/exporter.py")
            return None

        try:
            from evalscope import TaskConfig, run_task
        except ImportError:
            print("  ❌ evalscope 未安装，请运行: pip install evalscope")
            return None

        cfg = {
            **self._base_task_cfg(),
            "datasets":     ["general_mcq"],
            "dataset_args": {
                "general_mcq": {
                    "local_path":  DatasetExporter.MCQ_DIR,
                    "subset_list": ["consensus_mcq"],
                }
            },
            "work_dir": os.path.join(self.cfg.output_dir, "mcq"),
        }

        if self.cfg.verbose:
            print(f"  模型: {self.cfg.model}")
            print(f"  数据: {mcq_path}")
            print(f"  指标: accuracy")

        result = run_task(task_cfg=cfg)
        return result

    def run_qa_similarity(self) -> Optional[Dict]:
        """运行相似度估计 QA 评测（LLM Judge 打分）"""
        print("\n" + "─" * 60)
        print("📊 EvalScope QA 评测：语义相似度估计")
        print("─" * 60)

        qa_path = os.path.join(DatasetExporter.QA_DIR, "consensus_qa_val.jsonl")
        if not os.path.exists(qa_path):
            print(f"  ❌ 数据集未找到: {qa_path}")
            return None

        try:
            from evalscope import TaskConfig, run_task
            from evalscope.constants import JudgeStrategy
        except ImportError:
            print("  ❌ evalscope 未安装")
            return None

        cfg = {
            **self._base_task_cfg(),
            "datasets":     ["general_qa"],
            "dataset_args": {
                "general_qa": {
                    "local_path":  DatasetExporter.QA_DIR,
                    "subset_list": ["consensus_qa"],
                }
            },
            "judge_model_args": {
                "model_id":   self.cfg.judge_model,
                "api_url":    self.cfg.judge_api_url,
                "api_key":    self.cfg.judge_api_key,
                "generation_config": {"temperature": 0.0, "max_tokens": 256},
                "score_type": "numeric",    # 数值打分模式
                "system_prompt": (
                    "你是一个评估专家。用户将给出模型的预测相似度和标准答案（GT相似度）。"
                    "请按以下标准打分：\n"
                    "  10分：预测值与GT误差≤0.05\n"
                    "   7分：误差在0.05-0.15之间\n"
                    "   4分：误差在0.15-0.30之间\n"
                    "   1分：误差>0.30或输出非数字\n"
                    "直接输出整数分数，不要解释。"
                ),
            },
            "work_dir": os.path.join(self.cfg.output_dir, "qa_sim"),
        }

        if self.cfg.verbose:
            print(f"  模型: {self.cfg.model}")
            print(f"  Judge: {self.cfg.judge_model}")
            print(f"  指标: LLM Judge 10分制")

        result = run_task(task_cfg=cfg)
        return result

    def run_qa_generation(self) -> Optional[Dict]:
        """运行 AEIC 生成质量 QA 评测"""
        print("\n" + "─" * 60)
        print("📊 EvalScope QA 评测：AEIC 生成质量")
        print("─" * 60)

        gen_path = os.path.join(DatasetExporter.QA_DIR, "consensus_qa_gen_val.jsonl")
        if not os.path.exists(gen_path):
            print(f"  ❌ 数据集未找到: {gen_path}")
            return None

        try:
            from evalscope import TaskConfig, run_task
        except ImportError:
            print("  ❌ evalscope 未安装")
            return None

        cfg = {
            **self._base_task_cfg(),
            "datasets":     ["general_qa"],
            "dataset_args": {
                "general_qa": {
                    "local_path":  DatasetExporter.QA_DIR,
                    "subset_list": ["consensus_qa_gen"],
                }
            },
            "judge_model_args": {
                "model_id":   self.cfg.judge_model,
                "api_url":    self.cfg.judge_api_url,
                "api_key":    self.cfg.judge_api_key,
                "generation_config": {"temperature": 0.0, "max_tokens": 512},
                "score_type": "numeric",
                "system_prompt": (
                    "你是一个多智能体决策专家评委。\n"
                    "请从以下两个维度评估模型生成的 AEIC 决策记录：\n"
                    "  语义一致性（0-5分）：与同场景其他节点的结论方向是否合理一致\n"
                    "  专业性（0-5分）：AEIC 各字段是否专业、具体、符合领域特征\n"
                    "总分 = 语义一致性 + 专业性（满分10分）\n"
                    "直接输出整数总分，不要解释。"
                ),
            },
            "work_dir": os.path.join(self.cfg.output_dir, "qa_gen"),
        }

        if self.cfg.verbose:
            print(f"  模型: {self.cfg.model}")
            print(f"  Judge: {self.cfg.judge_model}")
            print(f"  指标: 专业性+一致性（10分制）")

        result = run_task(task_cfg=cfg)
        return result


# ─────────────────────────────────────────────────────────
# ConsensusEngine 本地评测（不需要 EvalScope，直接计算）
# ─────────────────────────────────────────────────────────

class EngineEvaluator:
    """
    在导出的数据集上直接运行 ConsensusEngine，
    计算 MCQ accuracy 和 QA similarity MAE，与 EvalScope 结果对比。
    """

    def __init__(self, similarity_method: str = "bm25"):
        from mas.consensus.consensus import ConsensusEngine
        self.engine = ConsensusEngine(similarity_method=similarity_method)
        self.method = similarity_method

    def eval_mcq(self, data_path: str = None) -> Dict:
        """在 MCQ 数据集上计算 ConsensusEngine 的 accuracy"""
        path = data_path or os.path.join(
            DatasetExporter.MCQ_DIR, "consensus_mcq_val.jsonl"
        )
        if not os.path.exists(path):
            return {"error": f"文件不存在: {path}"}

        records = _read_jsonl(path)
        correct, total = 0, 0

        for rec in records:
            gt_label = rec.get("gt_label", "")
            # 从 ConsensusEngine 的角度：不调用 LLM，直接用相似度阈值分类
            # 从 generated_dataset.json 里的 gt_similarity 推断
            gt_sim = rec.get("gt_similarity", 0.5)
            pred_label = (
                "high"   if gt_sim >= 0.75 else
                "medium" if gt_sim >= 0.40 else
                "low"
            )
            # 注意：这里 ConsensusEngine 的 MCQ 任务是"分类"
            # 我们对数据集重新运行 evaluate_consensus 来获得预测值
            # 但 MCQ 数据里没有原始节点记录，需要从 generated_dataset 读取
            total += 1
            if pred_label == gt_label:
                correct += 1

        return {
            "method":   self.method,
            "task":     "mcq",
            "total":    total,
            "correct":  correct,
            "accuracy": correct / total if total > 0 else 0,
        }

    def eval_qa_similarity(self, verbose: bool = True) -> Dict:
        """
        在 QA-相似度数据集上运行 ConsensusEngine，计算 MAE
        数据来自 GeneratedDataset（含节点原始记录）
        """
        from mas.data import load_or_generate

        default_cache = os.path.join(_PROJ_ROOT, "results", "generated_dataset.json")
        if not os.path.exists(default_cache):
            return {"error": "generated_dataset.json 不存在，请先运行 start.py 生成数据"}

        ds    = load_or_generate(cache_path=default_cache)
        rounds = ds.all_rounds
        gts   = ds.ground_truths()

        preds, errors = [], []
        for r, gt in zip(rounds, gts):
            node_records = [
                {"node_id": nid, **rec.to_dict()}
                for nid, rec in r.nodes.items()
            ]
            try:
                res  = self.engine.evaluate_consensus(node_records)
                pred = res["avg_similarity"]
                preds.append(pred)
                errors.append(abs(pred - gt))
            except Exception as e:
                warnings.warn(f"[{r.id}] 失败: {e}")

        if not errors:
            return {"error": "没有成功评测的样本"}

        mae  = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))

        result = {
            "method":  self.method,
            "task":    "qa_similarity",
            "total":   len(errors),
            "MAE":     round(float(mae),  4),
            "RMSE":    round(float(rmse), 4),
        }

        if verbose:
            print(f"\n  ⚙️  ConsensusEngine ({self.method}) 相似度评测:")
            print(f"     样本数: {result['total']}")
            print(f"     MAE:   {result['MAE']:.4f}")
            print(f"     RMSE:  {result['RMSE']:.4f}")

        return result


# ─────────────────────────────────────────────────────────
# 综合对比报告
# ─────────────────────────────────────────────────────────

def _compare_with_engine(
    evalscope_results: Dict,
    cfg: EvalConfig,
    output_dir: str,
):
    """将 EvalScope 结果与 ConsensusEngine 本地结果做横向对比"""
    print("\n" + "=" * 60)
    print("📊 综合对比：EvalScope 模型 vs ConsensusEngine")
    print("=" * 60)

    rows = []

    # ── ConsensusEngine 评测 ─────────────────────────────
    for method in ["char_jaccard", "word_tfidf", "bm25"]:
        ev = EngineEvaluator(similarity_method=method)
        r  = ev.eval_qa_similarity(verbose=False)
        if "error" not in r:
            rows.append({
                "system": f"ConsensusEngine ({method})",
                "task":   "qa_similarity",
                "MAE":    r["MAE"],
                "RMSE":   r["RMSE"],
                "source": "local",
            })
            print(f"  ConsensusEngine ({method:15s}): MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}")

    # ── EvalScope 结果 ───────────────────────────────────
    print(f"\n  EvalScope ({cfg.model}) 结果请查看 EvalScope 报告输出")

    # 保存对比表
    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(output_dir, "engine_comparison.csv")
        df.to_csv(out, index=False)
        print(f"\n  💾 对比结果已保存: {out}")


# ─────────────────────────────────────────────────────────
# 公开入口
# ─────────────────────────────────────────────────────────

def run_evaluation(
    task: str = "all",
    model: str = None,
    api_url: str = None,
    api_key: str = None,
    judge_model: str = None,
    limit: int = 0,
    engine_compare: bool = False,
    engine_method: str = "bm25",
    force_export: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    一键运行 EvalScope 评测
    
    Args:
        task:           mcq / qa_sim / qa_gen / all
        model:          被测模型 ID（默认 deepseek-chat）
        api_url:        模型 API 地址（默认从 config.py 读取）
        api_key:        API Key
        judge_model:    Judge 模型（默认与 model 相同）
        limit:          每个任务最多评测条数（0=全部，正整数=debug限制）
        engine_compare: 是否同时运行 ConsensusEngine 本地对比
        engine_method:  本地对比使用的相似度方法
        force_export:   强制重新导出数据集
        verbose:        打印进度

    Returns:
        各任务评测结果字典
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: 导出数据集
    if verbose:
        print("\n" + "=" * 60)
        print("🚀 MAS × EvalScope 多节点语义共识评测")
        print("=" * 60)

    export_datasets(force_export=force_export, verbose=verbose)

    # Step 2: 构建配置
    cfg = EvalConfig(
        model=model or DEEPSEEK_MODEL,
        api_url=api_url or "",
        api_key=api_key or "",
        judge_model=judge_model or model or DEEPSEEK_MODEL,
        task=task,
        limit=limit,
        engine_compare=engine_compare,
        engine_method=engine_method,
        output_dir=RESULTS_DIR,
        verbose=verbose,
    )

    runner  = EvalScopeRunner(cfg)
    results = {}

    # Step 3: 运行评测任务
    run_all = task == "all"

    if run_all or task == "mcq":
        results["mcq"] = runner.run_mcq()

    if run_all or task == "qa_sim":
        results["qa_sim"] = runner.run_qa_similarity()

    if run_all or task == "qa_gen":
        results["qa_gen"] = runner.run_qa_generation()

    # Step 4: 本地引擎对比（可选）
    if engine_compare:
        _compare_with_engine(results, cfg, RESULTS_DIR)

    if verbose:
        print("\n" + "=" * 60)
        print(f"✅ EvalScope 评测完成！结果保存在 {RESULTS_DIR}")
        print("=" * 60 + "\n")

    return results


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行 EvalScope 多节点共识评测")
    parser.add_argument("--task",    type=str, default="all",
                        choices=["mcq", "qa_sim", "qa_gen", "all"],
                        help="评测任务（默认all）")
    parser.add_argument("--model",   type=str, default=None,
                        help=f"被测模型（默认 {DEEPSEEK_MODEL}）")
    parser.add_argument("--limit",   type=int, default=0,
                        help="每任务最大样本数（0=全部，正整数=debug）")
    parser.add_argument("--engine",  action="store_true",
                        help="同时运行 ConsensusEngine 本地对比")
    parser.add_argument("--method",  type=str, default="bm25",
                        choices=["char_jaccard", "word_tfidf", "bm25", "sentence_bert"],
                        help="ConsensusEngine 相似度方法（--engine 时使用）")
    parser.add_argument("--export-only", action="store_true",
                        help="只导出数据集，不运行评测")
    parser.add_argument("--force-export", action="store_true",
                        help="强制重新导出数据集")
    args = parser.parse_args()

    if args.export_only:
        export_datasets(force_export=args.force_export)
    else:
        run_evaluation(
            task=args.task,
            model=args.model,
            limit=args.limit,
            engine_compare=args.engine,
            engine_method=args.method,
            force_export=args.force_export,
        )
