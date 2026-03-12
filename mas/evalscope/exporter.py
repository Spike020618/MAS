"""
EvalScope 数据集导出器
================================================================

将 GeneratedDataset（多节点 AEIC 共识数据）转换为 EvalScope 
原生格式，输出到 mas/evalscope/datasets/ 目录下。

导出两种数据集：

1. consensus_mcq/consensus_mcq_val.jsonl   （MCQ 格式）
   ─────────────────────────────────────────────────────────
   每条样本：给出所有节点 AEIC 记录 → 判断共识等级
   字段：id, question, A, B, C, answer, domain, scenario
   选项：A=高度共识, B=中等共识, C=低共识
   答案映射：gt_label high→A, medium→B, low→C
   
2. consensus_qa/consensus_qa_val.jsonl     （QA 格式）
   ─────────────────────────────────────────────────────────
   每条样本：给出场景 + N-1 节点记录 → 估计整体相似度分值
   字段：system, query, response (GT 相似度)
   评测方式：LLM-as-Judge 对模型输出分值与 GT 打分
   
   额外导出：
   consensus_qa/consensus_qa_gen_val.jsonl  （生成任务）
   给出场景描述，生成完整的第 N 个节点 AEIC 记录
   用 LLM Judge 从语义一致性、专业性两个维度打分
"""

from __future__ import annotations

import json
import os
import sys
import random
from typing import Dict, List, Optional

# ── 路径设置 ──────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_MAS_ROOT  = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_MAS_ROOT)
sys.path.insert(0, _MAS_ROOT)

DATASETS_DIR = os.path.join(_HERE, "datasets")


# ─────────────────────────────────────────────────────────
# 格式化辅助
# ─────────────────────────────────────────────────────────

def _fmt_aeic(node_id: str, rec: Dict) -> str:
    """把一条 AEIC 记录格式化为易读的文本块"""
    evidence = rec.get("evidence", [])
    if isinstance(evidence, list):
        evidence_str = "；".join(evidence)
    else:
        evidence_str = str(evidence)
    return (
        f"【{node_id}】\n"
        f"  前提假设：{rec.get('assumptions', '')}\n"
        f"  支撑证据：{evidence_str}\n"
        f"  推理过程：{rec.get('inference', '')}\n"
        f"  最终结论：{rec.get('conclusion', '')}"
    )


def _fmt_round_context(round_obj, excluded_node: Optional[str] = None) -> str:
    """格式化一轮所有节点记录（可选排除某节点）"""
    lines = []
    for nid, rec in round_obj.nodes.items():
        if excluded_node and nid == excluded_node:
            continue
        lines.append(_fmt_aeic(nid, rec.to_dict()))
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────
# 导出器主类
# ─────────────────────────────────────────────────────────

class DatasetExporter:
    """
    将 GeneratedDataset 导出为 EvalScope 评测数据集
    
    目录结构：
      mas/evalscope/datasets/
        consensus_mcq/
          consensus_mcq_val.jsonl    共识等级分类（MCQ）
        consensus_qa/
          consensus_qa_val.jsonl     相似度估计（QA）
          consensus_qa_gen_val.jsonl AEIC 生成质量（QA）
    """

    MCQ_DIR  = os.path.join(DATASETS_DIR, "consensus_mcq")
    QA_DIR   = os.path.join(DATASETS_DIR, "consensus_qa")

    # MCQ 选项到 gt_label 的映射
    LABEL_TO_OPTION = {"high": "A", "medium": "B", "low": "C"}
    OPTION_TO_LABEL = {"A": "high", "B": "medium", "C": "low"}

    MCQ_OPTIONS = {
        "A": "高度共识（各节点结论高度一致，核心语义相同，相似度≥0.75）",
        "B": "中等共识（结论方向大体一致，但细节或措施有分歧，相似度0.40-0.75）",
        "C": "低共识（节点间结论存在明显分歧或相反，相似度<0.40）",
    }

    MCQ_SYSTEM = (
        "你是一个多智能体语义共识分析专家。"
        "你将看到多个独立智能体节点对同一任务场景的 AEIC 决策记录，"
        "需要综合判断各节点之间的整体共识程度。"
        "AEIC 格式包含：前提假设(A)、支撑证据(E)、推理过程(I)、最终结论(C)。"
    )

    QA_SYSTEM = (
        "你是一个多智能体语义共识评估专家。"
        "给定多个智能体节点的 AEIC 决策记录，请估计它们之间的整体语义相似度。"
        "输出一个 0.00 到 1.00 之间的浮点数（保留两位小数），"
        "数值越高表示各节点的决策越一致。"
        "仅输出数字，不要任何解释。"
    )

    GEN_SYSTEM = (
        "你是一个多智能体决策系统中的节点。"
        "你将看到任务场景描述和其他节点已有的 AEIC 决策记录，"
        "请以独立节点的视角，生成你自己对该任务的 AEIC 决策记录。"
        "输出格式：\n"
        "前提假设：[你的假设]\n"
        "支撑证据：[证据1]；[证据2]；[证据3]\n"
        "推理过程：[你的推理]\n"
        "最终结论：[你的结论]"
    )

    def __init__(self, dataset=None, cache_path: str = None):
        """
        Args:
            dataset:    GeneratedDataset 对象，None 则自动从缓存加载
            cache_path: 数据集缓存路径（None 则用默认路径）
        """
        if dataset is None:
            from mas.data import load_or_generate
            default_cache = os.path.join(_PROJ_ROOT, "results", "generated_dataset.json")
            cache = cache_path or default_cache
            dataset = load_or_generate(cache_path=cache)
        self.dataset = dataset

    # ── 导出 MCQ 数据集 ──────────────────────────────────

    def export_mcq(self, out_dir: str = None, verbose: bool = True) -> str:
        """
        导出共识等级分类 MCQ 数据集
        
        每条样本：展示所有节点 AEIC 记录 → 判断 high/medium/low
        """
        out_dir  = out_dir or self.MCQ_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "consensus_mcq_val.jsonl")

        records = []
        for r in self.dataset.all_rounds:
            # 构建题目：列出所有节点记录
            node_section = _fmt_round_context(r)
            question = (
                f"以下是多个智能体节点对任务【{r.scenario}】（领域：{r.domain}）"
                f"的独立 AEIC 决策记录：\n\n"
                f"{node_section}\n\n"
                f"请综合以上 {r.n_nodes} 个节点的决策记录，"
                f"判断这些节点之间的整体共识程度："
            )

            records.append({
                "id":       r.id,
                "system":   self.MCQ_SYSTEM,
                "question": question,
                "A":        self.MCQ_OPTIONS["A"],
                "B":        self.MCQ_OPTIONS["B"],
                "C":        self.MCQ_OPTIONS["C"],
                "answer":   self.LABEL_TO_OPTION[r.gt_label],
                # 元信息（EvalScope 会忽略额外字段，但便于分析）
                "domain":       r.domain,
                "scenario":     r.scenario,
                "gt_label":     r.gt_label,
                "gt_similarity": r.gt_similarity,
                "n_nodes":      r.n_nodes,
            })

        _write_jsonl(out_path, records)

        if verbose:
            from collections import Counter
            label_dist = Counter(r["gt_label"] for r in records)
            print(f"  ✅ MCQ 数据集导出完成")
            print(f"     路径: {out_path}")
            print(f"     样本数: {len(records)}")
            print(f"     标签分布: {dict(label_dist)}")

        return out_path

    # ── 导出 QA：相似度估计 ──────────────────────────────

    def export_qa_similarity(self, out_dir: str = None, verbose: bool = True) -> str:
        """
        导出相似度估计 QA 数据集
        
        每条样本：展示所有节点记录 → 输出相似度分值（0-1）
        """
        out_dir  = out_dir or self.QA_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "consensus_qa_val.jsonl")

        records = []
        for r in self.dataset.all_rounds:
            node_section = _fmt_round_context(r)
            query = (
                f"任务场景：{r.scenario}（领域：{r.domain}）\n\n"
                f"以下是 {r.n_nodes} 个节点的独立决策记录：\n\n"
                f"{node_section}\n\n"
                f"请给出这些节点决策记录的整体语义相似度（0.00-1.00）："
            )
            records.append({
                "id":       r.id,
                "system":   self.QA_SYSTEM,
                "query":    query,
                "response": f"{r.gt_similarity:.2f}",
                # 元信息
                "domain":       r.domain,
                "scenario":     r.scenario,
                "gt_label":     r.gt_label,
                "n_nodes":      r.n_nodes,
            })

        _write_jsonl(out_path, records)

        if verbose:
            sims = [r.gt_similarity for r in self.dataset.all_rounds]
            print(f"  ✅ QA-相似度估计数据集导出完成")
            print(f"     路径: {out_path}")
            print(f"     样本数: {len(records)}")
            print(f"     GT 相似度均值: {sum(sims)/len(sims):.3f}")

        return out_path

    # ── 导出 QA：AEIC 生成 ───────────────────────────────

    def export_qa_generation(
        self,
        out_dir: str = None,
        n_nodes_for_context: int = None,
        verbose: bool = True,
    ) -> str:
        """
        导出 AEIC 生成质量 QA 数据集
        
        每条样本：展示场景 + N-1 节点记录 → 生成第 N 个节点的 AEIC
        GT response 为真实的最后一个节点记录（文本格式）
        
        Args:
            n_nodes_for_context: 提供给模型看的节点数（None 则用全部-1）
        """
        out_dir  = out_dir or self.QA_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "consensus_qa_gen_val.jsonl")

        records = []
        for r in self.dataset.all_rounds:
            node_ids   = list(r.nodes.keys())
            if len(node_ids) < 2:
                continue

            # 最后一个节点作为"待生成目标"，其余作为上下文
            target_id  = node_ids[-1]
            target_rec = r.nodes[target_id]

            ctx_count = (
                (len(node_ids) - 1)
                if n_nodes_for_context is None
                else min(n_nodes_for_context, len(node_ids) - 1)
            )
            ctx_ids   = node_ids[:ctx_count]
            ctx_recs  = "\n\n".join(
                _fmt_aeic(nid, r.nodes[nid].to_dict()) for nid in ctx_ids
            )

            query = (
                f"任务场景：{r.scenario}\n"
                f"领域：{r.domain}\n\n"
                f"以下是其他节点的 AEIC 决策记录（共 {ctx_count} 个节点）：\n\n"
                f"{ctx_recs}\n\n"
                f"请以独立节点 {target_id} 的视角，生成你对该任务的 AEIC 决策记录："
            )

            # GT response：目标节点的真实记录（文本化）
            ev_str = "；".join(target_rec.evidence)
            response = (
                f"前提假设：{target_rec.assumptions}\n"
                f"支撑证据：{ev_str}\n"
                f"推理过程：{target_rec.inference}\n"
                f"最终结论：{target_rec.conclusion}"
            )

            records.append({
                "id":        f"{r.id}_gen",
                "system":    self.GEN_SYSTEM,
                "query":     query,
                "response":  response,
                # 元信息
                "domain":       r.domain,
                "scenario":     r.scenario,
                "gt_label":     r.gt_label,
                "target_node":  target_id,
                "context_nodes": ctx_ids,
            })

        _write_jsonl(out_path, records)

        if verbose:
            print(f"  ✅ QA-AEIC 生成数据集导出完成")
            print(f"     路径: {out_path}")
            print(f"     样本数: {len(records)}")

        return out_path

    # ── 一键导出全部 ─────────────────────────────────────

    def export_all(self, verbose: bool = True) -> Dict[str, str]:
        """导出所有评测数据集，返回 {任务名: 文件路径} 字典"""
        if verbose:
            print("\n📦 导出 EvalScope 评测数据集...")
            print(f"   数据源: {len(self.dataset.all_rounds)} 轮共识数据\n")

        paths = {
            "mcq":        self.export_mcq(verbose=verbose),
            "qa_sim":     self.export_qa_similarity(verbose=verbose),
            "qa_gen":     self.export_qa_generation(verbose=verbose),
        }

        if verbose:
            print(f"\n   目录: {DATASETS_DIR}")

        return paths


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def _write_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_datasets(
    cache_path: str = None,
    force_export: bool = False,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    便捷入口：从缓存加载数据集并导出所有 EvalScope 格式文件
    
    Args:
        cache_path:    GeneratedDataset JSON 缓存路径
        force_export:  True 则即使文件已存在也重新导出
        verbose:       打印进度
        
    Returns:
        {task_name: file_path} 字典
    """
    # 检查是否已导出（全部3个文件都存在）
    expected = [
        os.path.join(DatasetExporter.MCQ_DIR, "consensus_mcq_val.jsonl"),
        os.path.join(DatasetExporter.QA_DIR,  "consensus_qa_val.jsonl"),
        os.path.join(DatasetExporter.QA_DIR,  "consensus_qa_gen_val.jsonl"),
    ]
    if not force_export and all(os.path.exists(p) for p in expected):
        if verbose:
            print("  📂 EvalScope 数据集已存在，跳过导出（--force-export 可强制重新导出）")
        return {
            "mcq":    expected[0],
            "qa_sim": expected[1],
            "qa_gen": expected[2],
        }

    exporter = DatasetExporter(cache_path=cache_path)
    return exporter.export_all(verbose=verbose)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="导出 EvalScope 评测数据集")
    parser.add_argument("--cache",   type=str, default=None, help="GeneratedDataset 缓存路径")
    parser.add_argument("--force",   action="store_true",    help="强制重新导出")
    parser.add_argument("--task",    type=str, default="all",
                        choices=["all", "mcq", "qa_sim", "qa_gen"], help="导出哪个数据集")
    args = parser.parse_args()

    exp = DatasetExporter(cache_path=args.cache)

    if args.task == "all":
        paths = exp.export_all()
    elif args.task == "mcq":
        paths = {"mcq": exp.export_mcq()}
    elif args.task == "qa_sim":
        paths = {"qa_sim": exp.export_qa_similarity()}
    elif args.task == "qa_gen":
        paths = {"qa_gen": exp.export_qa_generation()}

    print("\n📋 导出结果:")
    for name, path in paths.items():
        # 读取实际行数
        with open(path) as f:
            n = sum(1 for _ in f)
        print(f"  {name:10s}: {n:3d} 条  →  {path}")
