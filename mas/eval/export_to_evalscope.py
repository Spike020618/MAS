"""
EvalScope 数据导出器
================================================================

把 GeneratedDataset（多节点 AEIC 共识数据）导出为 evalscope 支持的两种格式：

  1. general_mcq（选择题）
     给模型 N 份节点 AEIC 记录，让它判断整体共识等级。
     三选一：A=high（高共识）/ B=medium（中等共识）/ C=low（低共识）
     → 自动计算分类准确率，无需 judge

  2. general_qa（开放问答）
     给模型完整场景 + AEIC 记录，让它给出共识分析报告。
     用 DeepSeek 作为 judge，从「准确性/专业性/结构完整性」三维度打分。
     → 适合评估模型对 AEIC 共识理解的深度

目录结构（导出后）：
  evalscope_data/
  ├── consensus_mcq/
  │   ├── consensus_dev.csv      ← few-shot 示例（5条）
  │   └── consensus_val.csv      ← 评测集（全量）
  └── consensus_qa/
      └── consensus_test.jsonl   ← 全量 QA 数据

MCQ CSV 格式（evalscope 标准）：
  id, question, A, B, C, answer, explanation

QA JSONL 格式（evalscope 标准）：
  {"system_prompt": "...", "query": "...", "response": "..."}
  其中 response 是参考答案（GT 标签 + 说明）
"""

from __future__ import annotations

import csv
import json
import os
import sys
import textwrap
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mas.data.generator import GeneratedDataset, ConsensusRound

# MCQ 选项映射
_LABEL_TO_OPTION = {"high": "A", "medium": "B", "low": "C"}
_OPTION_TO_LABEL = {v: k for k, v in _LABEL_TO_OPTION.items()}

# GT 标签的中文说明（用于 MCQ 选项文本）
_LABEL_DESC = {
    "high":   "高共识（各节点结论方向一致，语义高度相似）",
    "medium": "中等共识（节点结论方向大体一致，但细节存在分歧）",
    "low":    "低共识（节点间结论截然不同甚至相互矛盾）",
}


def _fmt_nodes(round_: "ConsensusRound", max_ev: int = 3) -> str:
    """把多个节点的 AEIC 记录格式化为可读文本"""
    parts = []
    for nid, rec in round_.nodes.items():
        ev_text = "；".join(rec.evidence[:max_ev])
        parts.append(
            f"【{nid}】\n"
            f"  假设：{rec.assumptions}\n"
            f"  证据：{ev_text}\n"
            f"  推理：{rec.inference}\n"
            f"  结论：{rec.conclusion}"
        )
    return "\n\n".join(parts)


def _mcq_question(round_: "ConsensusRound") -> str:
    """MCQ 题目文本"""
    nodes_text = _fmt_nodes(round_)
    return (
        f"【任务场景】领域：{round_.domain}  场景：{round_.scenario}\n\n"
        f"以下是 {round_.n_nodes} 个独立节点对同一任务的 AEIC 决策记录：\n\n"
        f"{nodes_text}\n\n"
        f"请判断上述节点间的整体语义共识等级："
    )


def _qa_query(round_: "ConsensusRound") -> str:
    """QA 题目文本（比 MCQ 更开放，要求给出分析）"""
    nodes_text = _fmt_nodes(round_, max_ev=5)
    return (
        f"【任务场景】领域：{round_.domain}  场景：{round_.scenario}\n\n"
        f"以下是 {round_.n_nodes} 个节点对同一任务的完整 AEIC 决策记录：\n\n"
        f"{nodes_text}\n\n"
        "请完成以下分析：\n"
        "1. 判断各节点间的语义共识等级（high/medium/low）\n"
        "2. 给出共识等级的量化估计（0-1 之间的相似度均值）\n"
        "3. 指出节点间主要的共识点和分歧点\n"
        "4. 给出共识引擎的推荐决策（ESS_Consensus / Audit_Required / Reject）"
    )


def _qa_reference(round_: "ConsensusRound") -> str:
    """QA 参考答案"""
    sim_range = {
        "high":   "0.75-0.95",
        "medium": "0.40-0.75",
        "low":    "0.02-0.40",
    }.get(round_.gt_label, "未知")

    conclusions = {nid: rec.conclusion for nid, rec in round_.nodes.items()}
    conc_text = "；".join(f"{nid}：{c}" for nid, c in conclusions.items())

    decision = (
        "ESS_Consensus" if round_.gt_similarity >= 0.75 else
        "Audit_Required" if round_.gt_similarity >= 0.25 else
        "Reject"
    )

    return (
        f"共识等级：{round_.gt_label}（{_LABEL_DESC[round_.gt_label]}）\n"
        f"相似度估计：{round_.gt_similarity:.3f}（正常范围 {sim_range}）\n"
        f"各节点结论：{conc_text}\n"
        f"场景说明：{round_.description}\n"
        f"推荐决策：{decision}"
    )


class EvalScopeExporter:
    """
    把 GeneratedDataset 导出为 evalscope 标准格式

    用法：
        from mas.eval import EvalScopeExporter
        from mas.data import load_or_generate

        ds = load_or_generate()
        exporter = EvalScopeExporter(ds, output_dir="evalscope_data")
        paths = exporter.export_all()
    """

    _MCQ_SYSTEM = (
        "你是一位多智能体语义共识评估专家。\n"
        "请根据多个节点的 AEIC 决策记录，判断节点间的整体语义共识等级。\n"
        "直接回答选项字母，不要解释。"
    )

    _QA_SYSTEM = (
        "你是一位多智能体语义共识评估专家，熟悉 AEIC 格式（Assumptions/Evidence/Inference/Conclusion）。\n"
        "请对给定的多节点 AEIC 记录进行完整的共识分析，包括等级判断、相似度估计、共识点与分歧点分析、"
        "以及最终的引擎决策建议。回答要专业、结构清晰。"
    )

    def __init__(
        self,
        dataset: "GeneratedDataset",
        output_dir: str = "evalscope_data",
        few_shot_n: int = 5,
    ):
        self.ds         = dataset
        self.out        = os.path.abspath(output_dir)
        self.few_shot_n = few_shot_n

        os.makedirs(os.path.join(self.out, "consensus_mcq"), exist_ok=True)
        os.makedirs(os.path.join(self.out, "consensus_qa"),  exist_ok=True)

    # ── MCQ ─────────────────────────────────────────────────

    def export_mcq(self) -> Dict[str, str]:
        """
        导出 general_mcq 格式。

        CSV 列：id, question, A, B, C, answer, explanation
        dev.csv  ← few-shot 示例（前 few_shot_n 条）
        val.csv  ← 评测集（全量）

        返回：{"dev": path, "val": path}
        """
        rounds = self.ds.all_rounds
        dev_rounds = rounds[:self.few_shot_n]
        val_rounds = rounds

        fieldnames = ["id", "question", "A", "B", "C", "answer", "explanation"]

        def write_csv(path: str, data: list):
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in data:
                    writer.writerow({
                        "id":          r.id,
                        "question":    _mcq_question(r),
                        "A":           _LABEL_DESC["high"],
                        "B":           _LABEL_DESC["medium"],
                        "C":           _LABEL_DESC["low"],
                        "answer":      _LABEL_TO_OPTION[r.gt_label],
                        "explanation": (
                            f"GT相似度={r.gt_similarity:.3f}，"
                            f"节点数={r.n_nodes}。{r.description}"
                        ),
                    })
            return path

        dev_path = os.path.join(self.out, "consensus_mcq", "consensus_dev.csv")
        val_path = os.path.join(self.out, "consensus_mcq", "consensus_val.csv")

        write_csv(dev_path, dev_rounds)
        write_csv(val_path, val_rounds)

        print(f"  📄 MCQ dev  → {dev_path}  ({len(dev_rounds)} 条)")
        print(f"  📄 MCQ val  → {val_path}  ({len(val_rounds)} 条)")
        return {"dev": dev_path, "val": val_path}

    # ── QA ──────────────────────────────────────────────────

    def export_qa(self) -> str:
        """
        导出 general_qa 格式。

        每行 JSONL：{"system_prompt": ..., "query": ..., "response": ...}
        response 是参考答案（GT 标签 + 分析），供 LLM judge 使用。

        返回：jsonl 文件路径
        """
        path = os.path.join(self.out, "consensus_qa", "consensus_test.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in self.ds.all_rounds:
                line = {
                    "system_prompt": self._QA_SYSTEM,
                    "query":         _qa_query(r),
                    "response":      _qa_reference(r),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"  📄 QA  test → {path}  ({len(self.ds.all_rounds)} 条)")
        return path

    # ── manifest ─────────────────────────────────────────────

    def export_manifest(self) -> str:
        """
        写一份 manifest.json，记录数据集元信息和 evalscope TaskConfig 参考。
        """
        manifest = {
            "dataset_name": "MAS Consensus AEIC Benchmark",
            "description":  "多节点 AEIC 语义共识评测基准，涵盖金融/医疗/行政/法律/供应链五个领域",
            "stats":        self.ds.summary(),
            "label_map":    _LABEL_TO_OPTION,
            "evalscope": {
                "mcq_task": {
                    "dataset":  "general_mcq",
                    "local_path": os.path.join(self.out, "consensus_mcq"),
                    "subset_list": ["consensus"],
                },
                "qa_task": {
                    "dataset":   "general_qa",
                    "dataset_id": os.path.join(self.out, "consensus_qa"),
                    "subset_list": ["consensus_test"],
                },
            },
        }
        path = os.path.join(self.out, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"  📋 manifest → {path}")
        return path

    def export_all(self) -> Dict[str, str]:
        """导出全部格式，返回各文件路径字典"""
        print(f"\n  🚀 开始导出 evalscope 数据集 → {self.out}")
        print(f"     共 {len(self.ds.all_rounds)} 轮，每轮 {self.ds.all_rounds[0].n_nodes if self.ds.all_rounds else 0} 节点")
        print()
        mcq  = self.export_mcq()
        qa   = self.export_qa()
        mani = self.export_manifest()
        print(f"\n  ✅ 导出完成")
        return {"mcq_dev": mcq["dev"], "mcq_val": mcq["val"], "qa": qa, "manifest": mani}


# ── 挂载到 GeneratedDataset ──────────────────────────────

def _inject_export_method():
    """动态把 export_evalscope() 方法挂到 GeneratedDataset 上"""
    try:
        from mas.data.generator import GeneratedDataset

        def export_evalscope(self, output_dir: str = "evalscope_data", few_shot_n: int = 5):
            """
            将当前数据集导出为 evalscope 格式。

            Args:
                output_dir: 输出目录（默认 evalscope_data/）
                few_shot_n: MCQ dev 集条数（默认5）

            Returns:
                { mcq_dev, mcq_val, qa, manifest } 文件路径字典
            """
            exporter = EvalScopeExporter(self, output_dir=output_dir, few_shot_n=few_shot_n)
            return exporter.export_all()

        if not hasattr(GeneratedDataset, "export_evalscope"):
            GeneratedDataset.export_evalscope = export_evalscope
    except ImportError:
        pass


_inject_export_method()
