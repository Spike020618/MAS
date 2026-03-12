"""
LLM 驱动的动态数据生成器
================================================================

核心思路：
  用 DeepSeek 模拟 N 个独立节点（默认3个）对同一任务场景分别
  生成 AEIC 格式的决策记录，真实还原多节点互相通信的共识场景。

  每个 ConsensusRound 包含：
    - N 个节点的 AEIC 记录（node_0, node_1, ..., node_N-1）
    - DeepSeek 标注的 GT 平均共识相似度
    - 场景元信息（领域、任务描述等）

  生成的数据特点：
    - 多节点（非二元 A/B 对比）
    - 语言自然、专业
    - 覆盖高/中/低三种共识梯度

AEIC 格式：
  A = Assumptions  前提假设
  E = Evidence     支撑证据（列表）
  I = Inference    推理过程
  C = Conclusion   最终结论
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# ── config ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config as _cfg
    DEEPSEEK_API_KEY  = getattr(_cfg, "DEEPSEEK_API_KEY",  None)
    DEEPSEEK_API_BASE = getattr(_cfg, "DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL    = getattr(_cfg, "DEEPSEEK_MODEL",    "deepseek-chat")
except Exception:
    DEEPSEEK_API_KEY  = None
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL    = "deepseek-chat"


# ═════════════════════════════════════════════════════════
# 数据结构
# ═════════════════════════════════════════════════════════

@dataclass
class AEICRecord:
    """单个节点的 AEIC 决策记录"""
    assumptions: str
    evidence: List[str]
    inference: str
    conclusion: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConsensusRound:
    """
    一轮多节点共识数据
    
    nodes: { "node_0": AEICRecord, "node_1": AEICRecord, ... }
    每个节点独立生成自己的 AEIC 记录，节点间通过共识引擎计算相似度
    """
    id: str
    domain: str
    scenario: str
    nodes: Dict[str, AEICRecord]       # 所有节点的记录，key 为 node_id
    gt_similarity: float               # DeepSeek 标注的 GT 平均共识相似度
    gt_label: str                      # high / medium / low
    description: str = ""

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def get_node_records(self) -> List[Dict]:
        """返回所有节点记录的字典列表，保持 node_id 顺序"""
        return [{"node_id": nid, **rec.to_dict()} for nid, rec in self.nodes.items()]

    def get_record(self, node_id: str) -> Optional[Dict]:
        rec = self.nodes.get(node_id)
        return rec.to_dict() if rec else None

    def to_row(self) -> Dict:
        """展平为单行，用于 DataFrame，列名格式: node_0_assumptions, node_1_evidence ..."""
        row = {
            "id":           self.id,
            "domain":       self.domain,
            "scenario":     self.scenario,
            "n_nodes":      self.n_nodes,
            "gt_similarity": self.gt_similarity,
            "gt_label":     self.gt_label,
            "description":  self.description,
        }
        for node_id, rec in self.nodes.items():
            for field, val in rec.to_dict().items():
                row[f"{node_id}_{field}"] = val
        return row


# 向后兼容别名（勿在新代码中使用）
AEICPair = ConsensusRound


# ═════════════════════════════════════════════════════════
# 任务场景定义
# ═════════════════════════════════════════════════════════

@dataclass
class TaskScenario:
    domain: str
    name: str
    description: str
    similarity_target: str   # high / medium / low


TASK_SCENARIOS: List[TaskScenario] = [

    # ── 金融 ──────────────────────────────────────────────
    TaskScenario("finance", "个人贷款审批",
        "银行信贷部门多个节点对同一个人住房贷款申请进行风险评估与审批决策",
        "high"),
    TaskScenario("finance", "企业融资尽调",
        "投资机构多个分析节点对同一初创企业融资申请进行独立尽职调查",
        "medium"),
    TaskScenario("finance", "反欺诈核查",
        "风控系统多个节点对同一可疑交易独立判断，部分节点认为欺诈，部分认为正常",
        "low"),
    TaskScenario("finance", "资产评估定价",
        "多个估值节点对同一房产进行独立评估，各自给出市场价格建议",
        "high"),
    TaskScenario("finance", "信用评级",
        "多个评级节点对同一企业债券独立评级，结论存在差异",
        "medium"),

    # ── 医疗 ──────────────────────────────────────────────
    TaskScenario("medical", "急症诊断",
        "急诊室多个医疗节点对同一急腹症患者进行独立诊断和处置方案",
        "high"),
    TaskScenario("medical", "慢性病管理方案",
        "多个全科节点对同一2型糖尿病患者独立制定管理方案，治疗路径存在分歧",
        "medium"),
    TaskScenario("medical", "手术必要性评估",
        "多个医疗节点对腰椎间盘突出患者是否手术各持不同意见",
        "low"),
    TaskScenario("medical", "药物剂量调整",
        "多个心内科节点对高血压患者的降压药剂量调整方案基本一致",
        "high"),
    TaskScenario("medical", "精神科评估",
        "多个心理节点对同一抑郁症患者的风险等级评估存在中等分歧",
        "medium"),

    # ── 行政审批 ──────────────────────────────────────────
    TaskScenario("admin", "餐饮许可证审核",
        "多个审批节点对同一餐饮企业的食品经营许可证申请独立审核",
        "high"),
    TaskScenario("admin", "建设项目环评",
        "多个环评节点对工业园区扩建项目的环境影响独立评估，结论不同",
        "low"),
    TaskScenario("admin", "税务稽查处理",
        "多个稽查节点对同一企业税务违规行为独立处理，意见基本一致但处罚力度有别",
        "medium"),
    TaskScenario("admin", "公务员录用体检",
        "多个体检节点对同一公务员录用申请人的体检结论高度一致",
        "high"),

    # ── 法律 ──────────────────────────────────────────────
    TaskScenario("legal", "劳动争议仲裁",
        "多个仲裁节点对劳动合同纠纷中的赔偿金额独立裁决，差距较大",
        "low"),
    TaskScenario("legal", "合同效力认定",
        "多个法律节点对同一商业合同的效力和义务分析基本一致",
        "high"),
    TaskScenario("legal", "知识产权侵权",
        "多个节点对专利侵权事实的法律分析立场截然相反",
        "low"),
    TaskScenario("legal", "刑事辩护策略",
        "多个辩护节点对同一刑事案件的辩护方向有所不同但整体方向一致",
        "medium"),

    # ── 供应链 ──────────────────────────────────────────
    TaskScenario("supply_chain", "供应商资质评审",
        "采购网络多个节点对同一新供应商的资质独立评审，意见高度一致",
        "high"),
    TaskScenario("supply_chain", "库存策略",
        "多个供应链节点对季节性商品备货策略方向一致但数量有差异",
        "medium"),
    TaskScenario("supply_chain", "物流路线规划",
        "多个路由节点对同一批次货物的最优配送路线计算结果差异显著",
        "low"),
]


# ═════════════════════════════════════════════════════════
# DeepSeek 数据生成器
# ═════════════════════════════════════════════════════════

class DataGenerator:
    """
    使用 DeepSeek 生成多节点 AEIC 共识数据

    每次调用 generate_round() 会让 DeepSeek：
      1. 模拟 n_nodes 个独立节点，各自生成一份 AEIC 决策记录
      2. 以裁判身份标注节点间的平均语义相似度（GT）
    """

    _SYSTEM_PROMPT = (
        "你是一个多智能体语义共识实验的数据生成助手。\n"
        "你的任务是模拟多个独立节点对同一任务场景的决策记录，"
        "并以 JSON 格式输出结构化结果。\n"
        "输出必须是合法的 JSON，不要包含任何 Markdown 代码块或多余文字。"
    )

    _GEN_PROMPT_TEMPLATE = """
请为以下任务场景生成 {n_nodes} 个独立节点的 AEIC 格式决策记录。

## 任务场景
领域：{domain}
场景：{name}
描述：{description}

## 节点数量
需要生成 {n_nodes} 个独立节点（node_0 到 node_{last_idx}），每个节点独立分析同一任务。

## 共识相似度目标
目标等级：{similarity_target}
- high（高）：各节点结论高度一致，措辞不同但语义相同，gt_similarity 在 0.80-0.95
- medium（中）：结论方向一致但细节有分歧，gt_similarity 在 0.45-0.75
- low（低）：节点间结论截然不同甚至相反，gt_similarity 在 0.02-0.30

gt_similarity 表示所有节点两两相似度的平均值。

## 输出要求
请严格按照以下 JSON 结构输出，不要有任何额外内容：

{{
  "nodes": {{
    "node_0": {{
      "assumptions": "节点0的前提假设（1-2句话）",
      "evidence": ["证据1", "证据2", "证据3"],
      "inference": "节点0的推理过程（1-2句话）",
      "conclusion": "节点0的最终结论（1句话）"
    }},
    "node_1": {{
      "assumptions": "节点1的前提假设（与其他节点用不同措辞）",
      "evidence": ["证据1", "证据2", "证据3"],
      "inference": "节点1的推理过程（1-2句话）",
      "conclusion": "节点1的最终结论（1句话）"
    }}{extra_nodes}
  }},
  "gt_similarity": 0.00,
  "description": "一句话说明各节点共识/分歧情况"
}}

注意：
1. 语言用中文，专业术语要符合 {domain} 领域
2. evidence 必须是字符串列表，每个节点 3-5 项
3. gt_similarity 是浮点数，必须符合相似度目标范围
4. 各节点记录要足够真实，像真实场景中专业人员独立撰写的
5. 节点间的差异程度要符合 {similarity_target} 等级设定
""".strip()

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        model: str = None,
        n_nodes: int = 3,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_interval: float = 0.5,
    ):
        self.api_key  = api_key  or DEEPSEEK_API_KEY
        self.api_base = api_base or DEEPSEEK_API_BASE
        self.model    = model    or DEEPSEEK_MODEL
        self.n_nodes  = n_nodes
        self.max_retries     = max_retries
        self.retry_delay     = retry_delay
        self.request_interval = request_interval

        if not self.api_key:
            raise ValueError("DeepSeek API Key 未配置，请在 mas/config.py 中设置 DEEPSEEK_API_KEY")

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            print(f"  [DataGenerator] 已连接 DeepSeek ({self.model})  节点数/轮={self.n_nodes}")
        except ImportError:
            self._client = None
            warnings.warn("openai 包未安装，将用 requests fallback。建议: pip install openai")

    def _build_prompt(self, scenario: TaskScenario) -> str:
        """构建包含 N 个节点的 prompt"""
        # 为 n_nodes > 2 的情况补充额外节点的 JSON 模板
        extra = ""
        for i in range(2, self.n_nodes):
            extra += f""",
    "node_{i}": {{
      "assumptions": "节点{i}的前提假设",
      "evidence": ["证据1", "证据2", "证据3"],
      "inference": "节点{i}的推理过程",
      "conclusion": "节点{i}的最终结论"
    }}"""
        return self._GEN_PROMPT_TEMPLATE.format(
            n_nodes=self.n_nodes,
            last_idx=self.n_nodes - 1,
            domain=scenario.domain,
            name=scenario.name,
            description=scenario.description,
            similarity_target=scenario.similarity_target,
            extra_nodes=extra,
        )

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        for attempt in range(self.max_retries):
            try:
                if self._client:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1800,
                        temperature=0.85,
                    )
                    return resp.choices[0].message.content.strip()
                else:
                    return self._call_via_requests(messages)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (attempt + 1)
                    print(f"    ⚠️  API 调用失败 ({e})，{wait}s 后重试...")
                    time.sleep(wait)
                else:
                    raise

    def _call_via_requests(self, messages: list) -> str:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": 1800, "temperature": 0.85}
        url = self.api_base.rstrip("/") + "/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _parse_response(self, raw: str, scenario: TaskScenario, round_id: str) -> ConsensusRound:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(cleaned)

        nodes_raw = data.get("nodes", {})
        nodes: Dict[str, AEICRecord] = {}
        for node_id, d in nodes_raw.items():
            ev = d.get("evidence", [])
            if isinstance(ev, str):
                ev = [ev]
            nodes[node_id] = AEICRecord(
                assumptions=str(d.get("assumptions", "")),
                evidence=list(ev),
                inference=str(d.get("inference", "")),
                conclusion=str(d.get("conclusion", "")),
            )

        if not nodes:
            raise ValueError("LLM 返回的 nodes 字段为空")

        gt_sim = float(data.get("gt_similarity", 0.5))
        gt_sim = max(0.0, min(1.0, gt_sim))
        label  = "high" if gt_sim >= 0.75 else ("medium" if gt_sim >= 0.40 else "low")

        return ConsensusRound(
            id=round_id,
            domain=scenario.domain,
            scenario=scenario.name,
            nodes=nodes,
            gt_similarity=round(gt_sim, 3),
            gt_label=label,
            description=str(data.get("description", "")),
        )

    def generate_round(self, scenario: TaskScenario, round_id: str) -> Optional[ConsensusRound]:
        """生成单轮多节点共识数据"""
        prompt = self._build_prompt(scenario)
        try:
            raw   = self._call_llm(prompt)
            round_ = self._parse_response(raw, scenario, round_id)
            time.sleep(self.request_interval)
            return round_
        except Exception as e:
            print(f"    ❌ [{round_id}] 生成失败: {e}")
            return None

    def generate_dataset(
        self,
        scenarios: List[TaskScenario] = None,
        n_per_scenario: int = 1,
        verbose: bool = True,
    ) -> "GeneratedDataset":
        if scenarios is None:
            scenarios = TASK_SCENARIOS

        rounds: List[ConsensusRound] = []
        total = len(scenarios) * n_per_scenario

        if verbose:
            print(f"\n  📡 开始生成实验数据（共 {total} 轮，每轮 {self.n_nodes} 节点）...")
            print(f"     模型: {self.model}  场景数: {len(scenarios)}")
            print()

        for s_idx, scenario in enumerate(scenarios):
            for n in range(n_per_scenario):
                round_id = f"{scenario.domain.upper()[:3]}_{s_idx:02d}_{n:02d}"
                if verbose:
                    icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                        scenario.similarity_target, "⚪"
                    )
                    print(f"  {icon} [{round_id}] {scenario.domain} · {scenario.name}")

                r = self.generate_round(scenario, round_id)
                if r:
                    rounds.append(r)
                    if verbose:
                        node_ids = list(r.nodes.keys())
                        print(f"       → GT={r.gt_similarity:.3f} ({r.gt_label})  "
                              f"节点: {node_ids}  ✓ {r.description[:45]}...")
                else:
                    if verbose:
                        print(f"       → 跳过（生成失败）")

        ds = GeneratedDataset(rounds)
        if verbose:
            print(f"\n  ✅ 生成完成: {len(rounds)}/{total} 轮成功")
            print(f"     {ds.summary()}")
        return ds


# ═════════════════════════════════════════════════════════
# 数据集容器
# ═════════════════════════════════════════════════════════

class GeneratedDataset:
    """多节点 AEIC 共识数据集"""

    def __init__(self, rounds: List[ConsensusRound]):
        self._rounds = rounds

    @property
    def all_rounds(self) -> List[ConsensusRound]:
        return self._rounds

    # 向后兼容别名
    @property
    def all_pairs(self) -> List[ConsensusRound]:
        return self._rounds

    def by_domain(self, domain: str) -> List[ConsensusRound]:
        return [r for r in self._rounds if r.domain == domain]

    def by_label(self, label: str) -> List[ConsensusRound]:
        return [r for r in self._rounds if r.gt_label == label]

    def get_node_records(self) -> List[List[Dict]]:
        """
        返回所有轮次的节点记录。
        结构：[ [node_0_dict, node_1_dict, ...], [...], ... ]
        """
        return [r.get_node_records() for r in self._rounds]

    def ground_truths(self) -> List[float]:
        return [r.gt_similarity for r in self._rounds]

    def labels(self) -> List[str]:
        return [r.gt_label for r in self._rounds]

    def summary(self) -> Dict:
        from collections import Counter
        label_cnt  = Counter(r.gt_label  for r in self._rounds)
        domain_cnt = Counter(r.domain    for r in self._rounds)
        sims = self.ground_truths()
        node_counts = Counter(r.n_nodes for r in self._rounds)
        return {
            "total_rounds":      len(self._rounds),
            "by_label":          dict(label_cnt),
            "by_domain":         dict(domain_cnt),
            "nodes_per_round":   dict(node_counts),
            "avg_gt_similarity": round(sum(sims) / len(sims), 3) if sims else 0,
        }

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([r.to_row() for r in self._rounds])

    # ── evalscope 导出 ───────────────────────────────────

    def export_evalscope(
        self,
        output_dir: str = "evalscope_data",
        few_shot_n: int = 5,
    ) -> Dict[str, str]:
        """
        将数据集导出为 evalscope 评测格式。

        导出两种任务数据：
          - general_mcq：共识等级三分类（A=high/B=medium/C=low），自动计算 Accuracy
          - general_qa：完整共识分析，用 LLM judge 打分（1-5分）

        Args:
            output_dir: 输出目录（默认 evalscope_data/）
            few_shot_n: MCQ dev 集条数，用于 few-shot 评测

        Returns:
            { "mcq_dev": path, "mcq_val": path, "qa": path, "manifest": path }

        示例：
            ds = load_or_generate()
            paths = ds.export_evalscope()
            # 然后运行：python mas/eval/run_eval.py
        """
        from mas.eval.export_to_evalscope import EvalScopeExporter
        exporter = EvalScopeExporter(self, output_dir=output_dir, few_shot_n=few_shot_n)
        return exporter.export_all()

    # ── 序列化 / 反序列化 ────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = []
        for r in self._rounds:
            data.append({
                "id":           r.id,
                "domain":       r.domain,
                "scenario":     r.scenario,
                "nodes":        {nid: rec.to_dict() for nid, rec in r.nodes.items()},
                "gt_similarity": r.gt_similarity,
                "gt_label":     r.gt_label,
                "description":  r.description,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  💾 数据集已保存: {path}  ({len(data)} 轮)")

    @classmethod
    def load(cls, path: str) -> "GeneratedDataset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rounds = []
        for d in data:
            nodes = {
                nid: AEICRecord(**rec)
                for nid, rec in d["nodes"].items()
            }
            rounds.append(ConsensusRound(
                id=d["id"],
                domain=d["domain"],
                scenario=d.get("scenario", ""),
                nodes=nodes,
                gt_similarity=d["gt_similarity"],
                gt_label=d["gt_label"],
                description=d.get("description", ""),
            ))
        return cls(rounds)


# ═════════════════════════════════════════════════════════
# 便捷入口
# ═════════════════════════════════════════════════════════

_DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "generated.json"
)


def load_or_generate(
    scenarios: List[TaskScenario] = None,
    n_per_scenario: int = 1,
    n_nodes: int = 3,
    cache_path: str = None,
    force_regenerate: bool = False,
    api_key: str = None,
) -> GeneratedDataset:
    """
    智能加载：有缓存就加载，否则调用 DeepSeek 生成并缓存。

    Args:
        scenarios:        任务场景列表，None 表示用默认 TASK_SCENARIOS
        n_per_scenario:   每个场景生成几轮
        n_nodes:          每轮参与共识的节点数（默认3）
        cache_path:       缓存 JSON 路径
        force_regenerate: 强制重新生成
        api_key:          DeepSeek API Key
    """
    cache = cache_path or _DEFAULT_CACHE

    if not force_regenerate and os.path.exists(cache):
        print(f"  📂 从缓存加载数据集: {cache}")
        ds = GeneratedDataset.load(cache)
        print(f"     {ds.summary()}")
        return ds

    gen = DataGenerator(api_key=api_key, n_nodes=n_nodes)
    ds  = gen.generate_dataset(
        scenarios=scenarios,
        n_per_scenario=n_per_scenario,
        verbose=True,
    )

    if len(ds.all_rounds) > 0:
        ds.save(cache)

    return ds


# ═════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, random

    parser = argparse.ArgumentParser(description="生成多节点 AEIC 共识数据集")
    parser.add_argument("--n",       type=int, default=1,   help="每个场景生成几轮（默认1）")
    parser.add_argument("--nodes",   type=int, default=3,   help="每轮节点数（默认3）")
    parser.add_argument("--force",   action="store_true",   help="强制重新生成")
    parser.add_argument("--out",     type=str, default=None, help="输出路径")
    parser.add_argument("--preview", action="store_true",   help="打印生成样例")
    parser.add_argument("--export",  action="store_true",   help="导出为 evalscope 格式")
    args = parser.parse_args()

    ds = load_or_generate(
        n_per_scenario=args.n,
        n_nodes=args.nodes,
        cache_path=args.out,
        force_regenerate=args.force,
    )

    print("\n=== 数据集统计 ===")
    for k, v in ds.summary().items():
        print(f"  {k}: {v}")

    if args.export:
        paths = ds.export_evalscope()
        print("\n=== evalscope 导出路径 ===")
        for k, v in paths.items():
            print(f"  {k}: {v}")

    if args.preview and ds.all_rounds:
        print("\n=== 随机样例 ===")
        for r in random.sample(ds.all_rounds, min(2, len(ds.all_rounds))):
            print(f"\n  [{r.id}] {r.scenario}  GT={r.gt_similarity} ({r.gt_label})")
            for nid, rec in r.nodes.items():
                print(f"  {nid} 结论: {rec.conclusion}")
            print(f"  说明: {r.description}")
