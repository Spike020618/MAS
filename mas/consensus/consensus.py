"""
共识引擎 v3 - 博弈驱动的去中心化语义机制
================================================================

核心变化（v2 → v3）：
  不再是二元 agent_a vs agent_b 对比，
  而是 N 个节点各自提交 AEIC 记录，
  引擎计算所有节点两两之间的语义相似度矩阵，
  输出全网共识分数和每对节点的相似度。

核心接口：
  evaluate_consensus(node_records)   → 多节点共识评估（主接口）
  evaluate_pair(rec_i, rec_j)        → 两节点对比（内部/兼容用）

公式对应：
  共识能量  E = (1/2) Σ w_ij (θ_i - θ_j)²
  博弈收益  U = sim_avg * R - C
  分布式学习 θ_i^{t+1} = θ_i^t + η∇U_i - γΣ_j w_ij(θ_i - θ_j)

相似度方法：
  char_jaccard   ⚡⚡⚡  ☆       baseline
  word_tfidf     ⚡⚡   ★★★     大规模离线
  bm25           ⚡⚡   ★★★★    推荐
  sentence_bert  ⚡    ★★★★★  阿里云百炼 Embedding
  llm_judge      🐢   ★★★★★  最高精度
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 可选依赖 ──────────────────────────────────────────
try:
    import jieba
    jieba.setLogLevel(60)
    _JIEBA_OK = True
except ImportError:
    _JIEBA_OK = False
    warnings.warn("jieba 未安装，word_tfidf/bm25 将退回字符级分词")

try:
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except ImportError:
    _BM25_OK = False
    warnings.warn("rank_bm25 未安装，bm25 将 fallback 到 word_tfidf")

try:
    from .hybrid_semantic_engine import HybridSemanticEngine
    _HYBRID_OK = True
except Exception:
    _HYBRID_OK = False

# ── config ────────────────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as _cfg
    API_KEY     = getattr(_cfg, "API_KEY",     None)
    API_BASE    = getattr(_cfg, "API_BASE",    None) or getattr(_cfg, "API_URL", None)
    API_MODEL   = getattr(_cfg, "API_MODEL",   "text-embedding-v4")
    DEVICE      = getattr(_cfg, "DEVICE",      "cpu")
    ENABLE_BERT = getattr(_cfg, "ENABLE_BERT", True)
    LLM_API_KEY = getattr(_cfg, "LLM_API_KEY", None)
    LLM_API_URL = getattr(_cfg, "LLM_API_URL",
                          "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    LLM_MODEL   = getattr(_cfg, "LLM_MODEL",   "qwen-turbo")
except (ImportError, ModuleNotFoundError):
    API_KEY = API_BASE = LLM_API_KEY = None
    API_MODEL   = "text-embedding-v4"
    DEVICE      = "cpu"
    ENABLE_BERT = True
    LLM_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    LLM_MODEL   = "qwen-turbo"

# ─────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────

_STOPWORDS = {
    "的", "了", "在", "是", "都", "很", "和", "与", "或", "不", "为",
    "有", "也", "而", "到", "以", "就", "但", "从", "被", "把",
}


def _tokenize(text: str) -> List[str]:
    text = str(text).strip()
    if not text:
        return []
    if _JIEBA_OK:
        return [t for t in jieba.cut(text) if t.strip() and t not in _STOPWORDS]
    return [c for c in text if c.strip() and c not in _STOPWORDS]


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), 0.0, 1.0))


# ─────────────────────────────────────────────────────
# 相似度计算器（不变，只是独立的静态方法集合）
# ─────────────────────────────────────────────────────

class SimilarityCalculator:
    """文本相似度计算，5 种方法，签名统一为 (text1, text2) -> float"""

    @staticmethod
    def char_jaccard(t1: str, t2: str) -> float:
        s1, s2 = set(str(t1)), set(str(t2))
        if not s1 and not s2: return 1.0
        if not s1 or  not s2: return 0.0
        return len(s1 & s2) / len(s1 | s2)

    @staticmethod
    def word_tfidf(t1: str, t2: str, corpus: Optional[List[str]] = None) -> float:
        toks1, toks2 = _tokenize(t1), _tokenize(t2)
        if not toks1 or not toks2: return 0.0
        vocab = sorted(set(toks1) | set(toks2))
        if corpus:
            N = len(corpus)
            df = defaultdict(int)
            for doc in corpus:
                for w in set(_tokenize(doc)): df[w] += 1
            idf = {w: math.log((N+1) / (df.get(w, 0)+1)) + 1.0 for w in vocab}
        else:
            idf = {w: 1.0 for w in vocab}
        tf1, tf2 = Counter(toks1), Counter(toks2)
        l1, l2   = max(len(toks1), 1), max(len(toks2), 1)
        v1 = np.array([tf1.get(w,0)/l1 * idf[w] for w in vocab])
        v2 = np.array([tf2.get(w,0)/l2 * idf[w] for w in vocab])
        return _cosine(v1, v2)

    @staticmethod
    def bm25_similarity(t1: str, t2: str, k1: float = 1.5, b: float = 0.75) -> float:
        if not _BM25_OK:
            return SimilarityCalculator.word_tfidf(t1, t2)
        toks1, toks2 = _tokenize(t1), _tokenize(t2)
        if not toks1 or not toks2: return 0.0
        bm25  = BM25Okapi([toks2], k1=k1, b=b)
        raw   = float(bm25.get_scores(toks1)[0])
        upper = len(toks1) * (k1 + 1)
        return float(np.clip(raw / upper, 0.0, 1.0)) if upper > 0 else 0.0

    @staticmethod
    def sentence_bert(t1: str, t2: str, engine: Optional[Any] = None) -> float:
        if engine is None or not _HYBRID_OK:
            return SimilarityCalculator.bm25_similarity(t1, t2)
        try:
            s = engine.bert_similarity(str(t1), str(t2))
            return s if s > 0.0 else SimilarityCalculator.bm25_similarity(t1, t2)
        except Exception:
            return SimilarityCalculator.bm25_similarity(t1, t2)

    @staticmethod
    def llm_judge(t1: str, t2: str,
                  api_key: str = None, api_url: str = None,
                  model: str = None, timeout: int = 15) -> float:
        key = api_key or LLM_API_KEY or API_KEY
        url = api_url or LLM_API_URL
        mdl = model   or LLM_MODEL
        if not key:
            return SimilarityCalculator.bm25_similarity(t1, t2)

        prompt = (
            "你是一位专业的语义相似度评审专家。\n"
            "请评估以下两段文本在**语义含义**上的相似程度，\n"
            "输出一个 0.00（完全不同）到 1.00（完全相同）之间的浮点数，\n"
            "**仅输出数字，不要任何解释**。\n\n"
            f"文本1：{str(t1)[:300]}\n"
            f"文本2：{str(t2)[:300]}\n\n相似度分数："
        )
        try:
            from openai import OpenAI
            base = url.split("/chat/completions")[0].split("/v1")[0] + "/v1"
            client = OpenAI(api_key=key, base_url=base)
            resp = client.chat.completions.create(
                model=mdl, messages=[{"role": "user", "content": prompt}],
                max_tokens=10, temperature=0.0)
            return float(np.clip(float(resp.choices[0].message.content.strip()), 0.0, 1.0))
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"LLM-Judge openai SDK 失败 ({e})，尝试 requests")
        try:
            import requests as _req
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {"model": mdl, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 10, "temperature": 0.0}
            resp = _req.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return float(np.clip(float(resp.json()["choices"][0]["message"]["content"].strip()), 0.0, 1.0))
        except Exception as e:
            warnings.warn(f"LLM-Judge 全部失败 ({e})，fallback BM25")
            return SimilarityCalculator.bm25_similarity(t1, t2)


# ─────────────────────────────────────────────────────
# 核心共识引擎
# ─────────────────────────────────────────────────────

class ConsensusEngine:
    """
    多节点语义共识引擎

    主接口：evaluate_consensus(node_records)
      输入：N 个节点的 AEIC 记录列表
      输出：
        similarity_matrix  所有节点对的相似度 {(i,j): sim}
        avg_similarity     全网平均共识相似度
        utility            博弈收益 U = avg_sim * R - C
        decision           ESS_Consensus / Audit_Required / Reject
        theta              当前分布式参数

    AEIC 四层加权：
      sim_total = w_A·sim(A,A') + w_E·sim(E,E')
               + w_I·sim(I,I') + w_C·sim(C,C')
    """

    METHODS = ("char_jaccard", "word_tfidf", "bm25", "sentence_bert", "llm_judge")

    def __init__(
        self,
        reward: float = 100.0,
        cost: float = 25.0,
        similarity_method: str = "bm25",
        use_sentence_bert: bool = False,
        use_llm_judge: bool = False,
        llm_api_key: str = None,
        llm_api_url: str = None,
        llm_model: str = None,
        corpus: Optional[List[str]] = None,
        # 向后兼容参数
        enable_bert: bool = True,
        device: str = "cpu",
        use_api: bool = False,
        api_key: str = None,
        api_url: str = None,
        api_base: str = None,
        api_model: str = None,
    ):
        if similarity_method not in self.METHODS:
            warnings.warn(f"未知方法 '{similarity_method}'，使用 'bm25'")
            similarity_method = "bm25"

        self.similarity_method = similarity_method
        self.use_sentence_bert = use_sentence_bert
        self.use_llm_judge     = use_llm_judge
        self.llm_api_key       = llm_api_key or api_key or LLM_API_KEY or API_KEY
        self.llm_api_url       = llm_api_url or LLM_API_URL
        self.llm_model         = llm_model   or LLM_MODEL
        self.corpus            = corpus or []
        self.R                 = reward
        self.C                 = cost

        # AEIC 四层加权 - 可学习参数
        # 初始化为 logits，使用 softmax 转换为概率
        self.w_logits = {"A": np.log(0.2), "E": np.log(0.3), "I": np.log(0.2), "C": np.log(0.3)}
        self.w = self._compute_weights_from_logits()  # 实际权重
        self.w_history = []  # 记录权重演化历史
        self.utility_history = []  # 记录效用历史
        
        # 分布式参数 θ（每个节点独立维护，此处为共识引擎全局值）
        self.theta = 0.5

        # Embedding 引擎
        self._bert_engine = None
        if (use_sentence_bert or similarity_method == "sentence_bert") and _HYBRID_OK:
            resolved_base  = api_base or api_url or API_BASE
            resolved_key   = api_key or API_KEY
            resolved_model = api_model or API_MODEL
            try:
                self._bert_engine = HybridSemanticEngine(
                    enable_bert=enable_bert, device=device,
                    use_api=bool(resolved_key),
                    api_key=resolved_key, api_base=resolved_base, api_model=resolved_model,
                )
            except Exception as e:
                warnings.warn(f"HybridSemanticEngine 初始化失败: {e}")

        self._calc = SimilarityCalculator()

    # ── 权重学习方法 ─────────────────────────────────────────

    def _compute_weights_from_logits(self) -> Dict[str, float]:
        """从logits计算weights（使用softmax归一化）"""
        logits = np.array([
            self.w_logits["A"],
            self.w_logits["E"],
            self.w_logits["I"],
            self.w_logits["C"]
        ])
        # 数值稳定的softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        weights = exp_logits / np.sum(exp_logits)
        return {
            "A": float(weights[0]),
            "E": float(weights[1]),
            "I": float(weights[2]),
            "C": float(weights[3])
        }

    def update_weights(self, utility: float, learning_rate: float = 0.01) -> Dict[str, float]:
        """
        根据游戏效用更新权重（自适应学习）
        
        策略：
          1. 计算权重梯度：不同层的贡献与该层权重成正相关
          2. 基于utility的方向进行梯度上升
          3. 记录历史用于收敛分析
        
        Args:
            utility: 当前游戏收益
            learning_rate: 学习率（建议 0.01-0.1）
        
        Returns:
            更新后的权重字典
        """
        # 记录历史
        self.w_history.append(dict(self.w))
        self.utility_history.append(utility)
        
        # 梯度计算：使用utility作为信号强度
        # 每个权重的梯度 ∝ utility * (该层的相对重要性)
        # 这里使用一个简单的启发式：权重越大，学习信号越强
        
        normalized_utility = max(utility / self.R, 0.01)  # 避免零和负值
        
        for key in self.w_logits.keys():
            # 梯度：utility * 当前权重值
            # 目的是强化高效用时的权重配置
            gradient = normalized_utility * self.w[key]
            
            # 使用指数形式的更新来保持数值稳定性
            self.w_logits[key] += learning_rate * gradient
        
        # 重新计算normalized权重
        self.w = self._compute_weights_from_logits()
        
        return dict(self.w)

    def get_weights_evolution(self) -> Dict[str, list]:
        """返回权重演化历史"""
        if not self.w_history:
            return {}
        
        evolution = {"A": [], "E": [], "I": [], "C": []}
        for w_dict in self.w_history:
            for key in evolution.keys():
                evolution[key].append(w_dict[key])
        return evolution

    # ── 文本相似度路由 ───────────────────────────────────

    def _sim(self, t1: str, t2: str) -> float:
        if self.use_llm_judge:
            return self._calc.llm_judge(t1, t2, api_key=self.llm_api_key,
                                        api_url=self.llm_api_url, model=self.llm_model)
        if self.use_sentence_bert:
            return self._calc.sentence_bert(t1, t2, self._bert_engine)
        dispatch = {
            "char_jaccard":  lambda: self._calc.char_jaccard(t1, t2),
            "word_tfidf":    lambda: self._calc.word_tfidf(t1, t2, self.corpus or None),
            "bm25":          lambda: self._calc.bm25_similarity(t1, t2),
            "sentence_bert": lambda: self._calc.sentence_bert(t1, t2, self._bert_engine),
            "llm_judge":     lambda: self._calc.llm_judge(t1, t2, api_key=self.llm_api_key,
                                                          api_url=self.llm_api_url, model=self.llm_model),
        }
        return dispatch.get(self.similarity_method,
                            lambda: self._calc.bm25_similarity(t1, t2))()

    def _sim_evidence(self, ev_i: Any, ev_j: Any) -> float:
        if isinstance(ev_i, list):
            text_i, set_i = " ".join(str(e) for e in ev_i), set(str(e) for e in ev_i)
        else:
            text_i, set_i = str(ev_i), set(_tokenize(str(ev_i)))
        if isinstance(ev_j, list):
            text_j, set_j = " ".join(str(e) for e in ev_j), set(str(e) for e in ev_j)
        else:
            text_j, set_j = str(ev_j), set(_tokenize(str(ev_j)))
        sem = self._sim(text_i, text_j)
        ovl = len(set_i & set_j) / len(set_i | set_j) if (set_i | set_j) else 0.0
        return 0.6 * sem + 0.4 * ovl

    # ── 单对 AEIC 相似度 ─────────────────────────────────

    def _aeic_similarity(self, rec_i: Dict, rec_j: Dict) -> Dict[str, float]:
        """计算两个节点 AEIC 记录的四层相似度，返回各层 + total"""
        sim_a = self._sim(str(rec_i.get("assumptions", "")),
                          str(rec_j.get("assumptions", "")))
        sim_e = self._sim_evidence(rec_i.get("evidence", ""),
                                   rec_j.get("evidence", ""))
        sim_i = self._sim(str(rec_i.get("inference", "")),
                          str(rec_j.get("inference", "")))
        sim_c = self._sim(str(rec_i.get("conclusion", "")),
                          str(rec_j.get("conclusion", "")))
        total = (self.w["A"] * sim_a + self.w["E"] * sim_e +
                 self.w["I"] * sim_i + self.w["C"] * sim_c)
        return {
            "sim_a": round(sim_a, 4), "sim_e": round(sim_e, 4),
            "sim_i": round(sim_i, 4), "sim_c": round(sim_c, 4),
            "total": round(total, 4),
        }

    # ── 主接口：多节点共识评估 ───────────────────────────

    def evaluate_consensus(self, node_records: List[Dict]) -> Dict[str, Any]:
        """
        多节点全量共识评估

        Args:
            node_records: N 个节点的 AEIC 记录列表，每项为含
                          assumptions/evidence/inference/conclusion 的字典。
                          可带 node_id 字段，否则自动命名 node_0..N-1。

        Returns:
            {
              "n_nodes":            N,
              "pairwise":           { "node_i×node_j": {sim_a, sim_e, sim_i, sim_c, total} },
              "avg_similarity":     全网平均共识相似度,
              "similarity_matrix":  NxN 矩阵（list of list）,
              "node_ids":           节点 ID 列表,
              "utility":            U = avg_sim * R - C,
              "decision":           ESS_Consensus / Audit_Required / Reject,
              "theta":              当前分布式参数,
            }
        """
        n = len(node_records)
        if n < 2:
            raise ValueError(f"evaluate_consensus 需要至少 2 个节点，实际传入 {n} 个")

        # 提取 node_id
        node_ids = []
        for idx, rec in enumerate(node_records):
            node_ids.append(rec.get("node_id", f"node_{idx}"))

        # 两两计算
        pairwise: Dict[str, Dict] = {}
        sim_matrix = [[0.0] * n for _ in range(n)]
        all_sims = []

        for i, j in combinations(range(n), 2):
            key = f"{node_ids[i]}×{node_ids[j]}"
            result = self._aeic_similarity(node_records[i], node_records[j])
            pairwise[key] = result
            sim_matrix[i][j] = result["total"]
            sim_matrix[j][i] = result["total"]
            all_sims.append(result["total"])
            sim_matrix[i][i] = 1.0  # 自身相似度

        for i in range(n):
            sim_matrix[i][i] = 1.0

        avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0
        utility = avg_sim * self.R - self.C
        decision = self._decide(utility)

        return {
            "n_nodes":          n,
            "node_ids":         node_ids,
            "pairwise":         pairwise,
            "avg_similarity":   round(avg_sim, 4),
            "similarity_matrix": sim_matrix,
            "utility":          round(utility, 2),
            "decision":         decision,
            "theta":            self.theta,
        }

    # ── 两节点对比（内部使用 / 向后兼容） ──────────────

    def evaluate_pair(self, rec_i: Dict, rec_j: Dict) -> Dict[str, Any]:
        """两节点 AEIC 相似度（evaluate_consensus 的双节点特化版）"""
        result = self._aeic_similarity(rec_i, rec_j)
        utility  = result["total"] * self.R - self.C
        decision = self._decide(utility)
        return {
            "sim_a":       result["sim_a"],
            "sim_e":       result["sim_e"],
            "sim_i":       result["sim_i"],
            "sim_c":       result["sim_c"],
            "total_score": result["total"],
            "utility":     round(utility, 2),
            "decision":    decision,
        }

    # 向后兼容别名
    def evaluate_game(self, row_a: Dict, row_b: Dict) -> Dict[str, Any]:
        return self.evaluate_pair(row_a, row_b)

    # ── 决策判断 ─────────────────────────────────────────

    @staticmethod
    def _decide(utility: float) -> str:
        if utility > 55:   return "ESS_Consensus"
        if utility > 0:    return "Audit_Required"
        return "Reject"

    # ── 分布式学习更新（公式3） ──────────────────────────

    def update_theta(self, utility: float, learning_rate: float = 0.01) -> float:
        """θ_{t+1} = θ_t + η * (U/R - θ_t)"""
        target = utility / self.R
        self.theta += learning_rate * (target - self.theta)
        self.theta  = float(np.clip(self.theta, 0.0, 1.0))
        return self.theta

    # 向后兼容别名
    def update_with_learning(self, utility: float, learning_rate: float = 0.01) -> float:
        return self.update_theta(utility, learning_rate)

    def make_decision(self, utility: float) -> str:
        return self._decide(utility)


# ─────────────────────────────────────────────────────
# 批量模拟（供实验框架调用）
# ─────────────────────────────────────────────────────

def run_consensus_simulation(
    node_records_list: Optional[List[List[Dict]]] = None,
    compare_methods: bool = False,
    similarity_method: str = "bm25",
    ground_truths: Optional[List[float]] = None,
) -> Optional[pd.DataFrame]:
    """
    批量共识模拟

    Args:
        node_records_list: 每轮的节点记录列表，格式
                           [ [node_0_dict, node_1_dict, ...], [...], ... ]
                           None 时自动从 DeepSeek 生成的缓存加载
        compare_methods:   是否对比所有方法
        similarity_method: 使用的相似度方法
        ground_truths:     GT 相似度（可选，有则计算 MAE）
    """
    if node_records_list is None:
        from mas.data import load_or_generate
        ds = load_or_generate()
        node_records_list = ds.get_node_records()
        if ground_truths is None:
            ground_truths = ds.ground_truths()

    if not node_records_list:
        print("❌ 无可用数据")
        return None

    if compare_methods:
        methods = ["char_jaccard", "word_tfidf", "bm25"]
        if API_KEY:
            methods.append("sentence_bert")
        frames = [
            _simulate(
                ConsensusEngine(
                    similarity_method=m,
                    use_sentence_bert=(m == "sentence_bert"),
                    api_key=API_KEY, api_base=API_BASE, api_model=API_MODEL,
                ),
                node_records_list, m, ground_truths
            )
            for m in methods
        ]
        result = pd.concat(frames, ignore_index=True)
        _print_comparison(result)
        return result
    else:
        engine = ConsensusEngine(
            similarity_method=similarity_method,
            use_sentence_bert=(similarity_method == "sentence_bert"),
            api_key=API_KEY, api_base=API_BASE, api_model=API_MODEL,
        )
        return _simulate(engine, node_records_list, similarity_method, ground_truths)


def _simulate(
    engine: ConsensusEngine,
    node_records_list: List[List[Dict]],
    method_name: str,
    ground_truths: Optional[List[float]] = None,
) -> pd.DataFrame:
    records, errors = [], []

    for i, node_recs in enumerate(node_records_list):
        try:
            res   = engine.evaluate_consensus(node_recs)
            theta = engine.update_theta(res["utility"])
            # 权重学习：基于utility自适应更新权重
            weights = engine.update_weights(res["utility"], learning_rate=0.01)
            row   = {
                "round":          i + 1,
                "method":         method_name,
                "n_nodes":        res["n_nodes"],
                "avg_similarity": res["avg_similarity"],
                "utility":        res["utility"],
                "decision":       res["decision"],
                "theta":          theta,
                "w_a":            round(weights["A"], 4),
                "w_e":            round(weights["E"], 4),
                "w_i":            round(weights["I"], 4),
                "w_c":            round(weights["C"], 4),
            }
            if ground_truths and i < len(ground_truths):
                row["gt_similarity"] = ground_truths[i]
                row["abs_error"]     = abs(res["avg_similarity"] - ground_truths[i])
            records.append(row)
        except Exception as e:
            errors.append(f"Round {i+1}: {e}")

    if errors:
        print(f"⚠️  {len(errors)} 轮出错（前3）: " + " | ".join(errors[:3]))

    df = pd.DataFrame(records)
    if not df.empty:
        ess     = (df["decision"] == "ESS_Consensus").sum()
        avg_u   = df["utility"].mean()
        mae_str = f", MAE={df['abs_error'].mean():.4f}" if "abs_error" in df.columns else ""
        print(f"[{method_name:15s}] ESS={ess}/{len(df)}, AvgU={avg_u:.2f}{mae_str}")
    return df


def _print_comparison(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 相似度方法对比摘要")
    print("=" * 60)
    agg: Dict = {"utility": ["mean", "std"]}
    if "abs_error" in df.columns:
        agg["abs_error"] = "mean"
    print(df.groupby("method").agg(agg).to_string())
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────
# 向后兼容
# ─────────────────────────────────────────────────────

class Operators:
    """保持对旧代码的向后兼容"""
    pass


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 共识引擎 v3 - 多节点方法对比演示 ===\n")
    results = run_consensus_simulation(compare_methods=True)
    if results is not None:
        print(f"\n共 {len(results)} 条结果")
        print(results[["round", "method", "n_nodes", "avg_similarity",
                        "utility", "decision"]].head(12).to_string(index=False))
