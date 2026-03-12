"""
混合语义引擎 v2 - 多层次语义理解

三层架构：
  BERT/Embedding 层  → 向量语义理解（优先使用阿里云百炼 API）
  知识层             → 领域同义词 + 反义词规则
  MinHash 层         → 集合论证据匹配

Embedding 后端优先级：
  1. 阿里云百炼 text-embedding-v4（openai SDK 兼容模式，推荐）
  2. 其他 OpenAI 兼容 Embedding API（requests fallback）
  3. 本地 SentenceTransformer 模型（需 sentence-transformers）
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasketch import MinHash
import jieba

jieba.setLogLevel(60)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────
# Embedding 引擎（阿里云百炼 / OpenAI 兼容）
# ─────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    统一 Embedding 接口，支持三种后端：
      - dashscope : 阿里云百炼 (openai SDK 兼容模式)  ← 首选
      - openai_api: 任意 OpenAI 兼容 HTTP API
      - local_bert: 本地 SentenceTransformer
    """

    def __init__(
        self,
        api_key: str = None,
        api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_model: str = "text-embedding-v4",
        # 本地 BERT 备用
        local_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        # 缓存（避免重复请求相同文本）
        enable_cache: bool = True,
    ):
        self.api_key   = api_key
        self.api_base  = api_base
        self.api_model = api_model
        self.device    = device
        self._cache: Dict[str, List[float]] = {} if enable_cache else None

        # ── 尝试初始化 openai SDK ────────────────────
        self._openai_client = None
        if api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=api_key, base_url=api_base)
                self.backend = "dashscope"
            except ImportError:
                warnings.warn("openai 未安装，将用 requests fallback。"
                              "建议: pip install openai")
                self.backend = "openai_api_requests"

        # ── 若无 API Key，尝试本地 BERT ──────────────
        if not api_key:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(local_model_name, device=device)
                self.backend = "local_bert"
            except ImportError:
                self._local_model = None
                self.backend = "none"
                warnings.warn("既无 API Key 也无 sentence-transformers，"
                              "EmbeddingEngine 将不可用")
        else:
            self._local_model = None

        if self.backend != "none":
            print(f"  [EmbeddingEngine] 后端: {self.backend}  模型: {api_model if api_key else local_model_name}")

    @property
    def initialized(self) -> bool:
        return self.backend != "none"

    def embed(self, text: str) -> Optional[np.ndarray]:
        """获取单条文本的 embedding 向量"""
        text = str(text).strip()
        if not text or not self.initialized:
            return None

        # ── 命中缓存 ─────────────────────────────────
        if self._cache is not None and text in self._cache:
            return np.array(self._cache[text])

        vec = None

        if self.backend in ("dashscope", "openai_api_requests"):
            vec = self._embed_via_api(text)
        elif self.backend == "local_bert":
            vec = self._embed_local(text)

        # ── 写入缓存 ─────────────────────────────────
        if vec is not None and self._cache is not None:
            self._cache[text] = vec.tolist()

        return vec

    def similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的余弦相似度 ∈ [0, 1]"""
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        if v1 is None or v2 is None:
            return 0.0
        return float(np.clip(_cosine(v1, v2), 0.0, 1.0))

    # ── 内部：API 调用 ────────────────────────────────

    def _embed_via_api(self, text: str) -> Optional[np.ndarray]:
        """调用 OpenAI 兼容 Embedding API（优先 openai SDK）"""
        # 方式1：openai SDK（百炼官方推荐方式）
        if self._openai_client is not None:
            try:
                resp = self._openai_client.embeddings.create(
                    model=self.api_model,
                    input=text,
                    encoding_format="float",
                )
                vec = resp.data[0].embedding
                return np.array(vec, dtype=np.float32)
            except Exception as e:
                warnings.warn(f"openai SDK 调用失败: {e}，尝试 requests fallback")
                # 降级到 requests
                self.backend = "openai_api_requests"

        # 方式2：requests fallback
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.api_model,
                "input": text,
                "encoding_format": "float",
            }
            url = self.api_base.rstrip("/") + "/embeddings"
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            vec = data["data"][0]["embedding"]
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            warnings.warn(f"requests embedding 调用失败: {e}")
            return None

    # ── 内部：本地 BERT ───────────────────────────────

    def _embed_local(self, text: str) -> Optional[np.ndarray]:
        try:
            emb = self._local_model.encode(text, convert_to_numpy=True)
            return emb.astype(np.float32)
        except Exception as e:
            warnings.warn(f"本地 BERT encode 失败: {e}")
            return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────
# 领域知识库（同义词 / 反义词）
# ─────────────────────────────────────────────────────

class DomainKnowledgeBase:
    """领域知识库 - 同义词和规则"""

    def __init__(self):
        self.synonym_groups = {
            "批准": ["同意", "通过", "核准", "接受", "允许", "赞同", "核准"],
            "拒绝": ["否决", "驳回", "不同意", "拒批", "否认", "不通过"],
            "资产": ["财产", "资金", "资源", "财富"],
            "证明": ["凭证", "文件", "材料", "证件", "证书", "文书"],
            "评估": ["审核", "审计", "验证", "评价", "考核"],
            "高":   ["优", "好", "强", "大", "多"],
            "低":   ["差", "弱", "小", "少", "劣"],
        }
        self.antonyms = {
            "批准": ["拒绝", "否决", "驳回"],
            "通过": ["否决", "拒绝"],
            "高":   ["低", "差", "弱"],
            "优":   ["差", "低", "劣"],
        }
        self.word_to_group: Dict[str, int] = {}
        for gid, (key, syns) in enumerate(self.synonym_groups.items()):
            for word in [key] + syns:
                self.word_to_group[word] = gid

    def similarity(self, text1: str, text2: str) -> float:
        t1, t2 = str(text1).strip(), str(text2).strip()
        if not t1 or not t2:
            return 0.0
        w1 = set(self._tok(t1))
        w2 = set(self._tok(t2))
        if not w1 or not w2:
            return 0.0
        literal = len(w1 & w2) / len(w1 | w2)
        syn_s   = self._synonym_match(w1, w2)
        ant_p   = self._antonym_penalty(w1, w2)
        return max(0.0, min(1.0, (0.3 * literal + 0.7 * syn_s) * (1 - ant_p)))

    def _tok(self, text: str) -> List[str]:
        _SW = {"的", "了", "在", "是", "都", "很", "和", "与", "或", "不"}
        try:
            return [w for w in jieba.cut(text) if w.strip() and w not in _SW]
        except Exception:
            return list(text)

    def _synonym_match(self, w1: set, w2: set) -> float:
        if not w1 or not w2:
            return 0.0
        matched = sum(
            2.0 if a == b else (1.5 if self._are_synonyms(a, b) else 0.0)
            for a in w1 for b in w2
        )
        return matched / (len(w1) + len(w2))

    def _are_synonyms(self, a: str, b: str) -> bool:
        g1 = self.word_to_group.get(a)
        g2 = self.word_to_group.get(b)
        return g1 is not None and g1 == g2

    def _antonym_penalty(self, w1: set, w2: set) -> float:
        for a in w1:
            for b in w2:
                if b in self.antonyms.get(a, []):
                    return 0.8
        return 0.0


# ─────────────────────────────────────────────────────
# MinHash 集合匹配
# ─────────────────────────────────────────────────────

class SetMatchingEngine:
    """MinHash 集合相似度（用于 evidence 层）"""

    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm

    def similarity(self, set1: Any, set2: Any) -> float:
        l1, l2 = self._to_list(set1), self._to_list(set2)
        if not l1 or not l2:
            return 0.0
        m1, m2 = MinHash(num_perm=self.num_perm), MinHash(num_perm=self.num_perm)
        for item in l1:
            m1.update(str(item).encode("utf-8"))
        for item in l2:
            m2.update(str(item).encode("utf-8"))
        return m1.jaccard(m2)

    @staticmethod
    def _to_list(obj: Any) -> List[str]:
        if isinstance(obj, list):
            return obj
        if isinstance(obj, str) and obj.startswith("["):
            try:
                return json.loads(obj)
            except Exception:
                return [obj]
        return [obj]


# ─────────────────────────────────────────────────────
# 混合语义引擎（对外主接口）
# ─────────────────────────────────────────────────────

class HybridSemanticEngine:
    """
    混合语义引擎 - 整合三层技术

    构造参数兼容原有接口，新增 api_base 字段支持百炼服务。
    """

    def __init__(
        self,
        enable_bert: bool = True,
        device: str = "cpu",
        # API 参数（百炼 / OpenAI 兼容）
        use_api: bool = False,
        api_key: str = None,
        api_url: str = None,      # 原字段，对应 api_base
        api_base: str = None,     # 新字段，与 api_url 二选一
        api_model: str = "text-embedding-v4",
    ):
        # api_base 优先，api_url 向后兼容
        resolved_base = api_base or api_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # ── 初始化 Embedding 引擎 ─────────────────────
        if enable_bert:
            self._emb = EmbeddingEngine(
                api_key=api_key if use_api else None,
                api_base=resolved_base,
                api_model=api_model,
                device=device,
            )
        else:
            self._emb = None

        self.domain_kb   = DomainKnowledgeBase()
        self.set_matcher = SetMatchingEngine()
        self.weights     = {"bert": 0.5, "domain": 0.3, "minhash": 0.2}

    # ── 公共接口 ──────────────────────────────────────

    def bert_similarity(self, text1: str, text2: str) -> float:
        """Embedding 余弦相似度（对外暴露，供 consensus.py 调用）"""
        if self._emb and self._emb.initialized:
            return self._emb.similarity(text1, text2)
        return 0.0

    def evaluate_game(self, row_a: Dict, row_b: Dict) -> Dict[str, Any]:
        """四层 AEIC 融合评估"""
        def _best(text_a: str, text_b: str) -> float:
            bert_s   = self.bert_similarity(text_a, text_b)
            domain_s = self.domain_kb.similarity(text_a, text_b)
            return max(bert_s, domain_s)

        sim_a = _best(str(row_a.get("assumptions", "")),
                      str(row_b.get("assumptions", "")))
        sim_e = self.set_matcher.similarity(
            row_a.get("evidence", []), row_b.get("evidence", [])
        )
        sim_i = _best(str(row_a.get("inference", "")),
                      str(row_b.get("inference", "")))
        sim_c = _best(str(row_a.get("conclusion", "")),
                      str(row_b.get("conclusion", "")))

        total  = 0.2 * sim_a + 0.3 * sim_e + 0.2 * sim_i + 0.3 * sim_c
        utility = total * 100 - 25

        return {
            "sim_a": round(sim_a, 4),
            "sim_e": round(sim_e, 4),
            "sim_i": round(sim_i, 4),
            "sim_c": round(sim_c, 4),
            "total_score": round(total, 4),
            "utility": round(utility, 2),
        }

    def set_weights(self, bert_weight=None, domain_weight=None, minhash_weight=None):
        if bert_weight   is not None: self.weights["bert"]    = bert_weight
        if domain_weight is not None: self.weights["domain"]  = domain_weight
        if minhash_weight is not None: self.weights["minhash"] = minhash_weight


# ─────────────────────────────────────────────────────
# 向后兼容：原 BertSemanticEngine 别名
# ─────────────────────────────────────────────────────

class BertSemanticEngine(EmbeddingEngine):
    """向后兼容别名，新代码请使用 EmbeddingEngine"""
    def __init__(self, model_name=None, device="cpu",
                 use_api=False, api_key=None, api_url=None, api_model="text-embedding-v4"):
        super().__init__(
            api_key=api_key if use_api else None,
            api_base=api_url,
            api_model=api_model,
            local_model_name=model_name or "paraphrase-multilingual-MiniLM-L12-v2",
            device=device,
        )


# ─────────────────────────────────────────────────────
# CLI 快速验证
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from config import API_KEY, API_BASE, API_MODEL
    except ImportError:
        API_KEY, API_BASE, API_MODEL = None, None, "text-embedding-v4"

    print("=== HybridSemanticEngine 验证 ===\n")

    engine = HybridSemanticEngine(
        enable_bert=True,
        use_api=bool(API_KEY),
        api_key=API_KEY,
        api_base=API_BASE,
        api_model=API_MODEL,
    )

    # 相似对
    pairs = [
        ("批准贷款申请", "同意贷款申请",         "高相似（同义词）"),
        ("证件齐全材料完整", "资料完备文件齐备",  "高相似（语义近）"),
        ("批准申请",     "拒绝申请",              "低相似（反义）"),
        ("申请人信用良好", "市场行情波动",         "低相似（无关）"),
    ]

    print(f"  {'文本A':<20} {'文本B':<20} {'相似度':>8}  说明")
    print("  " + "-" * 65)
    for t1, t2, desc in pairs:
        s = engine.bert_similarity(t1, t2)
        print(f"  {t1:<20} {t2:<20} {s:>8.4f}  {desc}")

    print("\n=== AEIC 博弈评估 ===\n")
    row_a = {"assumptions": "申请人信用良好", "evidence": ["征信报告", "流水单"],
             "inference": "还款能力充足", "conclusion": "批准贷款"}
    row_b = {"assumptions": "申请人资质优秀", "evidence": ["征信记录", "银行流水"],
             "inference": "偿债能力足够", "conclusion": "同意放款"}
    result = engine.evaluate_game(row_a, row_b)
    print(f"  sim_total={result['total_score']:.4f}  utility={result['utility']:.2f}")
