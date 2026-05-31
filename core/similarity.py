"""
相似度计算模块 — 工业级多方法支持

Methods:
  - embedding: Sentence-Transformer cosine similarity (industry standard)
  - bm25: BM25 lexical retrieval score (Elasticsearch-style)
  - tfidf: TF-IDF cosine similarity (lightweight baseline)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

try:
    import jieba
    jieba.setLogLevel(60)
    _JIEBA = True
except ImportError:
    _JIEBA = False

try:
    from rank_bm25 import BM25Okapi
    _BM25 = True
except ImportError:
    _BM25 = False

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from sentence_transformers import SentenceTransformer
    _ST = True
except ImportError:
    _ST = False

_STOP_ZH = {"的", "了", "在", "是", "都", "很", "和", "与", "或", "不", "为",
             "有", "也", "而", "到", "以", "就", "但", "从", "被", "把", "这",
             "那", "一", "个", "上", "下", "中", "对", "之"}
_STOP_EN = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "of", "in", "to", "for", "with", "on", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "it", "its", "this", "that"}


def tokenize(text: str) -> List[str]:
    """Mixed Chinese/English tokenization with stopword removal."""
    text = str(text).strip()
    if not text:
        return []
    if _JIEBA:
        tokens = list(jieba.cut(text))
    else:
        tokens = text.split()
    return [t.strip().lower() for t in tokens
            if t.strip() and t.strip().lower() not in _STOP_ZH
            and t.strip().lower() not in _STOP_EN and len(t.strip()) > 0]


class EmbeddingSimilarity:
    """Sentence-Transformer based semantic similarity (industry standard)."""

    _instance: Optional["EmbeddingSimilarity"] = None
    _model: Optional[object] = None

    def __new__(cls, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        if self._model is None and _ST:
            self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("sentence-transformers not available")
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def similarity(self, t1: str, t2: str) -> float:
        embs = self.encode([t1, t2])
        return float(np.dot(embs[0], embs[1]))

    def similarity_batch(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        all_texts = []
        for a, b in pairs:
            all_texts.extend([a, b])
        embs = self.encode(all_texts)
        scores = np.array([
            float(np.dot(embs[i * 2], embs[i * 2 + 1]))
            for i in range(len(pairs))
        ])
        return scores


class SimilarityCalculator:
    """Unified similarity interface supporting multiple methods."""

    def __init__(self, method: str = "embedding"):
        self.method = method
        self._embedding: Optional[EmbeddingSimilarity] = None
        if method == "embedding" and _ST:
            self._embedding = EmbeddingSimilarity()

    def tfidf(self, t1: str, t2: str) -> float:
        toks1, toks2 = tokenize(t1), tokenize(t2)
        if not toks1 or not toks2:
            return 0.0
        vocab = sorted(set(toks1) | set(toks2))
        tf1, tf2 = Counter(toks1), Counter(toks2)
        l1, l2 = max(len(toks1), 1), max(len(toks2), 1)
        v1 = np.array([tf1.get(w, 0) / l1 for w in vocab])
        v2 = np.array([tf2.get(w, 0) / l2 for w in vocab])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.clip(np.dot(v1, v2) / (n1 * n2), 0.0, 1.0))

    def bm25(self, t1: str, t2: str) -> float:
        if not _BM25:
            return self.tfidf(t1, t2)
        toks1, toks2 = tokenize(t1), tokenize(t2)
        if not toks1 or not toks2:
            return 0.0
        corpus = [toks1, toks2]
        bm = BM25Okapi(corpus)
        s1 = float(bm.get_scores(toks1)[1])
        s2 = float(bm.get_scores(toks2)[0])
        upper1 = max(len(toks1) * 1.5, 1.0)
        upper2 = max(len(toks2) * 1.5, 1.0)
        bm_score = (np.clip(s1 / upper1, 0, 1) + np.clip(s2 / upper2, 0, 1)) / 2.0
        set1, set2 = set(toks1), set(toks2)
        overlap = len(set1 & set2) / max(len(set1 | set2), 1)
        return float(np.clip(0.6 * bm_score + 0.4 * overlap, 0.0, 1.0))

    def embedding(self, t1: str, t2: str) -> float:
        if self._embedding is None:
            if _ST:
                self._embedding = EmbeddingSimilarity()
            else:
                return self.tfidf(t1, t2)
        return max(0.0, self._embedding.similarity(t1, t2))

    def compute(self, t1: str, t2: str, method: str = None) -> float:
        m = method or self.method
        dispatch = {"embedding": self.embedding, "bm25": self.bm25, "tfidf": self.tfidf}
        return dispatch.get(m, self.embedding)(t1, t2)
