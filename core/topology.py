"""
网络拓扑生成器 — 多种去中心化通信拓扑

Topologies:
  - complete: 完全图，所有节点互连 (λ₂ = N/(N-1))
  - ring: 环形，每个节点仅与左右邻居通信 (λ₂ = 2(1-cos(2π/N)))
  - star: 星形，一个中心节点连接所有其他节点
  - random: Erdős–Rényi 随机图 G(N,p)
  - small_world: Watts-Strogatz 小世界网络 (高聚类 + 短路径)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def complete_graph(n: int) -> np.ndarray:
    """完全图：w_ij = 1/(N-1)，λ₂ = N/(N-1)"""
    w = np.full((n, n), 1.0 / max(n - 1, 1))
    np.fill_diagonal(w, 0.0)
    return w


def ring_graph(n: int) -> np.ndarray:
    """环形图：每个节点连接左右邻居"""
    w = np.zeros((n, n))
    for i in range(n):
        left = (i - 1) % n
        right = (i + 1) % n
        w[i, left] = 0.5
        w[i, right] = 0.5
    return w


def star_graph(n: int) -> np.ndarray:
    """星形图：节点0为中心，连接所有其他节点"""
    w = np.zeros((n, n))
    for i in range(1, n):
        w[0, i] = 1.0 / max(n - 1, 1)
        w[i, 0] = 1.0
    return w


def random_graph(n: int, p: float = 0.5, seed: int = 42) -> np.ndarray:
    """Erdős–Rényi 随机图 G(N,p)"""
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w[i, j] = 1.0
                w[j, i] = 1.0
    # 确保连通性：如果某节点孤立，连接到随机邻居
    for i in range(n):
        if w[i].sum() == 0:
            j = rng.integers(0, n)
            while j == i:
                j = rng.integers(0, n)
            w[i, j] = 1.0
            w[j, i] = 1.0
    # 行归一化
    for i in range(n):
        row_sum = w[i].sum()
        if row_sum > 0:
            w[i] /= row_sum
    return w


def small_world_graph(n: int, k: int = 4, p: float = 0.3,
                      seed: int = 42) -> np.ndarray:
    """Watts-Strogatz 小世界网络：k 近邻环 + 以概率 p 重连"""
    rng = np.random.default_rng(seed)
    k = min(k, n - 1)
    if k % 2 == 1:
        k -= 1
    k = max(k, 2)
    half_k = k // 2

    w = np.zeros((n, n))
    # 先构建 k-近邻环
    for i in range(n):
        for j in range(1, half_k + 1):
            right = (i + j) % n
            w[i, right] = 1.0
            w[right, i] = 1.0

    # 以概率 p 重连
    for i in range(n):
        for j in range(1, half_k + 1):
            if rng.random() < p:
                right = (i + j) % n
                # 断开原连接，连接到随机节点
                new_target = rng.integers(0, n)
                attempts = 0
                while (new_target == i or w[i, new_target] > 0) and attempts < 20:
                    new_target = rng.integers(0, n)
                    attempts += 1
                if new_target != i and w[i, new_target] == 0:
                    w[i, right] = 0.0
                    w[right, i] = 0.0
                    w[i, new_target] = 1.0
                    w[new_target, i] = 1.0

    # 确保连通性
    for i in range(n):
        if w[i].sum() == 0:
            j = (i + 1) % n
            w[i, j] = 1.0
            w[j, i] = 1.0

    # 行归一化
    for i in range(n):
        row_sum = w[i].sum()
        if row_sum > 0:
            w[i] /= row_sum
    return w


def path_graph(n: int) -> np.ndarray:
    """链式图（有向）：节点i连接到节点i+1，用于sequential模式"""
    w = np.zeros((n, n))
    for i in range(n - 1):
        w[i, i + 1] = 1.0
        w[i + 1, i] = 1.0
    # 行归一化
    for i in range(n):
        row_sum = w[i].sum()
        if row_sum > 0:
            w[i] /= row_sum
    return w


def make_topology(name: str, n: int, **kwargs) -> np.ndarray:
    """统一接口：根据名称生成拓扑权重矩阵"""
    builders = {
        "complete": complete_graph,
        "ring": ring_graph,
        "star": star_graph,
        "random": random_graph,
        "small_world": small_world_graph,
        "path": path_graph,
    }
    if name not in builders:
        raise ValueError(f"Unknown topology: {name}. Available: {list(builders.keys())}")
    return builders[name](n, **kwargs)


def laplacian_lambda2(w: np.ndarray) -> float:
    """计算拉普拉斯矩阵的代数连通度 λ₂（Fiedler 值）

    λ₂ 越大 → 收敛越快。完全图 λ₂ 最大，环形最小。
    """
    n = w.shape[0]
    D = np.diag(w.sum(axis=1))
    L = D - w
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    if n < 2:
        return 0.0
    return float(eigenvalues[1])


def topology_summary(w: np.ndarray) -> Dict[str, float]:
    """拓扑统计信息"""
    n = w.shape[0]
    edges = np.sum(w > 0) // 2
    density = edges / max(n * (n - 1) / 2, 1)
    degrees = (w > 0).sum(axis=1).astype(float)
    lam2 = laplacian_lambda2(w)
    return {
        "nodes": n,
        "edges": int(edges),
        "density": round(float(density), 4),
        "avg_degree": round(float(degrees.mean()), 2),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "lambda2": round(lam2, 6),
    }


TOPOLOGY_NAMES = ["complete", "ring", "star", "random", "small_world", "path"]
TOPOLOGY_LABELS = {
    "complete": "Complete",
    "ring": "Ring",
    "star": "Star",
    "random": "Random (ER)",
    "small_world": "Small-World",
    "path": "Path (Chain)",
}
