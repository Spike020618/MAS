import pandas as pd
import zlib
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasketch import MinHash
from simhash import Simhash

# 反义词惩罚列表 (用于修正逻辑偏差)
CONFLICT_PAIRS = [('approve', 'reject'), ('max', 'min'), ('high', 'low'), ('enable', 'disable'), ('shipped', 'cancelled')]

def calc_simhash_norm(t1, t2):
    features = [t1[i:i+2] for i in range(len(t1)-1)]
    return Simhash(features).distance(Simhash([t2[i:i+2] for i in range(len(t2)-1)])) / 64.0

def calc_minhash_jacc(t1, t2):
    m1, m2 = MinHash(num_perm=128), MinHash(num_perm=128)
    for i in range(len(t1)-1): m1.update(t1[i:i+2].encode('utf8'))
    for i in range(len(t2)-1): m2.update(t2[i:i+2].encode('utf8'))
    return m1.jaccard(m2)

def calc_ncd(t1, t2):
    def z(x): return len(zlib.compress(x.encode('utf8')))
    return (z(t1+t2) - min(z(t1), z(t2))) / max(z(t1), z(t2))

def calc_cosine_sim_refined(t1, t2):
    c1, c2 = Counter([t1[i:i+2] for i in range(len(t1)-1)]), Counter([t2[i:i+2] for i in range(len(t2)-1)])
    words = set(c1.keys()) | set(c2.keys())
    v1, v2 = [c1.get(w, 0) for w in words], [c2.get(w, 0) for w in words]
    mag = math.sqrt(sum(a**2 for a in v1)) * math.sqrt(sum(a**2 for a in v2))
    base_sim = sum(a * b for a, b in zip(v1, v2)) / mag if mag > 0 else 0
    
    # 简单的逻辑修正：如果检测到明显的反义冲突，大幅降低分数
    for p1, p2 in CONFLICT_PAIRS:
        if (p1 in t1.lower() and p2 in t2.lower()) or (p2 in t1.lower() and p1 in t2.lower()):
            base_sim *= 0.2  # 逻辑惩罚
    return base_sim

def run_refined_analysis(input_csv):
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        a, b = str(row['proposal_a']), str(row['proposal_b'])
        results.append({
            'label': row['label'],
            'SimHash_Inv': 1 - calc_simhash_norm(a, b),
            'MinHash_Jacc': calc_minhash_jacc(a, b),
            'NCD_Inv': 1 - calc_ncd(a, b),
            'Cosine_Sim': calc_cosine_sim_refined(a, b)
        })
    
    res_df = pd.DataFrame(results)
    print("📊 各指标与 Label 的相关系数 (优化后):")
    corr = res_df.corr()['label'].sort_values(ascending=False)
    print(corr)
    
    # 绘图分布
    plt.figure(figsize=(10, 6))
    sns.heatmap(res_df.corr(), annot=True, cmap='RdYlGn', center=0)
    plt.title("Correlation Matrix: Features vs Label")
    plt.savefig("correlation_heatmap.png")
    plt.show()

if __name__ == "__main__":
    run_refined_analysis("data.csv")