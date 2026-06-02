"""一键运行全部实验并生成图表"""
import sys
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
os.makedirs("figures", exist_ok=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sim", default="embedding", choices=["embedding", "bm25", "tfidf"],
                    help="Similarity method (default: embedding)")
args = parser.parse_args()

print("╔══════════════════════════════════════════════════════════╗")
print("║  AEIC-Stackelberg Decentralized Semantic Consensus      ║")
print("║  Experiment Suite (9 Experiments)                       ║")
print("╚══════════════════════════════════════════════════════════╝")
print(f"  Similarity method: {args.sim}")
print()

if args.sim == "embedding":
    print("  Loading sentence-transformer model (first run downloads ~420MB)...")
    from core.similarity import EmbeddingSimilarity
    EmbeddingSimilarity()
    print("  Model ready.\n")

from experiments.exp1_aeic_vs_e2e import run as run_exp1
from experiments.exp2_adaptive_weights import run as run_exp2
from experiments.exp3_stackelberg import run as run_exp3
from experiments.exp4_cognitive_diversity import run as run_exp4
from experiments.exp5_baseline_comparison import run as run_exp5
from experiments.exp6_trust_ablation import run as run_exp6
from experiments.exp7_scalability import run as run_exp7
from experiments.exp8_topology import run as run_exp8
from experiments.exp9_cross_dataset import run as run_exp9

print("[1/9] Experiment 1: AEIC Layered vs End-to-End")
run_exp1(sim_method=args.sim)

print("\n[2/9] Experiment 2: Adaptive Weight Learning")
run_exp2(sim_method=args.sim)

print("\n[3/9] Experiment 3: Stackelberg Convergence")
run_exp3()

print("\n[4/9] Experiment 4: Cognitive Diversity Ablation")
run_exp4(sim_method=args.sim)

print("\n[5/9] Experiment 5: Baseline Comparison")
run_exp5(sim_method=args.sim)

print("\n[6/9] Experiment 6: Trust Component Ablation")
run_exp6(sim_method=args.sim)

print("\n[7/9] Experiment 7: Scalability Analysis")
run_exp7()

print("\n[8/9] Experiment 8: Network Topology Comparison")
run_exp8()

print("\n[9/9] Experiment 9: Cross-Dataset Generalization")
run_exp9(sim_method=args.sim)

print("\n" + "=" * 60)
print("All experiments complete. Figures saved to figures/")
for f in sorted(os.listdir("figures")):
    if f.endswith((".png", ".pdf")):
        print(f"  • figures/{f}")
print("=" * 60)
