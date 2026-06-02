#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Set 2: 自适应拓扑实验 (系统验证)
# ═══════════════════════════════════════════════════════════════
# 对比: Adaptive / Random-W / MajVote-Adaptive
# Planner: DeepSeek 自由生成拓扑W + 角色分配
# 执行: 按W拓扑排序(DAG)或多轮迭代
# 共识: Option B (depth-weighted + IC/EC + Stackelberg)
# Agent规模: 3 / 5 / 7
# 数据集: 8个 (或指定单个)
#
# Usage:
#   bash run_exp_adaptive.sh                    # 跑全部8个, 20题/数据集
#   bash run_exp_adaptive.sh 10                 # 跑全部8个, 10题/数据集
#   bash run_exp_adaptive.sh 20 StrategyQA      # 只跑StrategyQA
#   bash run_exp_adaptive.sh 10 GPQA MMLU       # 跑指定的多个
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

N_SAMPLES=${1:-20}
shift 2>/dev/null || true

MAX_AGENTS=7
SIM=embedding
CSV_PATH="results/set2_results.csv"

# 如果有额外参数，作为数据集列表；否则跑全部
if [ $# -gt 0 ]; then
    DATASETS="$@"
else
    DATASETS="StrategyQA TruthfulQA HotpotQA GPQA MMLU GSM8K ARC-Challenge CommonsenseQA"
fi

echo "======================================"
echo " Set 2: Adaptive Topology Experiments"
echo "======================================"
echo "  Planner: DeepSeek (free topology generation)"
echo "  Executor: DAG/Iterative (topology-ordered)"
echo "  Consensus: Option B (depth-weighted)"
echo "  Samples/dataset: $N_SAMPLES"
echo "  Max agents: $MAX_AGENTS (Planner自行决定2~$MAX_AGENTS)"
echo "  Sim: $SIM"
echo "  Datasets: $DATASETS"
echo "  CSV: $CSV_PATH (checkpoint/resume)"
echo ""

mkdir -p results

for DS in $DATASETS; do
    echo ""
    echo "========================================"
    echo " Dataset: $DS"
    echo "========================================"
    python3 experiments/exp_adaptive.py \
        --dataset "$DS" \
        --n_samples $N_SAMPLES \
        --n_agents $MAX_AGENTS \
        --sim $SIM \
        --csv "$CSV_PATH"
done

echo ""
echo "======================================"
echo " Set 2 Complete!"
echo " Results: $CSV_PATH"
echo "======================================"
wc -l "$CSV_PATH" 2>/dev/null || echo "  (no results yet)"
