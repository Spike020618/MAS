#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Set 1: 固定拓扑实验 (机制验证)
# ═══════════════════════════════════════════════════════════════
# 对比: MajVote / QualWt / AEIC / AEIC+SG
# 模式: parallel / sequential / star / debate
# Agent规模: 3 / 5 / 7
# 数据集: 8个 (或指定单个)
#
# Usage:
#   bash run_exp_fixed.sh                    # 跑全部8个数据集, 20题/数据集
#   bash run_exp_fixed.sh 10                 # 跑全部8个, 10题/数据集
#   bash run_exp_fixed.sh 20 StrategyQA      # 只跑StrategyQA, 20题
#   bash run_exp_fixed.sh 20 StrategyQA TruthfulQA  # 跑指定的多个
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

N_SAMPLES=${1:-20}
shift 2>/dev/null || true

AGENT_SCALES="3 5 7"
SIM=embedding
MODES="parallel,sequential,star,debate"
CSV_PATH="results/set1_results.csv"

# 如果有额外参数，作为数据集列表；否则跑全部
if [ $# -gt 0 ]; then
    DATASETS="$@"
else
    DATASETS="StrategyQA TruthfulQA HotpotQA GPQA MMLU GSM8K ARC-Challenge CommonsenseQA"
fi

echo "======================================"
echo " Set 1: Fixed Topology Experiments"
echo "======================================"
echo "  Samples/dataset: $N_SAMPLES"
echo "  Agent scales: $AGENT_SCALES"
echo "  Sim: $SIM (local Sentence-BERT)"
echo "  Modes: $MODES"
echo "  Datasets: $DATASETS"
echo "  CSV: $CSV_PATH (checkpoint/resume)"
echo ""

mkdir -p results

for N in $AGENT_SCALES; do
    for DS in $DATASETS; do
        echo ""
        echo "========================================"
        echo " n_agents=$N | Dataset: $DS"
        echo "========================================"
        python3 experiments/exp_orchestrator.py \
            --dataset "$DS" \
            --n_samples $N_SAMPLES \
            --n_agents $N \
            --modes $MODES \
            --sim $SIM \
            --csv "$CSV_PATH"
    done
done

echo ""
echo "======================================"
echo " Set 1 Complete!"
echo " Results: $CSV_PATH"
echo "======================================"
wc -l "$CSV_PATH" 2>/dev/null || echo "  (no results yet)"
