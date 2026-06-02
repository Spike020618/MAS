#!/bin/bash
# AEIC-Stackelberg P2P全量实验脚本
# Usage: bash run_experiments.sh
#
# 真正的P2P架构: 每个Agent独立HTTP服务，通过网络通信
# Registry(8000) + Agent-0(8001) + Agent-1(8002) + ... + Agent-4(8005)
# 覆盖: 8个数据集 × 4种模式 × 4种方法

set -e
cd "$(dirname "$0")"

N_SAMPLES=20
N_AGENTS=5
SIM=embedding
MODES="parallel,sequential,star,debate"

echo "======================================"
echo " AEIC-Stackelberg P2P Experiments"
echo "======================================"
echo "  Architecture: P2P (each agent = independent HTTP service)"
echo "  Registry:     port 8000"
echo "  Agents:       ports 8001-800$N_AGENTS (DeepSeek + ZhiPu)"
echo "  Samples/dataset: $N_SAMPLES"
echo "  Agents:       $N_AGENTS"
echo "  Sim:          $SIM (local Sentence-BERT)"
echo "  Modes:        $MODES"
echo ""

mkdir -p results

DATASETS="StrategyQA TruthfulQA HotpotQA GPQA MMLU GSM8K ARC-Challenge CommonsenseQA"

for DS in $DATASETS; do
    echo ""
    echo "========================================"
    echo " Dataset: $DS (P2P mode)"
    echo "========================================"
    python3 run_p2p.py \
        --n_agents $N_AGENTS \
        --sim $SIM \
        --experiment \
        --dataset "$DS" \
        --n_samples $N_SAMPLES \
        --modes $MODES \
        --output "results/exp_p2p_${DS}_${N_SAMPLES}q.json"
done

echo ""
echo "======================================"
echo " All P2P experiments complete!"
echo " Results in: results/"
echo "======================================"
ls -la results/exp_p2p_*.json
