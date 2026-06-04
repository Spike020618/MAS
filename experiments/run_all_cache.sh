#!/bin/bash
# 一次性跑完所有剩余数据集的AEIC缓存收集 + 信号预计算
# Phase 1 用 HF_ENDPOINT（下载数据集），Phase 2 不用（避免sentence-transformer DNS问题）

set -e
cd "$(dirname "$0")/.."

DATASETS=("MMLU-Hard" "GSM8K" "MMLU" "TruthfulQA" "CommonsenseQA" "ARC-Challenge" "HotpotQA" "GPQA")
N_SAMPLES=200
N_AGENTS=5
OUT_DIR=results

for ds in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "  Dataset: $ds"
    echo "=========================================="

    cache_file="$OUT_DIR/aeic_cache_${ds}_${N_AGENTS}a.json"
    signals_file="$OUT_DIR/signals_${ds}_${N_AGENTS}a.json"

    # Phase 1: 收集AEIC（跳过已完成的）
    if [ -f "$cache_file" ]; then
        count=$(python3 -c "import json; d=json.load(open('$cache_file')); print(len(d['questions']))")
        if [ "$count" -ge "$N_SAMPLES" ]; then
            echo "  Phase 1 already done ($count questions), skipping."
        else
            echo "  Phase 1 resuming from $count questions..."
            HF_ENDPOINT=https://hf-mirror.com python3 -u experiments/collect_cache.py \
                --dataset "$ds" --n_samples $N_SAMPLES --n_agents $N_AGENTS --out_dir $OUT_DIR
        fi
    else
        echo "  Phase 1 starting..."
        HF_ENDPOINT=https://hf-mirror.com python3 -u experiments/collect_cache.py \
            --dataset "$ds" --n_samples $N_SAMPLES --n_agents $N_AGENTS --out_dir $OUT_DIR
    fi

    # Phase 2: 预计算信号（不设HF_ENDPOINT）
    if [ -f "$signals_file" ]; then
        echo "  Phase 2 already done, skipping."
    else
        echo "  Phase 2: precomputing signals..."
        python3 -u experiments/collect_cache.py --precompute "$cache_file"
    fi

    echo "  Done: $ds"
    echo ""
done

echo "=========================================="
echo "  ALL DATASETS COMPLETE"
echo "=========================================="
ls -lh $OUT_DIR/signals_*_${N_AGENTS}a.json
