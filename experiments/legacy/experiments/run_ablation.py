"""
Ablation study — zero out each signal and evaluate.

Usage:
  python experiments/run_ablation.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config, compute_ic_metric


CONFIG_PATH = "experiments/results/archive/configs/opt_MMLU-Hard+StrategyQA.json"
SIGNALS_DIR = "experiments/results/signals"


def load_all_signals():
    datasets = {}
    for fn in sorted(os.listdir(SIGNALS_DIR)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(SIGNALS_DIR, fn)
        with open(path, "r") as f:
            data = json.load(f)
        ds_name = data["meta"]["dataset"]
        datasets[ds_name] = data
    return datasets


def zero_signal(config: ConsensusConfig, signal: str) -> ConsensusConfig:
    """Create a config with a specific signal zeroed out."""
    d = {f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()}

    if signal == "IC":
        d["aw_ic_exp"] = 0.0
        d["ha_ic"] = 0.0
    elif signal == "EC":
        d["mrd_lo"] = 0.5
        d["mrd_hi"] = 0.5
        d["ha_ec"] = 0.0
    elif signal == "EQ":
        d["aw_eq_exp"] = 0.0
        d["ha_eq"] = 0.0
    elif signal == "EV":
        d["aw_ev_scale"] = 0.0
        d["ha_ev"] = 0.0

    return ConsensusConfig(**d)


def main():
    base_config = ConsensusConfig.load(CONFIG_PATH)
    datasets = load_all_signals()

    configs = {
        "Full mechanism": base_config,
        "Remove IC": zero_signal(base_config, "IC"),
        "Remove EC": zero_signal(base_config, "EC"),
        "Remove EQ": zero_signal(base_config, "EQ"),
        "Remove EV": zero_signal(base_config, "EV"),
    }

    print(f"{'Configuration':<22} {'Avg Acc':>8} {'Delta':>8}")
    print("-" * 42)

    full_acc = None
    results = {}

    for name, cfg in configs.items():
        accs = []
        for ds_name, data in datasets.items():
            n_agents = data["meta"]["n_agents"]
            result = evaluate_config(data["questions"], cfg, n_agents)
            accs.append(result["accuracy"])
        avg = sum(accs) / len(accs)

        if full_acc is None:
            full_acc = avg
            delta = 0.0
        else:
            delta = avg - full_acc

        results[name] = {"avg_accuracy": avg, "delta": delta}
        print(f"{name:<22} {avg:>7.1%} {delta:>+7.1%}")

    out_path = "experiments/results/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
