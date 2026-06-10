"""
Analyze the value of AEIC structured output for stance extraction.
Compares majority voting accuracy using C-field vs I-field extractions.
"""

import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOLOGIES = ["star", "parallel", "debate", "sequential", "hierarchical"]


def extract_stance_from_text(text, ground_truth_hint=None):
    """Extract yes/no/neutral stance from free-form (unstructured) text.

    Simulates stance extraction from text that lacks explicit answer markers.
    Uses aggressive pattern matching including Chinese patterns to be as fair
    as possible to the I-field baseline.
    """
    if not text or not isinstance(text, str):
        return "neutral"

    # Handle cases where I-field is a JSON string (malformed AEIC)
    if isinstance(text, str) and text.strip().startswith('{'):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ['I', 'C', 'conclusion']:
                    if key in parsed:
                        return extract_stance_from_text(parsed[key])
        except json.JSONDecodeError:
            pass

    text_clean = text.strip()
    text_lower = text_clean.lower()

    # Check for clear yes/no at start
    if re.match(r'^(yes|是的|对|正确|true)', text_lower):
        return "yes"
    if re.match(r'^(no|否|不是|不对|错误|false)', text_lower):
        return "no"

    # Check for answer letter patterns (A/B/C/D)
    letter_match = re.match(r'^([a-d])[:\.\s\)]', text_lower)
    if letter_match:
        return letter_match.group(1).upper()

    # Chinese patterns for yes/no embedded in reasoning
    # "支持Yes/No结论" patterns
    yes_patterns = [
        r'支持["\']?[Yy]es["\']?', r'结论[为是].*?[Yy]es',
        r'一致支持.*?肯定', r'答案[为是].*?[Yy]es',
        r'结论.*?正确', r'陈述.*?正确', r'该说法.*?正确',
        r'支持肯定', r'结论是肯定的',
    ]
    no_patterns = [
        r'支持["\']?[Nn]o["\']?', r'结论[为是].*?[Nn]o',
        r'一致支持.*?否定', r'答案[为是].*?[Nn]o',
        r'结论.*?不正确', r'陈述.*?不正确', r'该说法.*?不正确',
        r'支持否定', r'结论是否定的',
        r'不可能', r'原陈述不正确',
    ]

    yes_score = sum(1 for p in yes_patterns if re.search(p, text_clean))
    no_score = sum(1 for p in no_patterns if re.search(p, text_clean))

    if yes_score > no_score and yes_score > 0:
        return "yes"
    if no_score > yes_score and no_score > 0:
        return "no"

    # Fallback: English yes/no anywhere
    has_yes = bool(re.search(r'\byes\b', text_lower))
    has_no = bool(re.search(r'\bno\b', text_lower))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"

    # Last resort: check for Chinese affirmative/negative conclusion words
    # at the end of the text (where conclusions tend to appear)
    last_100 = text_clean[-100:] if len(text_clean) > 100 else text_clean
    if re.search(r'(因此|综上|总结).*?(正确|成立|可行|是的|肯定)', last_100):
        return "yes"
    if re.search(r'(因此|综上|总结).*?(不正确|不成立|不可行|否定|不是)', last_100):
        return "no"

    return "neutral"


def extract_stance_from_conclusion(conclusion_text):
    """Extract stance from structured C-field (e.g., 'Yes: explanation')."""
    if not conclusion_text or not isinstance(conclusion_text, str):
        return "neutral"

    text = conclusion_text.strip()

    # C-field typically starts with "Yes:" or "No:"
    if re.match(r'^(Yes|yes|YES)', text):
        return "yes"
    if re.match(r'^(No|no|NO)', text):
        return "no"

    # Check for answer letter
    letter_match = re.match(r'^([A-Da-d])[:\.\s\)]', text)
    if letter_match:
        return letter_match.group(1).upper()

    return extract_stance_from_text(text)


def majority_vote(stances):
    """Compute majority vote from list of stances."""
    from collections import Counter
    valid = [s for s in stances if s != "neutral"]
    if not valid:
        return "neutral"
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


def check_correct(predicted, ground_truth):
    """Check if prediction matches ground truth."""
    gt = str(ground_truth).strip().lower()
    pred = str(predicted).strip().lower()

    if gt in ["true", "yes", "1"]:
        return pred == "yes"
    if gt in ["false", "no", "0"]:
        return pred == "no"

    # Letter answers
    return pred.lower() == gt.lower()


def analyze_dataset_topology(dataset, topology):
    """Analyze C-field vs I-field extraction for one dataset-topology pair."""
    cache_path = f"results/cache/{dataset}_{topology}.json"
    if not os.path.exists(cache_path):
        # Try parallel = base dataset
        if topology == "parallel":
            cache_path = f"results/cache/{dataset}.json"
        if not os.path.exists(cache_path):
            return None

    with open(cache_path) as f:
        data = json.load(f)

    questions = data["questions"]

    c_stances_all = []
    i_stances_all = []
    c_correct = 0
    i_correct = 0
    c_neutral_count = 0
    i_neutral_count = 0
    total = 0

    for q in questions:
        gt = q["ground_truth"]
        agents = q["agents"]

        c_stances = []
        i_stances = []

        for agent in agents:
            aeic = agent.get("aeic", {})
            if not aeic:
                continue

            # Extract from C-field
            c_text = aeic.get("C", "")
            c_stance = extract_stance_from_conclusion(c_text)
            c_stances.append(c_stance)
            if c_stance == "neutral":
                c_neutral_count += 1

            # Extract from I-field (inference - less structured)
            i_text = aeic.get("I", "")
            i_stance = extract_stance_from_text(i_text)
            i_stances.append(i_stance)
            if i_stance == "neutral":
                i_neutral_count += 1

        c_stances_all.extend(c_stances)
        i_stances_all.extend(i_stances)

        c_vote = majority_vote(c_stances)
        i_vote = majority_vote(i_stances)

        if check_correct(c_vote, gt):
            c_correct += 1
        if check_correct(i_vote, gt):
            i_correct += 1
        total += 1

    total_agents = len(c_stances_all)

    return {
        "dataset": dataset,
        "topology": topology,
        "n_questions": total,
        "c_field": {
            "majvote_accuracy": round(c_correct / total * 100, 1) if total > 0 else 0,
            "neutral_rate": round(c_neutral_count / total_agents * 100, 1) if total_agents > 0 else 0,
            "n_neutral": c_neutral_count,
            "n_total_stances": total_agents,
        },
        "i_field": {
            "majvote_accuracy": round(i_correct / total * 100, 1) if total > 0 else 0,
            "neutral_rate": round(i_neutral_count / total_agents * 100, 1) if total_agents > 0 else 0,
            "n_neutral": i_neutral_count,
            "n_total_stances": total_agents,
        },
        "delta_accuracy": round((c_correct - i_correct) / total * 100, 1) if total > 0 else 0,
    }


def main():
    results = {}

    # Analyze all dataset-topology combinations
    for dataset in DATASETS:
        results[dataset] = {}
        for topo in TOPOLOGIES:
            r = analyze_dataset_topology(dataset, topo)
            if r:
                results[dataset][topo] = r

    # Compute per-dataset averages across topologies
    summary = {}
    for dataset in DATASETS:
        c_accs = []
        i_accs = []
        c_neutrals = []
        i_neutrals = []
        for topo, r in results[dataset].items():
            c_accs.append(r["c_field"]["majvote_accuracy"])
            i_accs.append(r["i_field"]["majvote_accuracy"])
            c_neutrals.append(r["c_field"]["neutral_rate"])
            i_neutrals.append(r["i_field"]["neutral_rate"])

        summary[dataset] = {
            "c_accuracy_mean": round(sum(c_accs) / len(c_accs), 1) if c_accs else 0,
            "i_accuracy_mean": round(sum(i_accs) / len(i_accs), 1) if i_accs else 0,
            "delta_mean": round((sum(c_accs) - sum(i_accs)) / len(c_accs), 1) if c_accs else 0,
            "c_neutral_mean": round(sum(c_neutrals) / len(c_neutrals), 1) if c_neutrals else 0,
            "i_neutral_mean": round(sum(i_neutrals) / len(i_neutrals), 1) if i_neutrals else 0,
        }

    # Overall averages
    all_c = [summary[d]["c_accuracy_mean"] for d in DATASETS]
    all_i = [summary[d]["i_accuracy_mean"] for d in DATASETS]
    overall = {
        "c_accuracy": round(sum(all_c) / len(all_c), 1),
        "i_accuracy": round(sum(all_i) / len(all_i), 1),
        "delta": round((sum(all_c) - sum(all_i)) / len(all_c), 1),
    }

    output = {
        "description": "AEIC structured output value: C-field vs I-field stance extraction",
        "per_topology": results,
        "per_dataset_summary": summary,
        "overall": overall,
    }

    # Save
    out_path = "results/structured_output_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("=" * 70)
    print("AEIC Structured Output Value: C-field vs I-field Stance Extraction")
    print("=" * 70)

    print(f"\n{'Dataset':<15} {'C-field Acc':>12} {'I-field Acc':>12} {'Delta':>8} {'C-neutral':>10} {'I-neutral':>10}")
    print("-" * 70)
    for dataset in DATASETS:
        s = summary[dataset]
        print(f"{dataset:<15} {s['c_accuracy_mean']:>11.1f}% {s['i_accuracy_mean']:>11.1f}% {s['delta_mean']:>+7.1f}pp {s['c_neutral_mean']:>9.1f}% {s['i_neutral_mean']:>9.1f}%")
    print("-" * 70)
    print(f"{'Overall':<15} {overall['c_accuracy']:>11.1f}% {overall['i_accuracy']:>11.1f}% {overall['delta']:>+7.1f}pp")

    print(f"\nPer-topology breakdown (C-field / I-field accuracy):")
    for topo in TOPOLOGIES:
        vals = []
        for d in DATASETS:
            if topo in results[d]:
                r = results[d][topo]
                vals.append(f"{r['c_field']['majvote_accuracy']:.0f}/{r['i_field']['majvote_accuracy']:.0f}")
            else:
                vals.append("--/--")
        print(f"  {topo:<15} " + "  ".join(f"{d}: {v}" for d, v in zip(DATASETS, vals)))

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
