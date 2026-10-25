"""Build an exogenous DeepSeek resource schedule from Full Mechanism traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LIMITS = {
    "low_effort": 350,
    "aeic_effort": 1000,
    "defer_imitate": 500,
    "verify": 1000,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--condition-suffix", default="__variant@full_mechanism")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    schedule = {
        "schema_version": "matched-compute-schedule.v1",
        "source": str(args.records),
        "source_condition_suffix": args.condition_suffix,
        "profiles": {},
    }
    for record in rows:
        if (record.get("completed") is not True
                or not str(record.get("condition", "")).endswith(args.condition_suffix)):
            continue
        for item in record.get("rounds", []):
            action = str(item["chosen_action"])
            executions = [trace for trace in item.get("action_trace", [])
                          if trace.get("stage") != "choose"
                          and not str(trace.get("stage", "")).startswith("format_repair")]
            profile_key = "|".join((
                str(record["dataset"]), str(record["question_id"]),
                str(item["round_id"]), str(item["agent_id"]),
            ))
            schedule["profiles"][profile_key] = {
                "source_action": action,
                "backend": (executions[0].get("backend") if executions else "deepseek"),
                "model": (executions[0].get("model") if executions else "deepseek-chat"),
                "execution_calls": len(executions),
                "max_tokens_per_call": LIMITS[action],
                "source_actual_prompt_tokens": sum(int(x.get("prompt_tokens") or 0) for x in executions),
                "source_actual_completion_tokens": sum(int(x.get("completion_tokens") or 0) for x in executions),
            }
    if not schedule["profiles"]:
        raise SystemExit("no matching Full Mechanism profiles found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schedule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "profiles": len(schedule["profiles"]),
        "datasets": sorted({key.split("|", 1)[0] for key in schedule["profiles"]}),
        "out": str(args.out),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
