"""Collect paired, compute-controlled report-interface observations."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.core.llm import LLMClient
from experiments.mechanism.collect import load_local_questions
from experiments.mechanism.report_interface import (
    PROTOCOLS, observe_interface, protocol_prompt, response_format,
)
from experiments.mechanism.runtime import classify_error


def key(item: dict) -> tuple:
    return (str(item["experiment_id"]), str(item["dataset"]),
            str(item["question_id"]), str(item["protocol"]), int(item["seed"]))


def load_records(path: Path) -> dict:
    records = {}
    if not path.exists():
        return records
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            item_key = key(item)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid checkpoint {path} line {line_no}: {exc}") from exc
        if item_key in records:
            raise ValueError(f"duplicate checkpoint unit {item_key}")
        records[item_key] = item
    return records


def atomic_upsert(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = load_records(path)
        item_key = key(item)
        previous = records.get(item_key)
        if previous and previous.get("completed") is True:
            return
        item["attempt_count"] = int((previous or {}).get("attempt_count", 0)) + 1
        item["prior_errors"] = list((previous or {}).get("prior_errors") or [])
        if previous and previous.get("error"):
            item["prior_errors"].append(previous["error"])
        records[item_key] = item
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for record in sorted(records.values(), key=key):
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)


def collect_one(client: Optional[LLMClient], question: dict, dataset: str, protocol: str,
                experiment_id: str, seed: int, max_tokens: int,
                temperature: float, dry_run: bool) -> dict:
    started = time.monotonic()
    base = {
        "schema_version": "report-interface.v1", "experiment_id": experiment_id,
        "dataset": dataset, "question_id": str(question["question_id"]),
        "protocol": protocol, "seed": seed, "ground_truth": question["ground_truth"],
        "backend": "dry_run" if dry_run else client.backend,
        "model": "dry_run" if dry_run else client.model,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    try:
        if dry_run:
            raw = {
                "free_text": "Evidence is mocked. Yes: mock answer",
                "generic_structure": '{"analysis":"mock reason","answer":"Yes: mock"}',
                "aeic": '{"A":"mock","E":["mock evidence"],"I":"mock","C":"Yes: mock"}',
            }[protocol]
            usage = {"prompt_tokens": 5, "completion_tokens": 8}
            latency = 0.001
        else:
            kwargs = {}
            fmt = response_format(protocol)
            if fmt:
                kwargs["response_format"] = fmt
            raw = client.chat(
                [{"role": "system", "content": "Answer carefully and concisely."},
                 {"role": "user", "content": protocol_prompt(question["question"], protocol)}],
                temperature=temperature, max_tokens=max_tokens, **kwargs,
            )
            usage = dict(client.last_usage)
            latency = float(client.last_latency_s or time.monotonic() - started)
        observation = observe_interface(raw, protocol)
        return {**base, "raw_response": raw, "observation": observation.to_dict(),
                "token_input": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "token_output": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                "latency_s": latency, "completed": True, "error": None}
    except Exception as exc:
        return {**base, "raw_response": None, "observation": None,
                "token_input": 0, "token_output": 0,
                "latency_s": time.monotonic() - started, "completed": False,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["StrategyQA", "MMLU-Hard", "TruthfulQA"])
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--backend", default="deepseek")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--experiment-id", default="report-interface-v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n_samples < 1 or args.max_tokens < 1 or args.max_attempts < 1:
        parser.error("sample, token and attempt counts must be positive")
    questions = ([{"question": "Mock question?", "ground_truth": "yes", "question_id": "dry-0"}]
                 if args.dry_run else load_local_questions(args.dataset, args.n_samples, args.split))
    client = None if args.dry_run else LLMClient(backend=args.backend, max_tokens=args.max_tokens)
    existing = load_records(args.out)
    for question in questions[:args.n_samples]:
        for protocol in args.protocols:
            item_key = (args.experiment_id + ("-dry" if args.dry_run else ""),
                        args.dataset, str(question["question_id"]), protocol, args.seed)
            if existing.get(item_key, {}).get("completed") is True:
                continue
            for attempt in range(1, args.max_attempts + 1):
                record = collect_one(
                    client, question, args.dataset, protocol, item_key[0], args.seed,
                    args.max_tokens, args.temperature, args.dry_run,
                )
                atomic_upsert(args.out, record)
                print(f"{question['question_id']} {protocol} attempt {attempt}: "
                      f"{'complete' if record['completed'] else record['error']}")
                if record["completed"]:
                    break
                if classify_error(record["error"]) == "protocol":
                    break
                time.sleep(min(2 ** (attempt - 1), 4))
            if not record["completed"]:
                raise SystemExit(f"stopping at incomplete unit {item_key}")


if __name__ == "__main__":
    main()
