"""Build conservative ReDial prefixes with physically separate offline labels.

Run: python3 -m experiments.recommendation_v2.prepare
Case selection retains the legacy positive-target cutoff. Selection is therefore
label-dependent; serving inputs at a *fixed cutoff* are label-independent.
This is a diagnostic reconstruction, not a fresh independently sampled test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.scripts.data.build_redial_benchmark import (
    MOVIE_TOKEN, label_is_positive, load_catalog, map_movie, read_dialogues,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments/results/recommendation_v2/data"
VERSION = "redial-prefix-only.v2"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path: Path, value: object, jsonl: bool = False) -> None:
    """Create outputs, or accept exact reruns; never silently overwrite changes."""
    text = ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in value) if jsonl else
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"changed output requires a new version directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def movie_mapping(dialogue: dict, exact: dict, title_only: dict) -> dict:
    return {str(rid): map_movie(str(title), exact, title_only)[0]
            for rid, title in (dialogue.get("movieMentions") or {}).items()}


def prefix_input(dialogue: dict, cutoff: int, mapping: dict) -> dict:
    """Whitelist only prefix observations. Never inspect dialogue-level labels."""
    messages = dialogue.get("messages") or []
    if not 0 <= cutoff <= len(messages):
        raise ValueError("invalid cutoff")
    mentions = dialogue.get("movieMentions") or {}
    context, events, ids = [], [], set()
    for index, message in enumerate(messages[:cutoff]):
        raw = str(message.get("text", ""))
        if not raw.strip():
            continue
        role = ("seeker" if message.get("senderWorkerId") == dialogue.get("initiatorWorkerId")
                else "recommender")
        rendered = MOVIE_TOKEN.sub(lambda m: str(mentions.get(m.group(1), m.group(0))), raw)
        context.append({"message_index": index, "sender": role, "text": rendered})
        for rid in sorted(set(MOVIE_TOKEN.findall(raw))):
            mid = mapping.get(rid)
            if mid:
                ids.add(mid)
                events.append({"message_index": index, "sender": role,
                               "redial_movie_id": rid, "movie_id": mid,
                               "kind": "mention", "sentiment": "unknown", "seen": "unknown"})
    return {"version": VERSION, "case_id": f"redial:{dialogue['conversationId']}",
            "cutoff_exclusive": cutoff, "context": context,
            "query": "\n".join(f"{m['sender']}: {m['text']}" for m in context),
            "context_movie_ids": sorted(ids), "mention_events": events,
            "liked_movie_ids": [], "excluded_seen_movie_ids": [],
            "hard_constraints": {}, "preference_status": "not_extracted"}


def select_case(dialogue: dict, mapping: dict):
    """Offline sampler; its labels must never be passed to a recommender."""
    targets = set()
    for rid, label in (dialogue.get("initiatorQuestions") or {}).items():
        if (isinstance(label, dict) and label_is_positive(label.get("suggested"))
                and label_is_positive(label.get("liked")) and mapping.get(str(rid))):
            targets.add(mapping[str(rid)])
    if not targets:
        return None
    target_rids = {rid for rid, mid in mapping.items() if mid in targets}
    for index, message in enumerate(dialogue.get("messages") or []):
        if target_rids.intersection(MOVIE_TOKEN.findall(str(message.get("text", "")))):
            inp = prefix_input(dialogue, index, mapping)
            if inp["context"]:
                return inp, {"case_id": inp["case_id"], "relevant_movie_ids": sorted(targets)}
            return None
    return None


def validate_pair(inp: dict, label: dict, catalog_ids: set) -> None:
    if inp["case_id"] != label["case_id"]:
        raise ValueError("input/label case mismatch")
    if not label["relevant_movie_ids"]:
        raise ValueError("empty target")
    if set(label["relevant_movie_ids"]) & set(inp["context_movie_ids"]):
        raise ValueError("target mentioned in prefix")
    all_ids = set(label["relevant_movie_ids"]) | set(inp["context_movie_ids"])
    if not all_ids <= catalog_ids:
        raise ValueError("unmapped movie")
    if inp["liked_movie_ids"] or inp["excluded_seen_movie_ids"] or inp["hard_constraints"]:
        raise ValueError("unvalidated preference state")
    if any(e["message_index"] >= inp["cutoff_exclusive"] for e in inp["mention_events"]):
        raise ValueError("future event")


def snapshot_legacy() -> dict:
    files = []
    for directory in ("experiments/results/movie", "experiments/benchmarks/movie_consensus"):
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    for name in ("experiments/scripts/data/build_redial_benchmark.py",
                 "experiments/scripts/data/validate_movie_benchmark.py"):
        files.append(ROOT / name)
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = ROOT / "experiments/data/raw/redial"
    catalog = ROOT / "experiments/data/processed/movie_catalog.jsonl"
    exact, titles, movies = load_catalog(catalog)
    frozen = snapshot_legacy()
    splits = {name: [] for name in ("train", "dev", "legacy_test_diagnostic")}
    workers = {name: set() for name in splits}
    source_counts, unmapped, skipped = {}, Counter(), Counter()
    seen = set()
    for source_name in ("train", "test"):
        dialogues = read_dialogues(root / f"{source_name}.jsonl")
        source_counts[source_name] = len(dialogues)
        for d in dialogues:
            cid = f"redial:{d['conversationId']}"
            if cid in seen:
                raise ValueError(f"duplicate source conversation: {cid}")
            seen.add(cid)
            mapping = movie_mapping(d, exact, titles)
            unmapped[source_name] += sum(mid is None for mid in mapping.values())
            pair = select_case(d, mapping)
            if pair is None:
                skipped[source_name] += 1
                continue
            inp, label = pair
            validate_pair(inp, label, set(movies))
            bucket = int(hashlib.sha256(("prefix-v2:" + cid).encode()).hexdigest(), 16) % 5
            split = "legacy_test_diagnostic" if source_name == "test" else ("dev" if bucket == 0 else "train")
            splits[split].append(pair)
            workers[split].update(str(d[key]) for key in ("initiatorWorkerId", "respondentWorkerId")
                                  if d.get(key) is not None)
    outputs = {}
    for split, pairs in splits.items():
        pairs.sort(key=lambda pair: pair[0]["case_id"])
        for kind, index in (("inputs", 0), ("labels", 1)):
            p = args.output / f"{split}.{kind}.jsonl"
            save(p, [pair[index] for pair in pairs], jsonl=True)
            outputs[p.name] = {"rows": len(pairs), "sha256": digest(p)}
    # Retain a source-scope usage ledger; it is not a complete semantic search of history.
    legacy_ids = set()
    for name in frozen:
        if not name.endswith(".jsonl"):
            continue
        for line in (ROOT / name).open(encoding="utf-8"):
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict) and item.get("case_id"):
                    legacy_ids.add(str(item["case_id"]))
    overlaps = {f"{a}__{b}": len(workers[a] & workers[b])
                for i, a in enumerate(splits) for b in list(splits)[i + 1:]}
    manifest = {"version": VERSION, "source_counts": source_counts,
                "sources": {name: digest(root / f"{name}.jsonl") for name in ("train", "test")},
                "catalog_sha256": digest(catalog), "code_sha256": digest(Path(__file__)),
                "outputs": outputs, "skipped": dict(skipped),
                "unmapped_mention_entries": dict(unmapped), "worker_overlap": overlaps,
                "split_rule": "train source: sha256(prefix-v2:case_id)%5; bucket0=dev",
                "user_isolation": "not_enforced; worker overlap reported",
                "case_selection_uses_offline_labels": True,
                "input_at_fixed_cutoff_uses_offline_labels": False,
                "constraint_extraction": "not_performed; no constraint satisfaction claim",
                "fresh_independent_test": False,
                "history_scope": "legacy movie results and benchmark JSONL only; not exhaustive repo history",
                "legacy_case_ids_observed": len(legacy_ids),
                "current_cases_observed_in_scanned_history": {
                    s: sum(inp["case_id"] in legacy_ids for inp, _ in pairs)
                    for s, pairs in splits.items()}}
    save(args.output / "legacy_snapshot.json", frozen)
    save(args.output / "manifest.json", manifest)
    if snapshot_legacy() != frozen:
        raise RuntimeError("legacy evidence changed during preparation")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
