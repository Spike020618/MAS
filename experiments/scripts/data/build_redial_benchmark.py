#!/usr/bin/env python3
"""Create leakage-safe ReDial dev/test recommendation benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


YEAR = re.compile(r"\((\d{4})\)\s*$")
ARTICLE = re.compile(r",\s*(the|a|an)$", re.I)
MOVIE_TOKEN = re.compile(r"@(\d+)")


def normalize_title(raw: str) -> tuple[str, int | None]:
    raw = str(raw).strip()
    match = YEAR.search(raw)
    year = int(match.group(1)) if match else None
    title = YEAR.sub("", raw).strip()
    article = ARTICLE.search(title)
    if article:
        title = f"{article.group(1)} {title[:article.start()]}"
    title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return title, year


def load_catalog(path: Path):
    exact, title_only, by_id = defaultdict(list), defaultdict(list), {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            title, _ = normalize_title(item["title"])
            exact[(title, item.get("year"))].append(item["movie_id"])
            title_only[title].append(item["movie_id"])
            by_id[item["movie_id"]] = item
    return exact, title_only, by_id


def map_movie(name: str, exact, title_only) -> tuple[str | None, str]:
    title, year = normalize_title(name)
    ids = exact.get((title, year), []) if year else []
    if len(ids) == 1:
        return ids[0], "exact_title_year"
    ids = title_only.get(title, [])
    if len(ids) == 1:
        return ids[0], "unique_title"
    return None, "ambiguous_or_missing"


def label_is_positive(value: object) -> bool:
    return value is True or value == 1 or str(value).lower() == "true"


def dialogue_to_case(dialogue: dict, exact, title_only) -> tuple[dict | None, list[dict]]:
    mentions = {str(k): str(v) for k, v in (dialogue.get("movieMentions") or {}).items()}
    labels = dialogue.get("initiatorQuestions") or {}
    mapped, review = {}, []
    for redial_id, name in mentions.items():
        movie_id, method = map_movie(name, exact, title_only)
        mapped[redial_id] = movie_id
        if movie_id is None:
            review.append({"conversation_id": dialogue.get("conversationId"),
                           "redial_movie_id": redial_id, "redial_title": name,
                           "status": "needs_review", "resolved_movie_id": ""})

    relevant = []
    for redial_id, label in labels.items():
        if not isinstance(label, dict):
            continue
        if label_is_positive(label.get("suggested")) and label_is_positive(label.get("liked")):
            if mapped.get(str(redial_id)):
                relevant.append(mapped[str(redial_id)])
    relevant = sorted(set(relevant))
    if not relevant:
        return None, review

    relevant_redial = {rid for rid, mid in mapped.items() if mid in relevant}
    messages = dialogue.get("messages") or []
    cutoff = len(messages)
    for index, message in enumerate(messages):
        if relevant_redial.intersection(MOVIE_TOKEN.findall(str(message.get("text", "")))):
            cutoff = index
            break
    context_messages = messages[:cutoff]
    if not context_messages:
        return None, review
    def render(text: object) -> str:
        return MOVIE_TOKEN.sub(lambda m: mentions.get(m.group(1), m.group(0)), str(text))

    context = [
        {"sender": "seeker" if msg.get("senderWorkerId") == dialogue.get("initiatorWorkerId") else "recommender",
         "text": render(msg.get("text", ""))}
        for msg in context_messages if str(msg.get("text", "")).strip()
    ]
    if not context:
        return None, review

    seen = []
    liked_context = []
    for redial_id, label in labels.items():
        if isinstance(label, dict) and label_is_positive(label.get("seen")) and mapped.get(str(redial_id)):
            seen.append(mapped[str(redial_id)])
        if isinstance(label, dict) and label_is_positive(label.get("liked")) and mapped.get(str(redial_id)):
            if str(redial_id) not in relevant_redial:
                liked_context.append(mapped[str(redial_id)])
    return {
        "case_id": f"redial:{dialogue.get('conversationId')}",
        "conversation_id": dialogue.get("conversationId"),
        "context": context,
        "query": "\n".join(f"{x['sender']}: {x['text']}" for x in context),
        "relevant_movie_ids": relevant,
        "excluded_seen_movie_ids": sorted(set(seen) - set(relevant)),
        "liked_context_movie_ids": sorted(set(liked_context) - set(relevant)),
        "source": "ReDial",
        "label_rule": "initiator suggested=1 and liked=1; context truncated before first relevant mention",
    }, review


def find_split(root: Path, split: str) -> Path:
    candidates = [p for p in root.rglob("*") if p.is_file() and split in p.name.lower()
                  and p.suffix.lower() in {".jsonl", ".json"}]
    if not candidates:
        raise FileNotFoundError(f"cannot find ReDial {split} file under {root}")
    return sorted(candidates, key=lambda p: ("data" not in p.name.lower(), len(str(p))))[0]


def read_dialogues(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("dialogues", [])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redial-root", type=Path, default=Path("experiments/data/raw/redial"))
    parser.add_argument("--catalog", type=Path, default=Path("experiments/data/processed/movie_catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/movie_consensus"))
    parser.add_argument("--dev-size", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    exact, title_only, _ = load_catalog(args.catalog)
    rng = random.Random(args.seed)
    summary, all_review = {}, []
    for split, size in (("train", args.dev_size), ("test", args.test_size)):
        source = find_split(args.redial_root, split)
        cases = []
        for dialogue in read_dialogues(source):
            case, review = dialogue_to_case(dialogue, exact, title_only)
            all_review.extend(review)
            if case:
                cases.append(case)
        rng.shuffle(cases)
        chosen = sorted(cases[:size], key=lambda x: str(x["case_id"]))
        if len(chosen) < size:
            raise RuntimeError(f"{split}: only {len(chosen)} usable mapped cases, need {size}")
        output = args.output / ("dev.jsonl" if split == "train" else "test.jsonl")
        write_jsonl(output, chosen)
        summary[split] = {"source": str(source), "eligible": len(cases), "selected": len(chosen),
                          "output": str(output), "sha256": digest(output)}
    review_path = args.output / "mapping_review.jsonl"
    unique_review = {f"{x['conversation_id']}:{x['redial_movie_id']}": x for x in all_review}
    write_jsonl(review_path, list(unique_review.values()))
    manifest = {"version": "movie-consensus.v1", "seed": args.seed, "splits": summary,
                "mapping_review_rows": len(unique_review), "catalog_sha256": digest(args.catalog)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
