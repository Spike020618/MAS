"""Label-blind cutoffs, worker-purged splits, shared training-only retrieval.

Run: python3 -m experiments.consensus_recommendation.data
Official source test is not used. Reserved participants are never scored here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import ROOT, OUTPUT, create, fingerprint, read_rows, sha, verify_artifacts
from experiments.scripts.data.build_redial_benchmark import (
    MOVIE_TOKEN, label_is_positive, load_catalog, map_movie,
)
from experiments.recommendation_v2.audit import LexicalIndex

VERSION = "grounded-consensus-prefix.v1"


def worker_split(worker):
    h = int(hashlib.sha256(("evidence-consensus-worker-v1:" + str(worker)).encode()).hexdigest(), 16) % 20
    return "train" if h < 14 else "dev" if h < 17 else "reserved_test"


def cutoff(dialogue):
    """Before the first movie-bearing recommender turn after a seeker turn.

    Depends on message roles and movie-token occurrence, never sentiment/gold.
    A seeker can mention films in the prefix without being labelled as liking them.
    """
    seeker_seen = False
    for index, message in enumerate(dialogue.get("messages", [])):
        if message.get("senderWorkerId") == dialogue.get("initiatorWorkerId"):
            seeker_seen = seeker_seen or bool(str(message.get("text", "")).strip())
        elif seeker_seen and MOVIE_TOKEN.search(str(message.get("text", ""))):
            return index
    return None


def make_input(dialogue, exact, titles):
    stop = cutoff(dialogue)
    if stop is None:
        return None
    mentions = dialogue.get("movieMentions") or {}
    observations, mentioned = [], set()
    for i, msg in enumerate(dialogue["messages"][:stop]):
        raw = str(msg.get("text", ""))
        if not raw.strip():
            continue
        text = MOVIE_TOKEN.sub(lambda m: str(mentions.get(m.group(1), m.group(0))), raw)
        observations.append({"source_id": f"message:{i}", "message_index": i,
                             "role": "seeker" if msg.get("senderWorkerId") == dialogue.get("initiatorWorkerId") else "recommender",
                             "text": text})
        for rid in MOVIE_TOKEN.findall(raw):
            mid = map_movie(str(mentions.get(rid, "")), exact, titles)[0]
            if mid:
                mentioned.add(mid)
    if not observations:
        return None
    return {"version": VERSION, "case_id": f"redial:{dialogue['conversationId']}",
            "cutoff_exclusive": stop, "observations": observations,
            "context_movie_ids": sorted(mentioned),
            "query": "\n".join(x["role"] + ": " + x["text"] for x in observations),
            "preference_state": "raw_text_only; mentions_not_likes_or_seen"}


def make_label(dialogue, inp, exact, titles):
    mentions = dialogue.get("movieMentions") or {}
    future_rids = {rid for m in dialogue["messages"][inp["cutoff_exclusive"]:]
                   for rid in MOVIE_TOKEN.findall(str(m.get("text", "")))}
    ids = set()
    for rid, label in (dialogue.get("initiatorQuestions") or {}).items():
        if (str(rid) in future_rids and isinstance(label, dict)
                and label_is_positive(label.get("suggested")) and label_is_positive(label.get("liked"))):
            mid = map_movie(str(mentions.get(str(rid), "")), exact, titles)[0]
            if mid and mid not in inp["context_movie_ids"]:
                ids.add(mid)
    return {"case_id": inp["case_id"], "relevant_movie_ids": sorted(ids),
            "semantics": "future-mentioned, dialogue-questionnaire suggested+liked; not turn-local acceptance"}


def historical_case_ids():
    ids = set()
    # Bounded ledger, not a certification of all historical semantic exposure.
    for root in (ROOT / "experiments/results/movie", ROOT / "experiments/benchmarks/movie_consensus"):
        for p in root.rglob("*.jsonl"):
            for row in read_rows(p):
                if row.get("case_id"):
                    ids.add(str(row["case_id"]))
    return ids


def prepare(base):
    raw = ROOT / "experiments/data/raw/redial/train.jsonl"
    cat = ROOT / "experiments/data/processed/movie_catalog.jsonl"
    exact, titles, movies = load_catalog(cat)
    used = historical_case_ids()
    partitions = {s: [] for s in ("train", "dev", "reserved_test")}
    workers = {s: set() for s in partitions}
    counts = Counter()
    source_ids = set()
    for d in read_rows(raw):
        cid = f"redial:{d['conversationId']}"
        if cid in source_ids:
            raise ValueError("duplicate source dialogue")
        source_ids.add(cid)
        ws = [str(d[k]) for k in ("initiatorWorkerId", "respondentWorkerId")]
        groups = [worker_split(w) for w in ws]
        if groups[0] != groups[1]:
            counts["cross_partition_purged"] += 1
            continue
        inp = make_input(d, exact, titles)
        if inp is None:
            counts["no_structural_request_cutoff"] += 1
            continue
        split = groups[0]
        if split != "train" and cid in used:
            counts[f"{split}_bounded_usage_excluded"] += 1
            continue
        label = make_label(d, inp, exact, titles)
        # Keep unlabeled requests. Report their eligibility, never silently drop by gold.
        partitions[split].append((inp, label))
        workers[split].update(ws)
    hashes, sizes = {}, {}
    for split, pairs in partitions.items():
        pairs.sort(key=lambda pair: fingerprint(["sample-v1", pair[0]["case_id"]]))
        sizes[split] = {"requests": len(pairs), "participants": len(workers[split])}
        for kind, pos in (("inputs", 0), ("labels", 1)):
            name = f"{split}.{kind}.jsonl"
            create(base / name, [p[pos] for p in pairs], rows=True)
            hashes[name] = sha(base / name)
    overlap = {f"{a}:{b}": len(workers[a] & workers[b])
               for i, a in enumerate(workers) for b in list(workers)[i+1:]}
    if any(overlap.values()):
        raise ValueError("participant leakage")
    manifest = {"version": VERSION, "sources": {str(raw.relative_to(ROOT)): sha(raw), str(cat.relative_to(ROOT)): sha(cat)},
                "input_builder_sha256": sha(Path(__file__)), "artifacts": hashes,
                "sizes_without_scoring_reserved_labels": sizes, "excluded": dict(counts),
                "worker_overlap": overlap, "cutoff_uses_labels": False,
                "test_source_used": False, "reserved_scored": False,
                "bounded_usage_ledger_case_count": len(used),
                "limits": ["prior semantic exposure not exhaustively certified", "questionnaire positives are not prospective choice utility",
                           "worker hash + cross-edge purging changes population", "no hard-constraint annotations", "catalog snapshot, not historical reconstruction"]}
    create(base / "manifest.json", manifest)
    return movies


class Retriever:
    """Training-supervised context transfer plus lexical and neutral cooccurrence.

    All query-time methods read serving inputs only. Matching past contexts is
    an existing retrieval baseline, not a novel multi-agent mechanism.
    """
    def __init__(self, train_inputs, train_labels, movies):
        self.movies = movies
        self.ids = sorted(movies)
        self.positions = {m: i for i, m in enumerate(self.ids)}
        self.lexical = LexicalIndex(movies)
        labels = {r["case_id"]: r["relevant_movie_ids"] for r in train_labels}
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                                          max_features=40000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([r["query"] for r in train_inputs])
        row, col = [], []
        self.pop, self.assoc = Counter(), defaultdict(Counter)
        for n, inp in enumerate(train_inputs):
            for mid in labels[inp["case_id"]]:
                row.append(n)
                col.append(self.positions[mid])
                self.pop[mid] += 1
                for anchor in inp["context_movie_ids"]:
                    self.assoc[anchor][mid] += 1
        self.targets = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(train_inputs), len(self.ids)))
        self.prior = sorted(self.ids, key=lambda m: (-self.pop[m], m))

    def search(self, inp, limit=100):
        excluded = set(inp["context_movie_ids"])  # previously mentioned, not asserted seen
        lex = self.lexical.search({"query": inp["query"]}, limit=300)
        sim = (self.matrix @ self.vectorizer.transform([inp["query"]]).T).toarray().ravel()
        best = np.argsort(-sim, kind="stable")[:50]
        weight = sparse.csr_matrix((sim[best], (np.zeros(len(best), dtype=int), best)), shape=(1, len(sim)))
        scores = (weight @ self.targets).toarray().ravel()
        near = [self.ids[i] for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:300]
        assoc = Counter()
        for anchor in sorted(excluded):
            for mid, n in self.assoc.get(anchor, {}).items():
                assoc[mid] += n / max(1, self.pop[mid])**0.5
        co = sorted(assoc, key=lambda m: (-assoc[m], m))[:300]
        fused = Counter()
        for ranked in (lex, near, co, self.prior[:300]):
            for rank, mid in enumerate(ranked):
                if mid not in excluded:
                    fused[mid] += 1.0 / (60 + rank + 1)
        pool = sorted(fused, key=lambda m: (-fused[m], m))[:limit]
        return {"case_id": inp["case_id"], "ranked_movie_ids": pool,
                "lexical_movie_ids": [m for m in lex if m not in excluded][:limit],
                "example_movie_ids": [m for m in near if m not in excluded][:limit]}


def audit(base, movies):
    manifest = json.loads((base / "manifest.json").read_text())
    verify_artifacts(base, manifest["artifacts"])
    train = read_rows(base / "train.inputs.jsonl")
    retriever = Retriever(train, read_rows(base / "train.labels.jsonl"), movies)
    inputs = read_rows(base / "dev.inputs.jsonl")
    labels = {r["case_id"]: r for r in read_rows(base / "dev.labels.jsonl")}
    rows = [retriever.search(inp) for inp in inputs]
    create(base / "dev.candidates.jsonl", rows, rows=True)
    stats = {}
    for method, key in (("hybrid", "ranked_movie_ids"), ("lexical", "lexical_movie_ids"), ("examples", "example_movie_ids")):
        counts = Counter()
        for inp, r in zip(inputs, rows):
            gold = set(labels[inp["case_id"]]["relevant_movie_ids"])
            counts["requests"] += 1
            counts["annotated_requests"] += bool(gold)
            for k in (5, 20, 50, 100):
                counts[f"hit_{k}"] += bool(gold & set(r[key][:k]))
            counts["pool_has_target_top5_misses"] += bool(gold & set(r[key])) and not bool(gold & set(r[key][:5]))
        n = counts["annotated_requests"]
        stats[method] = {**dict(counts), "hit_rate_among_annotated": {str(k): counts[f"hit_{k}"]/max(1,n) for k in (5,20,50,100)}}
    # Gate fixed before audit results: enough independently sampled pilot cases,
    # and >=10 percentage points of headroom among annotated development cases.
    h = stats["hybrid"]
    eligible = [i["case_id"] for i in inputs if labels[i["case_id"]]["relevant_movie_ids"]]
    pilot = eligible[12:60]
    headroom = h["pool_has_target_top5_misses"] / max(1,h["annotated_requests"])
    gate = len(eligible) >= 60 and headroom >= 0.10
    report = {"methods": stats, "paid_pilot_data_gate": gate,
              "gate_rule": "at least 60 annotated dev requests; pool-vs-top5 headroom >=0.10",
              "headroom": headroom, "engineering_case_ids": eligible[:12], "pilot_case_ids": pilot,
              "sample_selection": "fixed hash order among nonempty offline target labels; no selection on baseline error or conflict",
              "reserved_labels_read_by_auditor": False, "model_calls": 0,
              "candidate_sha256": sha(base / "dev.candidates.jsonl")}
    create(base / "development_retrieval_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "data")
    args = parser.parse_args()
    movies = prepare(args.output)
    audit(args.output, movies)


if __name__ == "__main__":
    main()
