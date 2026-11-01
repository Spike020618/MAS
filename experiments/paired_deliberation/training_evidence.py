"""Explain the existing train-only Retriever without modifying candidate pools.

Run offline with the existing scipy/sklearn environment. Serving consumes exported
JSON via training_evidence_view; the network runtime need not import scipy.
"""
import argparse
import math
from collections import Counter
from pathlib import Path
import json

import numpy as np

from experiments.consensus_recommendation.common import ROOT, OUTPUT, create, fingerprint, read_rows, sha
from experiments.consensus_recommendation.data import Retriever
from experiments.paired_deliberation.training_evidence_view import render

SERVING_FIELDS = {"version", "case_id", "cutoff_exclusive", "observations", "context_movie_ids", "query", "preference_state"}


class TrainingEvidence:
    def __init__(self, train_inputs, train_labels, movies):
        self.train = train_inputs
        ids = [r["case_id"] for r in train_inputs]
        labels = [r["case_id"] for r in train_labels]
        if len(ids) != len(set(ids)) or len(labels) != len(set(labels)) or set(ids) != set(labels):
            raise ValueError("training input/target identity mismatch")
        self.positive = {r["case_id"]: set(r["relevant_movie_ids"]) for r in train_labels}
        self.retriever = Retriever(train_inputs, train_labels, movies)
        self.ids = set(ids)
        self.source_hash = fingerprint({"train_inputs": train_inputs, "train_targets": train_labels, "catalog": movies})

    def explain(self, inp, pool, visible_neighbors=5):
        if set(inp) - SERVING_FIELDS: raise ValueError("unexpected serving fields; evaluation labels prohibited")
        if inp["case_id"] in self.ids: raise ValueError("serving case overlaps training")
        if len(pool) < 5 or len(pool) != len(set(pool)) or not set(pool) <= set(self.retriever.ids):
            raise ValueError("invalid frozen candidate pool")
        if type(visible_neighbors) is not int or not 0 <= visible_neighbors <= 50: raise ValueError("invalid display bound")
        r = self.retriever
        sim = (r.matrix @ r.vectorizer.transform([inp["query"]]).T).toarray().ravel()
        best = np.argsort(-sim, kind="stable")[:50]
        weights = __import__("scipy.sparse", fromlist=["csr_matrix"]).csr_matrix(
            (sim[best], (np.zeros(len(best), dtype=int), best)), shape=(1, len(sim)))
        scores = (weights @ r.targets).toarray().ravel()
        near = [r.ids[i] for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:300]
        excluded = set(inp["context_movie_ids"])
        assoc = Counter()
        for anchor in sorted(excluded):
            for mid, count in r.assoc.get(anchor, {}).items():
                assoc[mid] += count / max(1, r.pop[mid]) ** .5
        co = sorted(assoc, key=lambda mid: (-assoc[mid], mid))[:300]
        lex = r.lexical.search({"query": inp["query"]}, limit=300)
        fused = Counter()
        channels = (lex, near, co, r.prior[:300])
        for ranked in channels:
            for rank, mid in enumerate(ranked):
                if mid not in excluded: fused[mid] += 1 / (60 + rank + 1)
        reconstructed = sorted(fused, key=lambda mid: (-fused[mid], mid))[:len(pool)]
        original = r.search(inp, limit=len(pool))["ranked_movie_ids"]
        if reconstructed != original: raise ValueError("score decomposition disagrees with existing retriever")
        positive_neighbors = [int(i) for i in best if sim[i] > 0]
        visible = positive_neighbors[:visible_neighbors]
        pool_set = set(pool)
        neighbor_contributions, examples = [], []
        for i in positive_neighbors:
            record = self.train[i]
            targets = self.positive[record["case_id"]]
            neighbor_contributions.append({"training_case_id": record["case_id"], "similarity": float(sim[i]),
                "historical_targets_in_pool": sorted(targets & pool_set), "outside_pool_target_count": len(targets - pool_set)})
            if i in visible:
                examples.append({"origin": "historical_training_context", "training_case_id": record["case_id"],
                    "similarity": round(float(sim[i]), 8), "training_prefix": record["query"],
                    "historical_targets_in_pool": sorted(targets & pool_set), "outside_pool_target_count": len(targets - pool_set),
                    "outcome_semantics": "training future-mentioned + dialogue-questionnaire suggested/liked; not current-user or turn-local acceptance"})
        maps = [{mid: n+1 for n, mid in enumerate(ranked)} for ranked in (lex, near, co, r.prior)]
        candidate_signals = []
        for mid in pool:
            full = float(scores[r.positions[mid]])
            shown = sum(float(sim[i]) for i in visible if mid in self.positive[self.train[i]["case_id"]])
            omitted = full - shown
            if omitted < -1e-10: raise ValueError("negative omitted contribution")
            # Remove only floating subtraction noise; full precision remains in full score.
            omitted = max(0.0, omitted)
            from_trace = sum(n["similarity"] for n in neighbor_contributions if mid in n["historical_targets_in_pool"])
            if not math.isclose(full, from_trace, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("training neighbor contribution sum mismatch")
            candidate_signals.append({"movie_id": mid, "context_transfer_score": full, "association_score": float(assoc[mid]),
                "training_positive_count": int(r.pop[mid]), "lexical_rank": maps[0].get(mid),
                "context_transfer_rank": maps[1].get(mid), "association_rank": maps[2].get(mid),
                "popularity_rank": maps[3].get(mid), "fused_rrf_score": float(fused[mid]),
                "visible_context_support_mass": shown, "omitted_context_support_mass": omitted})
        return {"case_id": inp["case_id"], "input_hash": fingerprint(inp), "training_source_hash": self.source_hash,
            "candidate_signals": candidate_signals, "visible_neighbors": examples,
            "positive_similarity_neighbors": len(positive_neighbors), "neighbor_contributions": neighbor_contributions,
            "reproduced_frozen_pool": original == list(pool), "frozen_candidate_ids": list(pool),
            "scope": "existing training-supervised context transfer, mention-to-training-target association and popularity; not AgentCF or user-profile CF"}


def export(output):
    output, data = Path(output), OUTPUT / "data"
    source_manifest = json.loads((data / "manifest.json").read_text())
    sources = [data / name for name in ("train.inputs.jsonl", "train.labels.jsonl", "dev.inputs.jsonl", "dev.candidates.jsonl", "manifest.json")]
    for name in ("train.inputs.jsonl", "train.labels.jsonl", "dev.inputs.jsonl"):
        if sha(data / name) != source_manifest["artifacts"][name]: raise ValueError("frozen data changed")
    catalog = ROOT / "experiments/data/processed/movie_catalog.jsonl"
    if sha(catalog) != source_manifest["sources"][str(catalog.relative_to(ROOT))]: raise ValueError("catalog changed")
    sources.append(catalog)
    train = read_rows(data / "train.inputs.jsonl")
    train_labels = read_rows(data / "train.labels.jsonl")
    inputs = read_rows(data / "dev.inputs.jsonl")
    pools = {r["case_id"]: r["ranked_movie_ids"] for r in read_rows(data / "dev.candidates.jsonl")}
    from experiments.scripts.data.build_redial_benchmark import load_catalog
    from experiments.consensus_recommendation.protocol import bundle
    from experiments.paired_deliberation.contracts_v6 import initial_request
    from experiments.paired_deliberation.budget_v6 import AttemptClient
    from experiments.paired_deliberation.training_evidence_view import attach
    _, _, movies = load_catalog(catalog)
    backend = TrainingEvidence(train, train_labels, movies)
    entries, all_sizes = {}, []
    nonzero_context = nonzero_assoc = reproduced = changed_top5 = 0
    for inp in inputs:
        pool = pools[inp["case_id"]]
        trace = backend.explain(inp, pool)
        view = render(trace)
        stem = inp["case_id"].replace(":", "_") + ".json"
        create(output / "traces" / stem, trace)
        create(output / "views" / stem, view)
        dummy = {"ranked_movie_ids": pool}
        evidence = bundle(inp, dummy, movies)
        request = attach(initial_request(evidence, "current_request"), view, "independent")
        size = AttemptClient.upper(request)
        all_sizes.append(size)
        entries[inp["case_id"]] = {"input_hash": fingerprint(inp), "view_hash": fingerprint(view), "trace_hash": fingerprint(trace),
                                  "initial_request_utf8_upper": size}
        reproduced += trace["reproduced_frozen_pool"]
        nonzero_context += any(row["context_transfer_score"] > 0 for row in trace["candidate_signals"])
        nonzero_assoc += any(row["association_score"] > 0 for row in trace["candidate_signals"])
        reranked = sorted(trace["candidate_signals"], key=lambda row: (-row["context_transfer_score"], row["movie_id"]))[:5]
        changed_top5 += [r["movie_id"] for r in reranked] != pool[:5]
    import sklearn, scipy
    manifest = {"version": "training-evidence-explicit-v1", "training_inputs": len(train), "dev_inputs": len(inputs),
        "cases": entries, "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), Path(__file__).with_name("training_evidence_view.py"),
            ROOT / "experiments/consensus_recommendation/data.py", ROOT / "experiments/recommendation_v2/audit.py"]},
        "environment": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "training_labels_used": True, "current_dev_labels_read": False, "reserved_files_read": False,
        "candidate_pools_changed": False, "five_method_permission_policy": "identical evidence view, no role-dependent withholding",
        "serving_container": "separate training_prediction_evidence; never first-party dialogue observations",
        "collection_started": False, "new_api_calls": 0,
        "audit": {"retrieval_pools_reproduced": reproduced, "cases_with_context_signal": nonzero_context,
                  "cases_with_association_signal": nonzero_assoc, "context_score_top5_order_differs": changed_top5,
                  "maximum_augmented_initial_utf8_upper": max(all_sizes),
                  "augmented_initial_over_existing_64000_cap": sum(n > 64000 for n in all_sizes)},
        "limits": ["explicit evidence already contributed to candidate retrieval; not a new learning algorithm",
                   "ranking changes without dev labels do not establish usefulness or recommendation gain",
                   "new actor prompts/effect protocol must be frozen before model collection"]}
    manifest["integrity"] = fingerprint(manifest)
    create(output / "manifest.json", manifest)
    return {key: value for key, value in manifest.items() if key not in ("cases", "source_sha256", "code_sha256")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(export(parser.parse_args().output), ensure_ascii=False, indent=2))
