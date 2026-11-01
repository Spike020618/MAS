"""Label-blind case selection, shared candidates, raw-source reconstruction."""
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict

import numpy as np

from experiments.slate_choice.baselines import ridge
from experiments.slate_choice.common import sample_key, user_bucket
from .core import CFG, ROOT, OUT, sha, rows, save, fingerprint, code_dependencies, messages

LEGACY = ROOT / "experiments/results/slate_choice_v1"


def source_audit(selected):
    manifest = json.loads((LEGACY / "data/manifest.json").read_text())
    # Preserve the previously damaged archive. This separately downloaded file
    # must pass the official checksum before any source rows are used.
    archive = ROOT / "experiments/data/raw/movielens/ml-25m.official-revalidation.zip"
    archive_hash = sha(archive)
    expected_md5 = (archive.parent / "ml-25m.zip.md5").read_text().split()[0]
    md5 = hashlib.md5()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
    if md5.hexdigest() != expected_md5:
        raise ValueError("source ZIP MD5 mismatch")
    collected = defaultdict(list)
    raw_hash = hashlib.sha256()
    count = 0
    with zipfile.ZipFile(archive) as z:
        with z.open("ml-25m/ratings.csv") as handle:
            header = handle.readline()
            raw_hash.update(header)
            if header.strip() != b"userId,movieId,rating,timestamp":
                raise ValueError("unexpected ratings header")
            for line in handle:
                raw_hash.update(line)
                count += 1
                uid = line.split(b",", 1)[0].decode()
                if uid in selected:
                    user, mid, rating, timestamp = line.decode().strip().split(",")
                    collected[user].append({"movie_id": mid, "rating": float(rating), "timestamp": int(timestamp)})
        with z.open("ml-25m/movies.csv") as handle:
            movie_bytes = handle.read()
    if raw_hash.hexdigest() != manifest["dependencies"]["ratings"]:
        raise ValueError("raw ratings not bound to historical data build")
    if hashlib.sha256(movie_bytes).hexdigest() != manifest["dependencies"]["movies"]:
        raise ValueError("raw movies mismatch")
    return collected, {"archive_path": str(archive.relative_to(ROOT)),
                       "zip_sha256": archive_hash, "zip_md5": md5.hexdigest(),
                       "ratings_sha256": raw_hash.hexdigest(), "scanned_ratings": count,
                       "movies_sha256": hashlib.sha256(movie_bytes).hexdigest()}


def point_scores(x, base, history, positions, penalty):
    idx = [positions[r["movie_id"]] for r in history]
    beta = ridge(x[idx], np.array([r["rating"] for r in history]) - base[idx], penalty)
    return np.clip(base + np.einsum("ij,j->i", x, beta, optimize=False), 0.5, 5)


def main():
    inputs_path = LEGACY / "data/dev.inputs.jsonl"
    # Select before opening any evaluation labels, independent of outcomes.
    inputs = sorted(rows(inputs_path), key=lambda x: sample_key(x["user_id"]))[:CFG["cases"]]
    selected = {x["user_id"] for x in inputs}
    if len(selected) != CFG["cases"] or any(user_bucket(uid) != 0 for uid in selected):
        raise ValueError("not a development-only independent-user batch")
    manifest = json.loads((LEGACY / "data/manifest.json").read_text())
    priors_path = LEGACY / "data/train_priors.json"
    for p in (inputs_path, priors_path):
        if sha(p) != manifest["artifacts"][p.name]:
            raise ValueError("historical input artifact changed")
    items = json.loads(priors_path.read_text())["items"]
    source, raw_audit = source_audit(selected)
    ids = sorted(items)
    positions = {mid: i for i, mid in enumerate(ids)}
    base = np.array([items[mid]["prior"] for mid in ids])
    genres = sorted({g for v in items.values() for g in v["genres"]})
    gf = np.array([[float(g in items[mid]["genres"]) for g in genres] for mid in ids])
    gx = np.column_stack((np.ones(len(ids)), gf / np.maximum(1, np.linalg.norm(gf, axis=1))[:, None]))
    cfdir = LEGACY / "controls/collaborative"
    cfmanifest = json.loads((cfdir / "manifest.json").read_text())
    if sha(cfdir / "features.npz") != cfmanifest["features_sha256"]:
        raise ValueError("CF features changed")
    if cfmanifest["dependencies"]["priors"] != sha(priors_path):
        raise ValueError("CF training prior mismatch")
    if cfmanifest["dependencies"]["data_manifest"] != sha(LEGACY / "data/manifest.json"):
        raise ValueError("CF training dataset mismatch")
    for name, path in {"code": ROOT / "experiments/slate_choice/train_cf.py",
                       "common": ROOT / "experiments/slate_choice/common.py",
                       "protocol": ROOT / "experiments/slate_choice/protocol.json",
                       "controls": ROOT / "experiments/slate_choice/controls_protocol.json"}.items():
        if sha(path) != cfmanifest["dependencies"][name]:
            raise ValueError("CF training source code/config changed: " + name)
    if cfmanifest["future_ratings_used"] or cfmanifest["dev_or_test_users_in_training"]:
        raise ValueError("CF training contamination")
    if sha(cfdir / "training_matrix.npz") != json.loads((cfdir / "matrix_manifest.json").read_text())["sha256"]:
        raise ValueError("CF training matrix changed")
    with np.load(cfdir / "features.npz", allow_pickle=False) as data:
        if data["ids"].tolist() != ids:
            raise ValueError("CF item alignment mismatch")
        features = data["features"].copy()
    cx = np.column_stack((np.ones(len(ids)), features))
    cfg = json.loads((ROOT / "experiments/slate_choice/protocol.json").read_text())
    cases, labels = [], []
    for inp in inputs:
        uid = inp["user_id"]
        raw = source[uid]
        past = sorted([r for r in raw if r["timestamp"] < inp["cutoff"]], key=lambda r: (r["timestamp"], r["movie_id"]))
        history40 = [r for r in past if r["movie_id"] in items][-40:]
        if history40 != inp["history"] or {r["movie_id"] for r in past} != set(inp["seen_movie_ids"]):
            raise ValueError("raw prefix reconstruction mismatch")
        history = history40[-CFG["history"]:]
        ps = point_scores(gx, base, history, positions, cfg["ridge_lambda"])
        cs = point_scores(cx, base, history, positions, cfg["ridge_lambda"])
        seen = set(inp["seen_movie_ids"])
        available = [j for j, mid in enumerate(ids) if mid not in seen]
        po = sorted(available, key=lambda j: (-ps[j], ids[j]))
        co = sorted(available, key=lambda j: (-cs[j], ids[j]))
        pool = set(po[:CFG["candidates_per_base"]] + co[:CFG["candidates_per_base"]])
        pool = sorted(pool, key=lambda j: hashlib.sha256((uid + ":" + ids[j]).encode()).hexdigest())
        candidate_ids = [ids[j] for j in pool]
        p0 = [ids[j] for j in po if j in pool]
        c0 = [ids[j] for j in co if j in pool]
        catalog = [{"movie_id": ids[j], "title": items[ids[j]]["title"], "genres": items[ids[j]]["genres"]} for j in pool]
        history_packet = [{"record_id": "h" + str(n), "title": items[r["movie_id"]]["title"],
                           "genres": items[r["movie_id"]]["genres"], "observed_rating": r["rating"]}
                          for n, r in enumerate(history)]
        cf_packet = []
        hi = [positions[r["movie_id"]] for r in history]
        for j in pool:
            similarities = np.einsum("ij,j->i", features[hi], features[j], optimize=False)
            near = sorted(range(len(hi)), key=lambda k: (-similarities[k], k))[:2]
            cf_packet.append({"movie_id": ids[j], "predicted_rating": round(float(cs[j]), 5),
                              "nearest_observed": [{"title": history_packet[k]["title"],
                                                    "observed_rating": history[k]["rating"],
                                                    "latent_similarity": round(float(similarities[k]), 5)} for k in near]})
        case = {"case_id": "ml-dev-" + uid, "user_id": uid, "candidate_ids": candidate_ids,
                "catalog": catalog, "history_packet": history_packet, "cf_packet": cf_packet,
                "P0": p0, "C0": c0, "cutoff": inp["cutoff"],
                "provenance": {"input_history": fingerprint(history), "source": "public MovieLens25M"}}
        for role in ("P", "C", "U1", "U2"):
            messages(case, role, candidate_ids if role == "U2" else None)
        cases.append(case)
        # Future outcomes are separated only AFTER the case and pool are fixed.
        relevant = sorted({r["movie_id"] for r in raw if r["timestamp"] >= inp["cutoff"] and r["rating"] >= 4})
        labels.append({"case_id": case["case_id"], "positive_ids": relevant})
    save(OUT / "cases.json", cases)
    save(OUT / "offline_labels.json", labels)
    public_payloads = [messages(case, role) for case in cases for role in ("P", "C", "U1")]
    payload_text = json.dumps(public_payloads)
    if any(key in payload_text for key in ("positive_ids", "future_ratings", "user_id", "DEEPSEEK_API_KEY")):
        raise ValueError("forbidden payload field")
    if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", payload_text):
        raise ValueError("email-like string in payload")
    save(OUT / "data_audit.json", {"raw_source": raw_audit, "cases": len(cases),
          "selection": "first 8 development users by pre-existing hash, no outcome filter",
          "history_reconstructed": True, "future_in_history": False, "reserved_test_opened": False,
          "CF_features": sha(cfdir / "features.npz"), "public_payload_screened": True,
          "limitation": "CF training provenance inherited and hash-checked, not independently retrained; static metadata snapshot"})
    save(OUT / "plan.json", {"config": CFG, "code": code_dependencies(), "cases_sha256": sha(OUT / "cases.json"),
          "labels_sha256": sha(OUT / "offline_labels.json"), "audit_sha256": sha(OUT / "data_audit.json"),
          "methods": ["P0", "C0", "M00", "M10", "M01", "M11", "U1", "U2"],
          "scientific_status": "engineering development; no fitted router, no game, no AgentCF performance reproduction"})
    print(json.dumps({"prepared_cases": len(cases), "candidate_sizes": [len(c["candidate_ids"]) for c in cases]}))


if __name__ == "__main__":
    main()
