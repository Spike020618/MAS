"""Training-only collaborative features. No development/test labels are read."""
from __future__ import annotations

import csv
import json
import time
from array import array
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from .common import ROOT, OUTPUT, PROTOCOL, sha, write_once, user_bucket

CONTROL = Path(__file__).with_name("controls_protocol.json")
CF_OUTPUT = OUTPUT / "controls/collaborative"


def train():
    cfg = json.loads(PROTOCOL.read_text())
    ctrl = json.loads(CONTROL.read_text())
    data = OUTPUT / "data"
    priors_path = data / "train_priors.json"
    dependency = {"protocol": sha(PROTOCOL), "controls": sha(CONTROL),
                  "code": sha(Path(__file__)), "common": sha(Path(__file__).with_name("common.py")),
                  "priors": sha(priors_path), "data_manifest": sha(data / "manifest.json")}
    final = CF_OUTPUT / "manifest.json"
    if final.exists():
        result = json.loads(final.read_text())
        if result["dependencies"] != dependency or sha(CF_OUTPUT / "features.npz") != result["features_sha256"]:
            raise ValueError("CF dependency/artifact changed")
        print("CF features verified", flush=True)
        return
    priors = json.loads(priors_path.read_text())
    ids = sorted(priors["items"])
    positions = {mid: i for i, mid in enumerate(ids)}
    prior = {mid: priors["items"][mid]["prior"] for mid in ids}
    matrix_path = CF_OUTPUT / "training_matrix.npz"
    matrix_manifest = CF_OUTPUT / "matrix_manifest.json"
    write_once(CF_OUTPUT / "provenance.json", dependency)
    started = time.perf_counter()
    if matrix_manifest.exists():
        from scipy.sparse import load_npz
        cached = json.loads(matrix_manifest.read_text())
        if cached["dependencies"] != dependency or sha(matrix_path) != cached["sha256"]:
            raise ValueError("CF matrix checkpoint changed")
        matrix = load_npz(matrix_path)
        training_users = cached["users"]
    else:
        row_indices, cols, values = array("i"), array("i"), array("d")
        user_rows = {}
        print("START CF training-user pre-cutoff matrix", flush=True)
        source = ROOT / "experiments/data/raw/movielens/ml-25m/ratings.csv"
        manifest = json.loads((data / "manifest.json").read_text())
        if sha(source) != manifest["dependencies"]["ratings"]:
            raise ValueError("ratings source changed")
        previous, allowed = None, False
        with source.open(newline="", encoding="utf-8") as f:
            for record in csv.DictReader(f):
                uid, mid = record["userId"], record["movieId"]
                if uid != previous:
                    allowed = user_bucket(uid) >= 2
                    previous = uid
                if not allowed or mid not in positions or int(record["timestamp"]) >= cfg["global_cutoff_timestamp"]:
                    continue
                if uid not in user_rows:
                    user_rows[uid] = len(user_rows)
                row_indices.append(user_rows[uid])
                cols.append(positions[mid])
                values.append(float(record["rating"]) - prior[mid])
        matrix = coo_matrix((np.frombuffer(values, dtype=np.float64),
                             (np.frombuffer(row_indices, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
                            shape=(len(user_rows), len(ids))).tocsr()
        from scipy.sparse import save_npz
        save_npz(matrix_path, matrix)
        training_users = len(user_rows)
        write_once(matrix_manifest, {"dependencies": dependency, "sha256": sha(matrix_path),
                                     "users": training_users, "observations": matrix.nnz})
        print(f"DONE matrix {matrix.shape}, nnz={matrix.nnz}", flush=True)
    rng = np.random.default_rng(ctrl["seed"])
    width = ctrl["latent_dimensions"] + ctrl["oversampling"]
    q, _ = np.linalg.qr(matrix @ rng.normal(size=(len(ids), width)), mode="reduced")
    for i in range(ctrl["power_iterations"]):
        z, _ = np.linalg.qr(matrix.T @ q, mode="reduced")
        q, _ = np.linalg.qr(matrix @ z, mode="reduced")
        print(f"DONE CF power iteration {i+1}", flush=True)
    small = (matrix.T @ q).T
    _, singular, vt = np.linalg.svd(small, full_matrices=False)
    features = vt[:ctrl["latent_dimensions"]].T
    features /= np.maximum(np.linalg.norm(features, axis=1), 1e-12)[:, None]
    if not np.isfinite(features).all():
        raise ValueError("nonfinite CF features")
    path = CF_OUTPUT / "features.npz"
    np.savez_compressed(path, ids=np.array(ids), features=features)
    result = {"dependencies": dependency, "features_sha256": sha(path),
              "training_users": training_users, "training_observations": matrix.nnz,
              "items": len(ids), "dimensions": features.shape[1],
              "leading_singular_values": singular[:ctrl["latent_dimensions"]].tolist(),
              "elapsed_seconds": time.perf_counter() - started,
              "dev_or_test_users_in_training": False, "future_ratings_used": False,
              "model_api_calls": 0}
    write_once(final, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    train()
