"""Evidence-relative controlled semantics; not human-labelled real dialogue."""
import argparse
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint


def build():
    rows = []
    contexts = [("Drama", "Comedy", "Film A", "Film B"),
                ("Comedy", "Horror", "Film C", "Film D"),
                ("Animation", "Thriller", "Film E", "Film F"),
                ("Documentary", "Romance", "Film G", "Film H")]
    for g, h, a, b in contexts:
        categories = {
            "current_preference": (
                [f"For tonight I want {g}, and I do not want {h}."],
                [(f"The seeker currently requests {g}.", "supported"),
                 (f"The seeker currently requests {h}.", "refuted"),
                 (f"The seeker has always enjoyed {g}.", "insufficient")]),
            "temporal_scope": (
                [f"I used to like {g}, but I do not like {g} now."],
                [(f"The seeker liked {g} in the past.", "supported"),
                 (f"The seeker likes {g} now.", "refuted"),
                 (f"The seeker likes {h} now.", "insufficient")]),
            "subject_attribution": (
                [f"My sister likes {g}. I dislike {g}."],
                [(f"The seeker's sister likes {g}.", "supported"),
                 (f"The seeker likes {g}.", "refuted"),
                 (f"The seeker's sister likes {h}.", "insufficient")]),
            "negation": (
                [f"I have not watched {a}. I have watched {b}."],
                [(f"The seeker has not watched {a}.", "supported"),
                 (f"The seeker has watched {a}.", "refuted"),
                 (f"The seeker enjoyed {b}.", "insufficient")]),
            "partial_support": (
                [f"I enjoyed {a}. I have never watched {b}."],
                [(f"The seeker enjoyed {a}.", "supported"),
                 (f"The seeker has watched {b}.", "refuted"),
                 (f"The seeker enjoyed {a} and enjoys {g} generally.", "insufficient")]),
            "conflicting_evidence": (
                [f"At the same point in time, one seeker statement says: I want {g} tonight.",
                 f"At that same point in time, another seeker statement says: I do not want {g} tonight. No ordering or retraction is available."],
                [(f"The evidence contains conflicting statements about wanting {g} tonight.", "supported"),
                 (f"Both seeker statements agree that the seeker wants {g} tonight.", "refuted"),
                 (f"The seeker's final resolved preference is to watch {g} tonight.", "insufficient")]),
            "mention_vs_experience": (
                [f"Can you tell me about {a}? I have not watched it."],
                [(f"The seeker asked for information about {a}.", "supported"),
                 (f"The seeker has watched {a}.", "refuted"),
                 (f"The seeker liked {a}.", "insufficient")]),
        }
        for category, (texts, claims) in categories.items():
            evidence = {"message:%d" % (i + 1): {"source_id": "message:%d" % (i + 1),
                         "role": "seeker", "text": text} for i, text in enumerate(texts)}
            for text, label in claims:
                payload = {"claim": {"claim_id": "c1", "text": text, "source_ids": list(evidence),
                                     "candidate_ids": [], "stance": "support"}, "evidence": evidence}
                rows.append({"id": "nl_" + fingerprint(payload)[:16], "input": payload,
                             "label": label, "category": category, "context_group": fingerprint(texts)[:12]})
    return sorted(rows, key=lambda row: row["id"])


def export(output):
    rows = build()
    output = Path(output)
    inputs = [{"id": row["id"], **row["input"]} for row in rows]
    labels = [{key: row[key] for key in ("id", "label", "category", "context_group")} for row in rows]
    create(output / "inputs.json", inputs)
    create(output / "labels.json", labels)
    manifest = {"version": "v51-controlled-semantics-84", "items": len(rows),
                "label_counts": dict(Counter(row["label"] for row in rows)),
                "categories": dict(Counter(row["category"] for row in rows)),
                "context_groups": len({row["context_group"] for row in rows}),
                "inputs_hash": fingerprint(inputs), "labels_available_to_execution": False,
                "scope": "synthetic evidence-relative semantic calibration; not real-dialogue accuracy",
                "gates": {"accuracy_min": .90, "recall_each_class_min": .80,
                          "accuracy_each_category_min": .75, "false_supported_max": 0},
                "new_api_calls": 0}
    create(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(export(parser.parse_args().output))
