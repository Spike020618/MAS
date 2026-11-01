"""Standard-library serving view: historical predictive evidence, not user facts."""
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.contracts_v6 import METHODS
from experiments.paired_deliberation.verified_state import canonical

COLUMNS = ("movie_id", "context_transfer_score", "association_score", "training_positive_count",
           "lexical_rank", "context_transfer_rank", "association_rank", "popularity_rank",
           "fused_rrf_score", "visible_context_support_mass", "omitted_context_support_mass")
ACTOR_KINDS = {"initial_v6", "whole_review_v6", "patch_v6", "select_v6", "ranking_v6"}


def render(trace):
    def visible(value): return round(value, 8) if type(value) is float else value
    return {"kind": "historical_training_prediction_evidence", "case_id": trace["case_id"],
        "current_case_labels_available": False, "historical_training_outcomes_used": True,
        "candidate_columns": list(COLUMNS),
        "candidate_rows": [[visible(row[name]) for name in COLUMNS] for row in trace["candidate_signals"]],
        "historical_examples": copy.deepcopy(trace["visible_neighbors"]),
        "neighborhood_size": trace["positive_similarity_neighbors"], "training_source_hash": trace["training_source_hash"],
        "display_precision_decimals": 8,
        "interpretation": ["These are other training conversations and historical outcome targets, not the current seeker's statements.",
            "Scores and similarities are not probabilities of liking or factual current preferences.",
            "Zero/absent support is not dislike. Context mentions are not confirmed likes or seen films.",
            "Only a bounded neighbor display is shown; omitted_support_mass records unshown contribution.",
            "Candidate IDs and order are unchanged. Full precision and provenance are in the trace artifact."]}


def for_method(view, method):
    if method not in METHODS: raise ValueError("unknown core method")
    return copy.deepcopy(view)  # No role-conditioned withholding or reweighting.


def attach(request, view, method):
    if request["kind"] not in ACTOR_KINDS:
        raise ValueError("training predictions cannot enter claim parsing or first-party semantic verification")
    if view.get("kind") != "historical_training_prediction_evidence" or view.get("current_case_labels_available") is not False:
        raise ValueError("untrusted training evidence origin")
    result = copy.deepcopy(request)
    payload = json.loads(result["messages"][1]["content"])
    offered = for_method(view, method)
    if "training_prediction_evidence" in payload and payload["training_prediction_evidence"] != offered:
        raise ValueError("different evidence already attached")
    payload["training_prediction_evidence"] = offered
    result["messages"][1]["content"] = canonical(payload)
    result["messages"][0]["content"] += (
        " Training prediction evidence describes historical OTHER conversations. Use it as a recommendation "
        "signal, never as the current seeker's explicit preference or a calibrated probability.")
    return result


def load(root, serving_input, candidate_ids, method):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["integrity"] != fingerprint({k: v for k, v in manifest.items() if k != "integrity"}):
        raise ValueError("training evidence manifest changed")
    cid = serving_input["case_id"]
    if cid not in manifest["cases"]: raise ValueError("case not exported")
    entry = manifest["cases"][cid]
    if fingerprint(serving_input) != entry["input_hash"]: raise ValueError("serving prefix changed")
    path = root / "views" / (cid.replace(":", "_") + ".json")
    view = json.loads(path.read_text())
    if fingerprint(view) != entry["view_hash"]: raise ValueError("training evidence view changed")
    if [row[0] for row in view["candidate_rows"]] != list(candidate_ids): raise ValueError("candidate pool/order changed")
    return for_method(view, method)
