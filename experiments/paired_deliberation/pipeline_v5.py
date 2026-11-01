"""V5: registered target -> patch -> evidence verification -> scoped update -> output."""
import copy
import time

from experiments.consensus_recommendation.protocol import validate
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.verified_state import ClaimState, EvidenceVerifier


def source_map(evidence):
    rows = list(evidence["observations"]) + list(evidence["candidates_in_shared_retrieval_order"])
    result = {}
    for row in rows:
        if row["source_id"] in result:
            raise ValueError("duplicate evidence source ID")
        result[row["source_id"]] = copy.deepcopy(row)
    return result


def run_case(evidence, client, initial=None, use_model_verifier=False):
    started = time.monotonic()
    builder = RequestsV5(evidence)
    imported_initial = initial is not None
    owners = ["current_request", "history_applicability", "movie_fit"]
    case = evidence["case_id"].replace(":", "_")
    if initial is None:
        initial = {owner: client.complete(case + "_initial_" + owner, builder.initial(owner))["report"]
                   for owner in owners}
    else:
        # Offline fixtures only: importing initial reports is disclosed explicitly.
        initial = copy.deepcopy(initial)
    if set(initial) != set(owners):
        raise ValueError("initial reports must cover the declared owner set")
    for report in initial.values():
        validate(report, evidence)
    verification_index = [0]

    def judge(payload):
        index = verification_index[0]
        verification_index[0] += 1
        record = client.complete(case + "_semantic_%03d" % index, builder.verification(payload))
        report = copy.deepcopy(record["report"])
        report["usage"] = record["usage"]
        return report

    state = ClaimState(initial, EvidenceVerifier(source_map(evidence), judge if use_model_verifier else None),
                       invalid_verification="defer")
    # Fixed label-free routing covers every registered claim once. Reviewer and
    # owner are separate identities; cross-agent challenges are executable.
    for index, target in enumerate(sorted(state.claims)):
        owner = state.claims[target]["owner"]
        reviewer = owners[(owners.index(owner) + 1) % len(owners)]
        record = client.complete(case + "_patch_%03d" % index, builder.review(reviewer, state, target))
        state.apply(reviewer, record["report"])
    final_request = builder.aggregation(state)
    final = client.complete(case + "_aggregate", final_request)["report"]
    pool = {row["movie_id"] for row in evidence["candidates_in_shared_retrieval_order"]}
    if set(final) != {"ranking"} or not isinstance(final["ranking"], list):
        raise ValueError("final output must contain ranking only")
    ranking = final["ranking"]
    if len(ranking) != 5 or len(set(ranking)) != 5 or not set(ranking) <= pool:
        raise ValueError("invalid final candidate ranking")
    return {"version": "claim-updates-v5", "case_id": evidence["case_id"],
            "final": final, "claim_state": state.aggregation_payload(), "events": state.history,
            "usage": client.summary(), "end_to_end_ms": int((time.monotonic() - started) * 1000),
            "initial_generation_imported": imported_initial,
            "model_verifier_enabled": use_model_verifier,
            "semantic_judgements_are_proxy": use_model_verifier,
            "evaluation_labels_available": False}
