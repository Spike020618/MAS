"""Local transport replay only. Never contact a provider or read labels.

Artificial rankings and zero usage fields are marked OFFLINE_FIXTURE. They are
not recommendation outcomes and must not be supplied to the production scorer.
"""
import json
import platform
import threading
from importlib.metadata import version
from unittest.mock import patch

from experiments.information_roles_pilot.core import OUT, save, fingerprint, validate_rank, sha
from experiments.information_roles_pilot.run import Collector, verify_plan


def main():
    plan = verify_plan()
    cases = json.loads((OUT / "cases.json").read_text())
    replay = OUT / "offline_transport_replay"
    def collector():
        obj = Collector.__new__(Collector)
        obj.plan = {"kind": "OFFLINE_FIXTURE", "production_plan_hash": fingerprint(plan)}
        obj.key, obj.endpoint = "OFFLINE_FIXTURE_NOT_A_KEY", "https://api.deepseek.com/chat/completions"
        obj.lock, obj.halt, obj.models = threading.Lock(), threading.Event(), set()
        obj.new_calls = obj.cached_calls = 0
        return obj
    def response(*args, **kwargs):
        payload = json.loads(kwargs["json"]["messages"][-1]["content"].split("\n", 1)[1])
        ranking = [m["movie_id"] for m in payload["candidates"]]
        raw = {"model": "OFFLINE_FIXTURE", "simulation": True,
               "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
               "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"ranking": ranking})}}]}
        return type("FixtureResponse", (), {"status_code": 200, "json": lambda self: raw})()
    with patch("experiments.information_roles_pilot.run.OUT", replay), patch("requests.post", side_effect=response) as post:
        first = collector()
        outputs = [first.run_case(case) for case in cases]
        simulated_first_requests = post.call_count
        second = collector()
        restored = [second.run_case(case) for case in cases]
        if outputs != restored or post.call_count != simulated_first_requests:
            raise ValueError("non-identical resume or unexpected new fixture request")
    if simulated_first_requests not in (0, 32):
        raise ValueError("unexpected fixture count")
    for case, output in zip(cases, outputs):
        for ranking in output["methods"].values():
            validate_rank(ranking, case["candidate_ids"])
        if any(s["model_returned"] != "OFFLINE_FIXTURE" or s["error"] for s in output["slots"].values()):
            raise ValueError("fixture identity/parse failure")
    source = json.loads((OUT / "data_audit.json").read_text())
    receipt = {"kind": "offline_engineering_acceptance_not_effectiveness",
               "production_plan_sha256": sha(OUT / "plan.json"),
               "source_raw_ratings_rows": source["raw_source"]["scanned_ratings"],
               "source_official_md5": source["raw_source"]["zip_md5"],
               "real_provider_calls": 0, "replayed_cases": len(outputs),
               "simulated_slots_in_saved_replay": sum(len(o["slots"]) for o in outputs),
               "second_pass_new_transport_requests": 0,
               "offline_labels_read_by_replay": False, "production_score_generated": False,
               "paid_collection": "blocked_before_execution_by_safety_review_pending_explicit_data_disclosure_consent",
               "runtime": {"python": platform.python_version(),
                           "packages": {p: version(p) for p in ("numpy", "scipy", "requests", "python-dotenv")}},
               "tests_observed_passed": {"new_pilot_unit_tests": 9, "source_and_design_tests": 12}}
    save(OUT / "engineering_acceptance.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
