"""Bounded provider runner. No label loading, retries, or import-time calls."""
import argparse
import concurrent.futures
import json
import os
import threading
import time
from datetime import datetime, timezone

from .core import CFG, ROOT, OUT, sha, save, fingerprint, messages, validate_rank, rrf, code_dependencies


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_plan():
    plan = json.loads((OUT / "plan.json").read_text())
    if plan["config"] != CFG or plan["code"] != code_dependencies():
        raise ValueError("code/config changed since preparation")
    if sha(OUT / "cases.json") != plan["cases_sha256"] or sha(OUT / "data_audit.json") != plan["audit_sha256"]:
        raise ValueError("case/source audit integrity mismatch")
    return plan


class Collector:
    def __init__(self, plan):
        self.plan = plan
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        self.key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.key:
            raise ValueError("DEEPSEEK_API_KEY missing")
        # Destination is fixed; no arbitrary endpoint receives source payloads.
        self.endpoint = "https://api.deepseek.com/chat/completions"
        self.lock = threading.Lock()
        self.halt = threading.Event()
        self.models = set()
        self.new_calls = 0
        self.cached_calls = 0

    def call(self, case, role, previous=None):
        if self.halt.is_set():
            raise RuntimeError("collection halted by another failed request")
        try:
            return self._call(case, role, previous)
        except BaseException:
            # An already in-flight request may finish; no new request starts.
            self.halt.set()
            raise

    def _call(self, case, role, previous=None):
        prompt = messages(case, role, previous)
        request = {"model": CFG["model"], "temperature": CFG["temperature"],
                   "max_tokens": CFG["max_tokens"], "messages": prompt,
                   "response_format": {"type": "json_object"}, "stream": False}
        dependency = fingerprint({"request": request, "plan": self.plan})
        path = OUT / "slots" / case["case_id"] / (role + ".json")
        intent = path.with_suffix(".intent.json")
        if path.exists():
            envelope = json.loads(path.read_text())
            if envelope["dependency"] != dependency or envelope["integrity"] != fingerprint(envelope["record"]):
                raise ValueError("cached slot integrity mismatch")
            record = envelope["record"]
            with self.lock:
                self.cached_calls += 1
        else:
            if intent.exists():
                raise RuntimeError("unfinished intent: provider billing unknown; no automatic retry")
            save(intent, {"dependency": dependency, "start_utc": now(), "request": request})
            import requests
            start = time.perf_counter()
            try:
                response = requests.post(self.endpoint, headers={"Authorization": "Bearer " + self.key},
                                         json=request, timeout=(15, 150))
            except requests.RequestException:
                raise RuntimeError("provider network failure; intent retained, billing unknown") from None
            elapsed = time.perf_counter() - start
            if response.status_code != 200:
                save(path, {"dependency": dependency, "record": {"http_status": response.status_code,
                     "elapsed_seconds": elapsed, "usage": None, "billing": "unknown"},
                     "integrity": fingerprint({"http_status": response.status_code,
                     "elapsed_seconds": elapsed, "usage": None, "billing": "unknown"})})
                raise RuntimeError("provider HTTP error " + str(response.status_code))
            raw = response.json()
            record = {"raw": raw, "elapsed_seconds": elapsed, "end_utc": now(),
                      "request_hash": fingerprint(request)}
            save(path, {"dependency": dependency, "record": record, "integrity": fingerprint(record)})
            with self.lock:
                self.new_calls += 1
        if "raw" not in record:
            raise RuntimeError("previous provider error; halted")
        raw = record["raw"]
        usage = raw.get("usage")
        if not isinstance(usage, dict) or any(not isinstance(usage.get(k), int) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            raise ValueError("unknown provider token usage")
        if usage["completion_tokens"] > CFG["max_tokens"]:
            raise ValueError("provider output cap violated")
        model = raw.get("model")
        if not model:
            raise ValueError("missing returned model")
        with self.lock:
            self.models.add(model)
            if len(self.models) != 1:
                raise ValueError("returned model changed during collection")
        error, ranking = None, None
        try:
            choice = raw["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("non-stop finish")
            obj = json.loads(choice["message"]["content"])
            if set(obj) != {"ranking"}:
                raise ValueError("unexpected schema")
            ranking = validate_rank(obj["ranking"], case["candidate_ids"])
        except (ValueError, KeyError, IndexError, TypeError):
            # Do not repair a malformed ranking or silently fill missing items.
            error = "invalid_complete_permutation_or_schema"
        return {"ranking": ranking, "error": error, "usage": usage, "elapsed_seconds": record["elapsed_seconds"],
                "model_returned": model, "slot_sha256": sha(path),
                "request_bytes": sum(len(m["content"].encode()) for m in prompt)}

    def run_case(self, case):
        if self.halt.is_set():
            raise RuntimeError("collection halted")
        destination = OUT / "outputs" / (case["case_id"] + ".json")
        expected = fingerprint({"case": case, "plan": self.plan})
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved["dependency"] != expected or saved["integrity"] != fingerprint(saved["record"]):
                raise ValueError("changed case checkpoint")
            for role, result in saved["record"]["slots"].items():
                if sha(OUT / "slots" / case["case_id"] / (role + ".json")) != result["slot_sha256"]:
                    raise ValueError("case references changed slot")
            return saved["record"]
        started = time.perf_counter()
        slots = {role: self.call(case, role) for role in ("P", "C", "U1")}
        slots["U2"] = self.call(case, "U2", slots["U1"]["ranking"] or rrf(case["P0"], case["C0"]))
        methods = {"P0": case["P0"], "C0": case["C0"], "M00": rrf(case["P0"], case["C0"])}
        # Explicit baseline fallback is an outcome, not a valid model report.
        p = slots["P"]["ranking"] or case["P0"]
        c = slots["C"]["ranking"] or case["C0"]
        methods.update(M10=rrf(p, case["C0"]), M01=rrf(case["P0"], c), M11=rrf(p, c),
                       U1=slots["U1"]["ranking"] or methods["M00"],
                       U2=slots["U2"]["ranking"] or slots["U1"]["ranking"] or methods["M00"])
        for ranking in methods.values():
            validate_rank(ranking, case["candidate_ids"])
        record = {"case_id": case["case_id"], "methods": methods, "slots": slots,
                  "case_wall_seconds_including_all_controls": time.perf_counter() - started}
        save(destination, {"dependency": expected, "record": record, "integrity": fingerprint(record)})
        print(case["case_id"] + " completed", flush=True)
        return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    plan = verify_plan()
    cases = json.loads((OUT / "cases.json").read_text())
    if not args.execute_authorized:
        print(json.dumps({"cases": len(cases), "max_new_calls": len(cases) * CFG["calls_per_case"], "executed": False}))
        return
    collector = Collector(plan)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(collector.run_case, cases))
    # Per-invocation runtime is separate from immutable saved per-request times.
    print(json.dumps({"completed": len(cases), "new_calls_this_invocation": collector.new_calls,
                      "reused_slots_this_invocation": collector.cached_calls,
                      "wall_seconds_this_invocation": time.perf_counter() - start,
                      "returned_models_this_invocation": sorted(collector.models)}))


if __name__ == "__main__":
    main()
