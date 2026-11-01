"""Bounded, restartable DeepSeek collection. No evaluation-label imports.

prepare-only is default; --execute-authorized executes an accepted collection plan.
Network failures of unknown billing state stop rather than silently repeating.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .common import ROOT, OUTPUT, code_hashes, create, fingerprint, read_rows, sha, verify_artifacts
from .protocol import ROLES, FOCUS, bundle, config, disputes, collect_parse, prompt


def utc():
    return datetime.now(timezone.utc).isoformat()


class Budget:
    def __init__(self):
        self.input_upper = self.output = 0

    def request(self, messages):
        upper = sum(len(m["content"].encode("utf8"))+512 for m in messages)
        cfg = config()
        if self.input_upper+upper > cfg["path_input_upper_token_budget"] or self.output+cfg["max_output_tokens_per_slot"] > cfg["path_output_token_budget"]:
            raise ValueError("path budget would be exceeded")
        self.input_upper += upper

    def charge(self, usage):
        # Unknown provider token usage is not a free call.
        if not isinstance(usage,dict) or not all(isinstance(usage.get(k),int) for k in ("prompt_tokens","completion_tokens")):
            raise ValueError("provider usage unavailable")
        self.output += usage["completion_tokens"]
        if self.output > config()["path_output_token_budget"]:
            raise ValueError("provider output budget exceeded")


class Collector:
    def __init__(self, output, settings, plan_hash, reuse=None):
        self.output, self.settings, self.plan_hash = output, settings, plan_hash
        self.key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.lock = threading.Lock()
        self.observed_models = set()
        self.started = time.perf_counter()
        self.reuse=reuse

    def slot(self, case, name, messages, evidence, budget, prior=None):
        budget.request(messages)
        deps = fingerprint({"plan":self.plan_hash,"case":case,"slot":name,"messages":messages,
                            "settings":self.settings,"protocol":config()})
        path = self.output / "slots" / case.replace(":","_") / f"{name}.json"
        reused_from=None
        compatible=None
        if self.reuse:
            old_plan=json.loads((self.reuse/"plan.json").read_text())
            old_path=self.reuse/"slots"/case.replace(":","_")/f"{name}.json"
            if old_path.exists():
                old=json.loads(old_path.read_text())
                old_deps=fingerprint({"plan":fingerprint(old_plan),"case":case,"slot":name,"messages":messages,
                                      "settings":self.settings,"protocol":old_plan["config"]})
                same_request=all(old_plan["config"][k]==config()[k] for k in
                                 ("temperature","max_output_tokens_per_slot","system_prompt_hash"))
                if old["dependency_hash"]==old_deps and old["integrity"]==fingerprint(old["record"]) and same_request:
                    compatible=old["record"]
                    reused_from={"path":str(old_path.relative_to(ROOT)),"sha256":sha(old_path)}
        if path.exists():
            row = json.loads(path.read_text())
            if row["dependency_hash"] != deps or row["integrity"] != fingerprint(row["record"]):
                raise ValueError("stale or tampered slot")
            record = row["record"]
            reused_from=row.get("reused_from")
        elif compatible is not None:
            record=compatible
            create(path,{"dependency_hash":deps,"record":record,"integrity":fingerprint(record),"reused_from":reused_from})
        else:
            intent = path.with_suffix(".intent.json")
            if intent.exists():
                raise RuntimeError("unfinished request intent: billing state unknown; not retried")
            create(intent, {"dependency_hash":deps,"start_utc":utc(),"slot":name})
            import requests
            start = time.perf_counter()
            try:
                response = requests.post(self.settings["url"], headers={"Authorization":"Bearer "+self.key},
                    json={"model":self.settings["model"],"messages":messages,"temperature":0,
                          "max_tokens":config()["max_output_tokens_per_slot"],"stream":False,"response_format":{"type":"json_object"}},timeout=(15,150))
            except requests.RequestException:
                # No raw exception serialization: it can include headers/credentials.
                raise RuntimeError("network failure, unknown billing; unfinished intent retained") from None
            if response.status_code != 200:
                record = {"status":"http_error","http_status":response.status_code,
                          "elapsed_seconds":time.perf_counter()-start,"end_utc":utc(),
                          "usage":None,"billing":"unknown; not counted as zero"}
                create(path, {"dependency_hash":deps,"record":record,"integrity":fingerprint(record)})
                raise RuntimeError(f"provider HTTP {response.status_code}; halted, no automatic retry")
            raw = response.json()
            record = {"status":"response","raw_response":raw,"usage":raw.get("usage"),
                      "model_returned":raw.get("model"),"elapsed_seconds":time.perf_counter()-start,"end_utc":utc()}
            create(path, {"dependency_hash":deps,"record":record,"integrity":fingerprint(record)})
        if record["status"] != "response":
            raise RuntimeError("previous provider error cannot be resumed as success")
        model = record.get("model_returned")
        if not model:
            raise RuntimeError("missing returned model identity")
        with self.lock:
            self.observed_models.add(model)
            if len(self.observed_models)>1:
                raise RuntimeError("provider changed model identity within collection")
        budget.charge(record["usage"])
        raw = record["raw_response"]
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("non-stop finish; incomplete output recorded, not repaired")
        parsed,issues = collect_parse(choice["message"]["content"], evidence, prior)
        return parsed, {"slot":name,"usage":record["usage"],"elapsed_seconds":record["elapsed_seconds"],
                        "model_returned":model,"input_hash":fingerprint(messages),"report_integrity_errors":issues,
                        "reused_response":reused_from is not None}

    def run_case(self, inp, candidates, movies):
        evidence = bundle(inp,candidates,movies)
        cid = inp["case_id"]
        result_path = self.output / "cases" / f"{cid.replace(':','_')}.json"
        case_hash = fingerprint({"plan":self.plan_hash,"input":inp,"candidates":candidates,"evidence":evidence})
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved["dependency_hash"] != case_hash or saved["integrity"] != fingerprint(saved["record"]):
                raise ValueError("case checkpoint changed")
            # Re-validate referenced slot hashes, not only the final JSON envelope.
            for name, expected in saved["record"]["slot_hashes"].items():
                if sha(self.output / "slots" / cid.replace(":","_") / name) != expected:
                    raise ValueError("case slot changed")
            return saved["record"]
        started = time.perf_counter()
        paths = {"R0":{"status":"valid","ranking":candidates["ranked_movie_ids"][:5],"logical_calls":0,"usage":[]}}
        budgets = {name:Budget() for name in ("R1","R2","shared","R3","R4","R5")}
        def call(slot, task, budget, reports=None, history=None):
            return self.slot(cid,slot,prompt(evidence,task,reports,history),evidence,budget,reports)
        for arm in ("R1","R2"):
            history, use = [], []
            try:
                for n in range(1 if arm=="R1" else 6):
                    task = ("Make the final five-film recommendation using all evidence." if arm=="R1" or n==5
                            else "Independently examine all evidence, then review and improve your own prior recommendation. Find errors only when justified.")
                    report, usage = call(f"{arm}_{n}",task,budgets[arm],history=history)
                    history.append(report)
                    use.append(usage)
                paths[arm] = {"status":"valid","ranking":report["ranking"],"report":report,"logical_calls":len(use),"usage":use}
            except (ValueError,KeyError,IndexError,TypeError) as exc:
                paths[arm] = {"status":"invalid","error":str(exc),"ranking":[],"usage":use}
        initial, shared_use = {}, []
        try:
            for role in ROLES:
                report, use = call("initial_"+role,FOCUS[role],budgets["shared"])
                initial[role] = report
                shared_use.append(use)
        except (ValueError,KeyError,IndexError,TypeError) as exc:
            for arm in ("R3","R4","R5"):
                paths[arm] = {"status":"invalid","error":"shared initial: "+str(exc),"ranking":[],"usage":shared_use}
        if len(initial)==3:
            focus = disputes(initial)
            for arm in ("R3","R4","R5"):
                use = list(shared_use)
                budget = budgets[arm]
                budget.input_upper = budgets["shared"].input_upper
                budget.output = budgets["shared"].output
                reviews=[]
                try:
                    for n in range(0 if arm=="R3" else 2):
                        task = "Review the independent reports against all raw evidence. Identify justified errors, revise if appropriate, or retain uncertainty."
                        if arm=="R5":
                            task = ("Check the specific assigned claim against its cited evidence and the current request. "
                                    "Locate any mismatch of subject, intent, time or restriction. Record accept/reject/unresolved with evidence; do not force disagreement. "
                                    "Assigned claim: "+json.dumps(focus[n] if n<len(focus) else {"none":True},ensure_ascii=False))
                        reviewed, u = call(f"{arm}_review_{n}",task,budget,initial,reviews)
                        reviews.append(reviewed)
                        use.append(u)
                    # Identical output instruction across R3/R4/R5; no privileged parser or rank repair.
                    report,u = call(arm+"_final","Produce the final five-film recommendation. Consider all supplied evidence and reports. "
                                    "Do not equate agreement with truth; preserve unresolved claims when appropriate.",budget,initial,reviews)
                    use.append(u)
                    paths[arm] = {"status":"valid","ranking":report["ranking"],"report":report,"reviews":reviews,
                                  "logical_calls":len(use),"usage":use}
                except (ValueError,KeyError,IndexError,TypeError) as exc:
                    paths[arm] = {"status":"invalid","error":str(exc),"ranking":[],"usage":use}
        slots = self.output / "slots" / cid.replace(":","_")
        record = {"case_id":cid,"paths":paths,"initial_reports":initial,
                  "elapsed_seconds":time.perf_counter()-started,"end_utc":utc(),
                  "slot_hashes":{p.name:sha(p) for p in sorted(slots.glob("*.json")) if not p.name.endswith(".intent.json")}}
        create(result_path,{"dependency_hash":case_hash,"record":record,"integrity":fingerprint(record)})
        print(json.dumps({"case":cid,"paths":{k:v["status"] for k,v in paths.items()}},ensure_ascii=False),flush=True)
        return record


def settings():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    base = os.environ.get("DEEPSEEK_BASE_URL","https://api.deepseek.com").rstrip("/")
    if urlparse(base).scheme!="https" or urlparse(base).hostname!="api.deepseek.com":
        raise ValueError("only explicitly scoped official DeepSeek endpoint accepted")
    return {"url":base+("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions"),
            "model":os.environ.get("DEEPSEEK_MODEL","deepseek-chat")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("engineering","pilot"),required=True)
    parser.add_argument("--execute-authorized",action="store_true")
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    base=OUTPUT/"data"
    audit=json.loads((base/"development_retrieval_audit.json").read_text())
    if not audit["paid_pilot_data_gate"]:
        raise RuntimeError("data gate did not pass; no paid calls")
    acceptance=json.loads((OUTPUT/"offline_acceptance_execution_v3.json").read_text())
    if not acceptance["pass"] or acceptance["code_hashes"] != code_hashes():
        raise RuntimeError("offline acceptance missing/stale")
    ids=audit[args.stage+"_case_ids"]
    cfg=settings()
    account_fp=fingerprint(os.environ.get("DEEPSEEK_API_KEY",""))
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DeepSeek API configuration needed; no request sent")
    manifest=json.loads((base/"manifest.json").read_text())
    verify_artifacts(base,manifest["artifacts"])
    plan={"stage":args.stage,"case_ids":ids,"config":config(),"provider":cfg,"code_hashes":code_hashes(),
          "credential_fingerprint":account_fp,"manifest_hash":sha(base/"manifest.json"),
          "candidate_hash":sha(base/"dev.candidates.jsonl"),"audit_hash":sha(base/"development_retrieval_audit.json"),
          "max_concurrent_cases":args.workers,"max_unique_slots":17*len(ids),"automatic_retries":0,
          "external_success_not_guaranteed":True}
    out=OUTPUT/"collection_execution_v3"/args.stage
    create(out/"plan.json",plan)
    if not args.execute_authorized:
        print(json.dumps({"ready":True,"stage":args.stage,"cases":len(ids),"max_calls":plan["max_unique_slots"]}))
        return
    if args.stage=="pilot":
        engineering=json.loads((OUTPUT/"engineering_execution_gate_execution_v3.json").read_text())
        if not engineering["pass"]:
            raise RuntimeError("engineering gate failed; pilot not started")
    inputs={r["case_id"]:r for r in read_rows(base/"dev.inputs.jsonl")}
    candidates={r["case_id"]:r for r in read_rows(base/"dev.candidates.jsonl")}
    _,_,movies=__import__("experiments.scripts.data.build_redial_benchmark",fromlist=["load_catalog"]).load_catalog(
        ROOT/"experiments/data/processed/movie_catalog.jsonl")
    reuse=OUTPUT/"collection_interface_v2"/"engineering" if args.stage=="engineering" else None
    collector=Collector(out,cfg,fingerprint(plan),reuse=reuse)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(collector.run_case,inputs[cid],candidates[cid],movies) for cid in ids]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__=="__main__":
    main()
