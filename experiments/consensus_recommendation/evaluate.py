"""Separate offline scorer; model collector never imports this module."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest, norm

from .common import ROOT, OUTPUT, create, fingerprint, read_rows, sha


def metrics(ranked, gold):
    gold=set(gold)
    matched=[int(m in gold) for m in ranked[:5]]
    ideal=sum(1/math.log2(i+2) for i in range(min(5,len(gold))))
    return {"hit":int(any(matched)),"recall":sum(matched)/max(1,len(gold)),
            "ndcg":sum(x/math.log2(i+2) for i,x in enumerate(matched))/max(1e-15,ideal)}


def paired(a,b,groups):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    delta=a-b
    unique=sorted(set(groups))
    sums=np.array([delta[np.array(groups)==g].sum() for g in unique])
    sizes=np.array([sum(x==g for x in groups) for g in unique])
    rng=np.random.default_rng(20261026)
    indices=rng.integers(0,len(unique),(5000,len(unique)))
    boot=sums[indices].sum(axis=1)/sizes[indices].sum(axis=1)
    wins,losses=int(np.sum(delta>0)),int(np.sum(delta<0))
    return {"difference":float(delta.mean()),"corrected":wins,"harmed":losses,
            "seeker_cluster_bootstrap_95ci":[float(x) for x in np.quantile(boot,[.025,.975])],
            "seeker_clusters":len(unique),"cases":len(a),
            "case_mcnemar_p_descriptive_only":float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
            "limits":"bootstrap clusters by seeker; shared recommender and repeated conversations can retain dependency; not confirmatory"}


def power(discordance, delta=.03, power=.8, alpha=.025):
    # Two primary comparisons -> conservative Bonferroni planning bound.
    z=norm.ppf(1-alpha/2)+norm.ppf(power)
    return int(math.ceil(z*z*max(discordance,delta)/(delta*delta)))


def stage_report(stage):
    root=OUTPUT/"collection_execution_v3"/stage
    plan=json.loads((root/"plan.json").read_text())
    labels={r["case_id"]:r for r in read_rows(OUTPUT/"data/dev.labels.jsonl")}
    candidates={r["case_id"]:r for r in read_rows(OUTPUT/"data/dev.candidates.jsonl")}
    raw={f"redial:{r['conversationId']}":r for r in read_rows(ROOT/"experiments/data/raw/redial/train.jsonl")}
    cases=[]
    for cid in plan["case_ids"]:
        path=root/"cases"/(cid.replace(":","_")+".json")
        if not path.exists():
            raise ValueError("collection incomplete; not scored")
        row=json.loads(path.read_text())
        if row["integrity"]!=fingerprint(row["record"]):
            raise ValueError("tampered result")
        for name,h in row["record"]["slot_hashes"].items():
            if sha(root/"slots"/cid.replace(":","_")/name)!=h:
                raise ValueError("tampered slot")
        cases.append(row["record"])
    methods=["R0","R0_examples","R0_lexical","R1","R2","R3","R4","R5"]
    values={m:[] for m in methods}
    valid=Counter()
    changed=Counter()
    groups=[]
    for case in cases:
        cid=case["case_id"]
        gold=labels[cid]["relevant_movie_ids"]
        groups.append(str(raw[cid]["initiatorWorkerId"]))
        pool=set(candidates[cid]["ranked_movie_ids"])
        for method in methods:
            if method.startswith("R0_"):
                key="example_movie_ids" if method=="R0_examples" else "lexical_movie_ids"
                ranked=[m for m in candidates[cid][key] if m in pool]
                # Deterministic fallback from SAME shared pool, no target injection.
                ranked=(ranked+[m for m in candidates[cid]["ranked_movie_ids"] if m not in ranked])[:5]
                ok=True
            else:
                p=case["paths"][method]
                ranked=p["ranking"]
                ok=p["status"]=="valid"
            valid[method]+=ok
            values[method].append(metrics(ranked if ok else [],gold))
        for other in ("R2","R3","R4"):
            changed["R5_vs_"+other]+=case["paths"]["R5"]["ranking"]!=case["paths"][other]["ranking"]
    n=len(cases)
    summary={m:{"valid":valid[m],"cases":n,"hit_lower":sum(v["hit"] for v in vs)/n,
                "hit_upper_missing_as_success":(sum(v["hit"] for v in vs)+n-valid[m])/n,
                "recall_execution_lower":sum(v["recall"] for v in vs)/n,
                "ndcg_execution_lower":sum(v["ndcg"] for v in vs)/n}
             for m,vs in values.items()}
    comparisons={f"R5_vs_{other}":paired([r["hit"] for r in values["R5"]],
                    [r["hit"] for r in values[other]],groups) for other in ("R4","R2","R3","R0","R0_examples")}
    slots=[]
    reused=0
    for p in sorted((root/"slots").glob("*/*.json")):
        if not p.name.endswith(".intent.json"):
            item=json.loads(p.read_text())
            reused+=bool(item.get("reused_from"))
            slots.append((p,item["record"]))
    known=[r for _,r in slots if isinstance(r.get("usage"),dict)]
    actual={"records":len(slots),"reused_exact_request_records":reused,"new_requests_with_record":len(slots)-reused,
            "successful_http_response_records":sum(r["status"]=="response" for _,r in slots),
            "prompt_tokens":sum(r["usage"]["prompt_tokens"] for r in known),
            "completion_tokens":sum(r["usage"]["completion_tokens"] for r in known),
            "unknown_usage_records":len(slots)-len(known),
            "request_elapsed_seconds_sum":sum(r.get("elapsed_seconds",0) for _,r in slots),
            "returned_models":sorted({r.get("model_returned") for _,r in slots if r.get("model_returned")})}
    logical={}
    for method in ("R1","R2","R3","R4","R5"):
        selected=[r for p,r in slots if p.stem.startswith(method+"_") or
                  (method in ("R3","R4","R5") and p.stem.startswith("initial_"))]
        u=[r["usage"] for r in selected if isinstance(r.get("usage"),dict)]
        logical[method]={"actual_slot_records_including_invalid":len(selected),
                         "prompt_tokens":sum(x["prompt_tokens"] for x in u),
                         "completion_tokens":sum(x["completion_tokens"] for x in u),
                         "unknown_usage":len(selected)-len(u),
                         "elapsed_seconds_sum":sum(r.get("elapsed_seconds",0) for r in selected)}
    engineering_pass=all(valid[m]==n for m in ("R1","R2","R3","R4","R5"))
    signal=all(valid[m]/n>=.95 for m in ("R2","R4","R5")) and all(
        comparisons[k]["difference"]>0 for k in ("R5_vs_R4","R5_vs_R2"))
    integrity_errors={m:sum(bool(u.get("report_integrity_errors")) for case in cases for u in case["paths"][m].get("usage",[]))
                      for m in ("R1","R2","R3","R4","R5")}
    report={"stage":stage,"plan_hash":sha(root/"plan.json"),"methods":summary,"paired":comparisons,
            "report_integrity_error_slots_by_path":integrity_errors,
            "ranking_changes":dict(changed),"project_actual_usage":actual,"per_method_logical_usage":logical,
            "engineering_execution_pass":engineering_pass,"pilot_positive_direction_gate":signal,
            "process_semantic_truth":"not independently annotated; no evidence-correctness claim",
            "constraint_satisfaction":"not evaluated; no independent hard-constraint labels",
            "reserved_test_scored":False,"recommendation_improvement_confirmed":False,
            "interpretation":"development only; invalid outputs yield execution lower/upper bounds, not knowledge scores"}
    if stage=="pilot":
        disc=max((comparisons[k]["corrected"]+comparisons[k]["harmed"])/n for k in ("R5_vs_R4","R5_vs_R2"))
        # Only read reserved INPUT cardinality. Its labels remain unscored.
        reserved_n=len(read_rows(OUTPUT/"data/reserved_test.inputs.jsonl"))
        report["formal_feasibility"]={"illustrative_target_delta":.03,"estimated_independent_pairs":power(disc),
                    "reserved_request_count_without_label_eligibility":reserved_n,
                    "automatic_formal_run_allowed":False,
                    "reason":"positive-direction gate is not a power/independence certificate; complete separate protocol first"}
    create(root/"summary.json",report)
    if stage=="engineering":
        create(OUTPUT/"engineering_execution_gate_execution_v3.json",{"pass":engineering_pass,"summary_sha256":sha(root/"summary.json"),
                                                       "plan_hash":sha(root/"plan.json"),"scientific_effect_test":False})
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("engineering","pilot"),required=True)
    args=parser.parse_args()
    stage_report(args.stage)


if __name__=="__main__": main()
