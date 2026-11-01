"""Independent final audit and attribution, no network or model calls.

Run after the full development collection. No reserved labels are parsed/scored.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import io
import unittest
from collections import Counter
from unittest.mock import patch

from experiments.consensus_recommendation.common import ROOT, OUTPUT, code_hashes, create, fingerprint, read_rows, sha, verify_artifacts
from experiments.consensus_recommendation.data import make_input, make_label, worker_split
from experiments.consensus_recommendation.protocol import ROLES, bundle, collect_parse
from experiments.scripts.data.build_redial_benchmark import load_catalog


def measure(ranking,gold):
    hits=[int(x in gold) for x in ranking[:5]]
    ideal=sum(1/math.log2(i+2) for i in range(min(len(gold),5)))
    return int(any(hits)),sum(hits)/max(1,len(gold)),sum(x/math.log2(i+2) for i,x in enumerate(hits))/max(ideal,1e-15)


def noninteractive_controls(candidate,movies):
    """Fixed MMR and an explicitly approximate intent-coverage reranker.

    Genre prior comes from training-example retrieval ranks, NOT treating prefix
    mentions as positive preference labels. Scores are rank surrogates, not
    calibrated relevance probabilities or a faithful paper reproduction.
    """
    pool=candidate["ranked_movie_ids"]
    order={m:i for i,m in enumerate(pool)}
    relevance={m:1-order[m]/max(1,len(pool)) for m in pool}
    genres={m:set(movies[m].get("genres",[])) for m in pool}
    weights=Counter()
    for rank,mid in enumerate(candidate["example_movie_ids"][:20]):
        gs=movies[mid].get("genres",[])
        for g in gs: weights[g]+=1/(rank+1)/max(1,len(gs))
    total=sum(weights.values())
    prior={g:w/max(total,1e-12) for g,w in weights.items()}
    outputs={}
    for method in ("mmr_lambda_0.7","intent_coverage_lambda_0.3"):
        selected=[]
        remaining=list(pool)
        for _ in range(min(5,len(pool))):
            scores={}
            for mid in remaining:
                if method.startswith("mmr"):
                    redundancy=max([len(genres[mid]&genres[s])/max(1,len(genres[mid]|genres[s])) for s in selected], default=0.)
                    scores[mid]=.7*relevance[mid]-.3*redundancy
                else:
                    novelty=sum(p*(g in genres[mid])*(not any(g in genres[s] for s in selected)) for g,p in prior.items())
                    scores[mid]=.7*relevance[mid]+.3*novelty
            chosen=min(remaining,key=lambda m:(-scores[m],order[m],m))
            selected.append(chosen);remaining.remove(chosen)
        outputs[method]=selected
    return outputs


def audit():
    base=OUTPUT/"data"
    manifest=json.loads((base/"manifest.json").read_text())
    verify_artifacts(base,manifest["artifacts"])
    source={f"redial:{d['conversationId']}":d for d in read_rows(ROOT/"experiments/data/raw/redial/train.jsonl")}
    exact,titles,movies=load_catalog(ROOT/"experiments/data/processed/movie_catalog.jsonl")
    workers={}; replayed=Counter()
    for split in ("train","dev","reserved_test"):
        inputs=read_rows(base/f"{split}.inputs.jsonl")
        # Reserved label files are only hashed for integrity, never parsed.
        labels={r["case_id"]:r for r in read_rows(base/f"{split}.labels.jsonl")} if split!="reserved_test" else None
        ws=set()
        for inp in inputs:
            d=source[inp["case_id"]]
            if make_input(d,exact,titles)!=inp:
                raise ValueError("input differs from raw prefix")
            changed={**d,"initiatorQuestions":{},"respondentQuestions":{}}
            if make_input(changed,exact,titles)!=inp:
                raise ValueError("questionnaire label leakage")
            for k in ("initiatorWorkerId","respondentWorkerId"):
                w=str(d[k]); ws.add(w)
                if worker_split(w)!=split: raise ValueError("worker partition mismatch")
            if labels is not None and make_label(d,inp,exact,titles)!=labels[inp["case_id"]]:
                raise ValueError("offline label reconstruction differs")
            replayed[split]+=1
        workers[split]=ws
    if any(workers[a]&workers[b] for i,a in enumerate(workers) for b in list(workers)[i+1:]):
        raise ValueError("participants overlap")
    candidates={r["case_id"]:r for r in read_rows(base/"dev.candidates.jsonl")}
    dev={r["case_id"]:r for r in read_rows(base/"dev.inputs.jsonl")}
    golds={r["case_id"]:set(r["relevant_movie_ids"]) for r in read_rows(base/"dev.labels.jsonl")}
    unique_responses={}
    stages={}
    for stage in ("engineering","pilot"):
        root=OUTPUT/"collection_execution_v3"/stage
        plan=json.loads((root/"plan.json").read_text())
        summary=json.loads((root/"summary.json").read_text())
        if plan["code_hashes"]!=code_hashes(): raise ValueError("collection code drift")
        if len(plan["case_ids"])!=(12 if stage=="engineering" else 48): raise ValueError("sample size drift")
        if sha(root/"plan.json")!=summary["plan_hash"]: raise ValueError("summary plan hash")
        sums=Counter(); errors=Counter(); updates=Counter(); initial=Counter(); oracle=Counter()
        changed=Counter(); corrected=Counter(); harmed=Counter(); simple=Counter()
        for cid in plan["case_ids"]:
            obj=json.loads((root/"cases"/(cid.replace(":","_")+".json")).read_text())
            record=obj["record"]
            if obj["integrity"]!=fingerprint(record): raise ValueError("case changed")
            evidence=bundle(dev[cid],candidates[cid],movies)
            gold=golds[cid]
            for rule,ranked in noninteractive_controls(candidates[cid],movies).items():
                hit,recall,ndcg=measure(ranked,gold)
                simple[rule+":hit"]+=hit;simple[rule+":recall"]+=recall;simple[rule+":ndcg"]+=ndcg
            h={}
            for arm,p in record["paths"].items():
                ranked=p["ranking"] if p["status"]=="valid" else []
                hit,recall,ndcg=measure(ranked,gold)
                h[arm]=hit
                sums[arm+":hit"]+=hit; sums[arm+":recall"]+=recall; sums[arm+":ndcg"]+=ndcg
                if ranked and (len(set(ranked))!=5 or not set(ranked)<=set(candidates[cid]["ranked_movie_ids"])):
                    raise ValueError("out of pool ranking")
                for u in p.get("usage",[]):
                    errors[arm]+=bool(u.get("report_integrity_errors"))
                for review in p.get("reviews",[]):
                    for u in review.get("updates",[]):
                        updates[f"{arm}:{u.get('decision','invalid')}"]+=1
            for other in ("R0","R1","R2","R3","R4"):
                corrected[other]+=h["R5"]>h[other]
                harmed[other]+=h["R5"]<h[other]
                changed[other]+=record["paths"]["R5"]["ranking"]!=record["paths"][other]["ranking"]
            initial_hits=[]
            for role,r in record["initial_reports"].items():
                ih=measure(r["ranking"],gold)[0]
                initial[role]+=ih; initial_hits.append(ih)
            pool_order={mid:n for n,mid in enumerate(candidates[cid]["ranked_movie_ids"])}
            for rule in ("borda","vote"):
                points=Counter()
                for r in record["initial_reports"].values():
                    for pos,mid in enumerate(r["ranking"]): points[mid]+=5-pos if rule=="borda" else 1
                ranked=sorted(pool_order,key=lambda mid:(-points[mid],pool_order[mid],mid))[:5]
                hit,recall,ndcg=measure(ranked,gold)
                simple[rule+":hit"]+=hit;simple[rule+":recall"]+=recall;simple[rule+":ndcg"]+=ndcg
            oracle["initial_role_union_hit"]+=any(initial_hits)
            oracle["candidate_pool_hit"]+=bool(gold&set(candidates[cid]["ranked_movie_ids"]))
            oracle["initial_disagree_on_hit"]+=bool(initial_hits) and min(initial_hits)!=max(initial_hits)
            for name,expected in record["slot_hashes"].items():
                path=root/"slots"/cid.replace(":","_")/name
                if sha(path)!=expected: raise ValueError("slot changed")
                envelope=json.loads(path.read_text()); r=envelope["record"]
                if envelope["integrity"]!=fingerprint(r): raise ValueError("slot envelope changed")
                response=r.get("raw_response",{})
                response_id=response.get("id") or fingerprint(response)
                if response_id in unique_responses and fingerprint(unique_responses[response_id]["raw_response"])!=fingerprint(response):
                    raise ValueError("provider response ID collision")
                unique_responses[response_id]=r
                # Full raw response stays immutable; parse only for trace integrity.
                if r["status"]=="response" and response["choices"][0].get("finish_reason")=="stop":
                    try:
                        collect_parse(response["choices"][0]["message"]["content"],evidence,record["initial_reports"])
                    except (ValueError,TypeError,KeyError):
                        pass  # recorded invalid slot, not silently counted as valid
        n=len(plan["case_ids"])
        for arm in ("R0","R1","R2","R3","R4","R5"):
            saved=summary["methods"][arm]
            for k,key in (("hit","hit_lower"),("recall","recall_execution_lower"),("ndcg","ndcg_execution_lower")):
                if abs(sums[arm+":"+k]/n-saved[key])>1e-12:
                    raise ValueError("independent metric differs")
        stages[stage]={"requests":n,"participants_as_seekers":len({str(source[c]["initiatorWorkerId"]) for c in plan["case_ids"]}),
                       "corrected_vs_R5":dict(corrected),"harmed_vs_R5":dict(harmed),"ranking_changed_vs_R5":dict(changed),
                       "initial_role_hit_counts":dict(initial),"oracle_diagnostic_counts":dict(oracle),
                       "report_integrity_error_slots":dict(errors),"review_update_decisions":dict(updates),
                       "secondary_frozen_report_aggregation":{k:v/n for k,v in simple.items()},
                       "oracle_scope":"label-aware diagnostic upper bound, not deployable method or full utility"}
    # Include engineering failures for total project cost without double-counting reused responses.
    for folder in ("collection","collection_interface_v2"):
        for p in (OUTPUT/folder/"engineering/slots").glob("*/*.json"):
            if p.name.endswith(".intent.json"): continue
            record=json.loads(p.read_text())["record"]
            response=record.get("raw_response",{})
            if response:
                unique_responses[response.get("id") or fingerprint(response)]=record
    known=[r for r in unique_responses.values() if isinstance(r.get("usage"),dict)]
    totals={"unique_successful_provider_responses":len(unique_responses),
            "prompt_tokens":sum(r["usage"]["prompt_tokens"] for r in known),
            "completion_tokens":sum(r["usage"]["completion_tokens"] for r in known),
            "unknown_usage":len(unique_responses)-len(known),
            "actual_returned_models":sorted({r.get("model_returned") for r in known}),
            "cost_currency":"not estimated without current provider pricing/cache rates or bill",
            "sum_request_elapsed_seconds":sum(r.get("elapsed_seconds",0) for r in unique_responses.values())}
    acceptance=json.loads((OUTPUT/"offline_acceptance_execution_v3.json").read_text())
    for name,h in acceptance["prior_artifact_hashes"].items():
        if sha(ROOT/name)!=h: raise ValueError("previous frozen evidence changed")
    loaded=unittest.defaultTestLoader.loadTestsFromNames([
        "experiments.tests.test_consensus_recommendation","experiments.tests.test_recommendation_v2",
        "experiments.tests.test_slate_choice","experiments.tests.test_slate_controls"])
    def flatten(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite): yield from flatten(item)
            else: yield item
    # Some modules import earlier TestCase classes. Do not count imported copies
    # as additional independent tests; stdout warning layout is not a test count.
    unique_tests={t.id():t for t in flatten(loaded)}
    stream=io.StringIO()
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite(unique_tests.values()))
    if not tests.wasSuccessful(): raise RuntimeError(stream.getvalue())
    from experiments.consensus_recommendation import runner
    with patch("requests.post",side_effect=AssertionError("offline resume must not call provider")):
        with patch.object(sys,"argv",["runner","--stage","pilot","--execute-authorized"]):
            runner.main()
    import importlib.metadata
    runtime={"python":platform.python_version()}
    for name in ("numpy","scipy","scikit-learn","requests","python-dotenv"):
        runtime[name]=importlib.metadata.version(name)
    create(OUTPUT/"final_attribution.json",{"stages":stages,"project_cost_including_failed_engineering":totals,"runtime":runtime,
             "raw_prefixes_reconstructed":dict(replayed),"reserved_labels_semantically_read":False,
             "worker_overlap":0,"prior_evidence_intact":True,"tests_passed":tests.testsRun,
             "unique_test_ids":sorted(unique_tests),
             "offline_resume_without_provider_calls":True,
             "code_hashes":code_hashes(),"new_model_calls_by_auditor":0})
    print(json.dumps({"pass":True,"tests":tests.testsRun,"stages":stages,"cost":totals},ensure_ascii=False,indent=2))


if __name__=="__main__": audit()
