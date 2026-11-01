"""No-paid preflight: tests, data hashes, source snapshot, all-path budget bounds."""
from __future__ import annotations

import json
import subprocess
import sys

from .common import ROOT, OUTPUT, code_hashes, create, read_rows, sha, verify_artifacts
from .protocol import config


def main():
    data=OUTPUT/"data"
    manifest=json.loads((data/"manifest.json").read_text())
    verify_artifacts(data,manifest["artifacts"])
    for name,h in manifest["sources"].items():
        if sha(ROOT/name)!=h: raise ValueError("raw source changed")
    if any(manifest["worker_overlap"].values()) or manifest["cutoff_uses_labels"]:
        raise ValueError("input isolation failed")
    audit=json.loads((data/"development_retrieval_audit.json").read_text())
    if sha(data/"dev.candidates.jsonl")!=audit["candidate_sha256"]:
        raise ValueError("candidate integrity failed")
    ids=audit["engineering_case_ids"]+audit["pilot_case_ids"]
    if len(ids)!=60 or len(set(ids))!=60:
        raise ValueError("collection lists overlap or lack required cases")
    result=subprocess.run([sys.executable,"-m","unittest","experiments.tests.test_consensus_recommendation","-v"],
                          cwd=ROOT,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    # Preserve evidence that serving-system / earlier research artifacts were untouched.
    previous={}
    for name in ("experiments/slate_choice/protocol.json","experiments/slate_choice/controls_protocol.json",
                 "experiments/results/slate_choice_v1/acceptance.json",
                 "experiments/results/slate_choice_v1/controls/acceptance.json",
                 "experiments/results/recommendation_v2/acceptance_verified.json"):
        p=ROOT/name
        if p.exists(): previous[name]=sha(p)
    create(OUTPUT/"offline_acceptance_execution_v3.json",{"pass":True,"code_hashes":code_hashes(),"protocol":config(),
            "tests_passed":result.stderr.count(" ... ok"),"data_manifest_sha256":sha(data/"manifest.json"),
            "data_gate_passed":audit["paid_pilot_data_gate"],"prior_artifact_hashes":previous,
            "model_calls":0,"reserved_test_scored":False})
    print(json.dumps({"pass":True,"data_gate":audit["paid_pilot_data_gate"],"tests":result.stderr.count(" ... ok")},ensure_ascii=False))


if __name__=="__main__": main()
