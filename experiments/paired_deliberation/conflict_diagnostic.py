"""Four symmetric conflict probes; diagnostic only, never repairs original scores."""
import argparse
import copy
import json
import re
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.controlled_update_gate import execute


def prepare(source, output):
    source, output = Path(source), Path(output)
    original_plan = json.loads((source / "plan.json").read_text())
    inputs, labels, groups = [], [], set()
    for row in json.loads((source / "inputs.json").read_text()):
        group = row["evidence"]["case_id"]
        if not group.endswith(":conflicting_evidence") or group in groups:
            continue
        groups.add(group)
        genre = re.search(r"I want (.+) tonight", row["evidence"]["observations"][0]["text"]).group(1)
        for negative in (False, True):
            inp = {key: copy.deepcopy(row[key]) for key in ("evidence", "reports")}
            inp["reports"]["current_request"]["claims"][0]["text"] = (
                "The seeker " + ("does not want " if negative else "wants ") + genre + " tonight.")
            item_id = "conflict_" + fingerprint(inp)[:16]
            inputs.append({"id": item_id, **inp})
            labels.append({"id": item_id, "label": "insufficient", "category": "direct_conflict", "group": group})
    assert len(inputs) == 4
    create(output / "inputs.json", inputs)
    create(output / "labels.json", labels)
    plan = {**original_plan, "version": "d1-symmetric-conflict-diagnostic-4-r1", "items": 4, "groups": 2,
        "maximum_requests": 4, "limits": {**original_plan["limits"], "max_calls": 4,
                                         "total_completion_tokens": 5120},
        "inputs_hash": fingerprint(inputs), "labels_hash": fingerprint(labels),
        "scope": "posthoc four-probe diagnosis of unresolved conflicting evidence. Both opposing preference "
                 "claims are prespecified insufficient; cannot replace original failed/ambiguous category or certify general accuracy."}
    plan.pop("plan_hash")
    plan["code_sha256"][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    plan["plan_hash"] = fingerprint(plan)
    create(output / "plan.json", plan)
    return {"status": "prepared", "items": 4, "new_api_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.output) if args.execute_authorized else prepare(args.source, args.output),
                     ensure_ascii=False, indent=2))
