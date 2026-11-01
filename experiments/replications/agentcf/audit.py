"""Offline audit of pinned AgentCF assets and isolated original methods.

Fixtures test control flow only, never recommendation effectiveness. No upstream
module import, deserialization of pickle, or network operation is performed.
"""
import ast
import asyncio
import contextlib
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from .acquire import COMMIT

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / "experiments/results/replications/agentcf_2026-10-26"


def original_method(source, class_name, method_name):
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {}
    exec(compile(module, str(source), "exec"), scope)
    return scope[method_name]


class Vector(list):
    def size(self, axis):
        assert axis == 0
        return len(self)


def training_probe(model_path, accuracies):
    method = original_method(model_path, "AgentCF", "calculate_loss")
    events = []
    agent = lambda: SimpleNamespace(update_memory=["initial"], memory_1=["initial"], memory_embedding={})
    state = SimpleNamespace(USER_ID="user", ITEM_ID="positive", NEG_ITEM_ID="negative",
                            config={"all_update_rounds": 2, "evaluation": "basic"},
                            user_agents={1: agent()}, item_agents={2: agent(), 3: agent()},
                            item_text={2: "positive", 3: "negative"},
                            rec_agent=SimpleNamespace(user_examples=defaultdict(dict)))
    schedule = iter(accuracies)
    def forward(users, positive, negative):
        events.append({"event": "forward", "users": list(users)})
        return ["choice"], ["fixture reason"]
    def convert(*args):
        return next(schedule)
    def backward(reasons, users, positive, negative):
        events.append({"event": "wrong_feedback", "users": list(users)})
    def true_feedback(reasons, users, positive, negative, first):
        events.append({"event": "correct_feedback", "users": list(users), "first": first})
    state.forward, state.convert_system_selections_to_accuracy = forward, convert
    state.backward, state.backward_true = backward, true_feedback
    state.logging_after_updation = lambda *args: None
    with contextlib.redirect_stdout(io.StringIO()):
        method(state, {"user": Vector([1]), "positive": Vector([2]), "negative": Vector([3])})
    return events


def memory_update_probe(model_path, update_negative):
    method = original_method(model_path, "AgentCF", "backward")
    method.__globals__["asyncio"] = asyncio
    async def response(prompts):
        return [{"choices": [{"message": {"content": "fixture"}}]} for _ in prompts]
    client = SimpleNamespace(agenerate_response=response, agenerate_response_without_construction=response)
    parser = SimpleNamespace(parse_update=lambda _: "user updated",
                             parse=lambda _: ["negative updated", "positive updated"])
    def agent(title):
        return SimpleNamespace(update_memory=["initial"], role_description={"item_title": title},
                               astep_backward=lambda *args: "fixture prompt", llm_chat=client, output_parser=parser)
    state = SimpleNamespace(user_agents={0: agent("pad"), 1: agent("user")},
                            item_agents={0: agent("pad"), 2: agent("positive"), 3: agent("negative")},
                            chat_api_batch=1, config={"update_neg_item": update_negative},
                            logging_during_updation=lambda *args: None)
    with contextlib.redirect_stdout(io.StringIO()):
        method(state, ["fixture reason"], [1], [2], [3])
    return {"user": state.user_agents[1].update_memory,
            "positive": state.item_agents[2].update_memory,
            "negative": state.item_agents[3].update_memory}


def inference_client_probe(upstream):
    """Execute original context construction and evaluator with a fake client."""
    import yaml
    config = defaultdict(lambda: None, yaml.safe_load((upstream / "props/AgentCF.yaml").read_text()))
    make_context = original_method(upstream / "model/agentcf.py", "AgentCF", "load_user_context")
    contexts = make_context(SimpleNamespace(config=config, dataset_name="CDs-100-user-dense", n_users=2))
    actual = contexts[0]["llm_chat"]
    calls = []
    async def response(prompts):
        calls.append({"model": actual["model"], "temperature": actual["temperature"], "prompts": len(prompts)})
        return [{"choices": [{"message": {"content": "fixture ranking"}}]} for _ in prompts]
    state = SimpleNamespace(config=config, chat_api_batch=10,
                            rec_agent=SimpleNamespace(astep_evaluation=lambda *args: "fixture prompt",
                                                      output_parser=SimpleNamespace(parse_evaluation=lambda text: text)),
                            user_agents={0: SimpleNamespace(llm_chat=SimpleNamespace(agenerate_response_without_construction=response))})
    evaluate = original_method(upstream / "model/agentcf.py", "AgentCF", "evaluation")
    evaluate.__globals__.update(asyncio=asyncio, tqdm=lambda seq: seq)
    evaluate(state, [1], ["fixture user"], None, ["fixture candidates"])
    return {"actual_client_calls": calls, "declared_test_temperature": config["llm_temperature_test"],
            "note": "fake transport; no provider request; parser mocked in this client-routing probe"}


def missing_rank_probe(upstream):
    parse = original_method(upstream / "agentverse/tasks/recommendation/output_parser.py", "RecommenderParser", "parse_evaluation")
    parse.__globals__["re"] = re
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            parse(None, "no required ranking prefix")
        except Exception as error:
            return {"exception": type(error).__name__, "note": "malformed fixture, not a model output"}
    return {"exception": None}


def data_audit(folder):
    splits = {}
    for split in ("train", "valid", "test"):
        path = folder / (folder.name + "." + split + ".inter")
        with path.open() as handle:
            rows = list(csv.reader(handle, delimiter="\t"))[1:]
        splits[split] = rows
    catalog = {line.split("\t")[0] for line in (folder / "CDs.item").read_text().splitlines()[1:]}
    by_user = {split: defaultdict(list) for split in splits}
    for split, rows in splits.items():
        for user, history, target in rows:
            by_user[split][user].append((history.split(), target))
    errors = []
    for user, train_rows in by_user["train"].items():
        previous = None
        for history, target in train_rows + by_user["valid"][user] + by_user["test"][user]:
            if previous is not None and history != previous:
                errors.append({"user": user, "issue": "non-prefix transition"})
            previous = history + [target]
    candidates = {}
    for line in (folder / (folder.name + ".random")).read_text().splitlines():
        user, items = line.split("\t")
        candidates[user] = items.split()
    ids = {item for rows in splits.values() for _, history, target in rows for item in history.split() + [target]}
    eval_candidates = {}
    for user, cases in by_user["test"].items():
        history, target = cases[0]
        pool = candidates[user][:]
        if target in pool:
            position = pool.index(target)
            pool[position:-1] = pool[position + 1:]
        final = pool[:9] + [target]  # exactly trainer.py before permutation
        eval_candidates[user] = {"duplicates": len(final) - len(set(final)),
                                 "historical_candidates": len(set(final) & set(history)),
                                 "target_in_history": target in history}
    return {"rows": {s: len(r) for s, r in splits.items()},
            "users": {s: len(by_user[s]) for s in splits},
            "observed_items_including_histories": len(ids),
            "train_target_count_per_user": dict(Counter(len(v) for v in by_user["train"].values())),
            "prefix_transition_errors": errors,
            "catalog_missing_interaction_ids": sorted(ids - catalog),
            "candidate_users": len(candidates),
            "candidate_length_counts": dict(Counter(map(len, candidates.values()))),
            "candidate_duplicate_rows": sum(len(v) != len(set(v)) for v in candidates.values()),
            "candidate_catalog_missing_count": len({i for v in candidates.values() for i in v} - catalog),
            "actual_test_10_candidate_duplicate_rows": sum(v["duplicates"] > 0 for v in eval_candidates.values()),
            "actual_test_10_candidate_history_overlap_rows": sum(v["historical_candidates"] > 0 for v in eval_candidates.values()),
            "test_target_in_own_history_rows": sum(v["target_in_history"] for v in eval_candidates.values()),
            "note": "prefix consistency is not timestamp provenance; no timestamps shipped in these split files"}


def run(root=SNAPSHOT):
    manifest = json.loads((root / "source_manifest.json").read_text())
    assert manifest["commit"] == COMMIT
    tree = json.loads((root / "upstream_tree.json").read_text())
    assert tree["sha"] == COMMIT and not tree.get("truncated")
    tree_blobs = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    assert len({entry["path"] for entry in manifest["files"]}) == len(manifest["files"])
    for entry in manifest["files"]:
        data = (root / "upstream" / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert digest == entry["git_blob"] == tree_blobs[entry["path"]]["sha"]
        assert len(data) == entry["bytes"] == tree_blobs[entry["path"]]["size"]
    upstream = root / "upstream/agentcf"
    probes = {"correct_then_correct": training_probe(upstream / "model/agentcf.py", [[1], [1]]),
              "wrong_then_correct": training_probe(upstream / "model/agentcf.py", [[0], [1]]),
              "correct_then_wrong": training_probe(upstream / "model/agentcf.py", [[1], [0]])}
    report = {"commit": COMMIT, "verified_files": len(manifest["files"]), "model_calls": 0,
              "status": "offline_source_and_control_flow_audit_only_not_replication_complete",
              "datasets": {name: data_audit(upstream / "dataset" / name) for name in
                           ["CDs-100-user-dense", "CDs-100-user-sparse"]},
              "control_flow_probes": probes,
              "memory_update_probes": {"default_null": memory_update_probe(upstream / "model/agentcf.py", None),
                                       "explicit_true": memory_update_probe(upstream / "model/agentcf.py", True)},
              "office_paths_in_upstream": [e["path"] for e in tree["tree"] if "office" in e["path"].lower()],
              "dependency_lock_paths": [e["path"] for e in tree["tree"] if e["path"].endswith(("requirements.txt", "poetry.lock", "environment.yml", "pyproject.toml"))]}
    (root / "offline_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
