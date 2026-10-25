"""Collect causal follower-response records for AEIC-MD pilots.

The mechanism is rendered before generation.  Each response must first declare
an action and then emit AEIC JSON.  Rounds are synchronous: agents in round t
can only observe records from round t-1.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from experiments.core.llm import AGENT_PERSONAS, LLMClient, LLM_BACKENDS, install_dns_fallback
from experiments.core.mechanism.contracts import (
    CONTRACTS, apply_signal_mask, parse_action, render_contract, scale_costs,
    scale_rewards,
)
from experiments.core.mechanism.controller import MechanismController
from experiments.core.mechanism.behavioral_metrics import majority_stance
from experiments.core.mechanism.schema import AEICReport, AgentAction, AgentRound, ExperimentRecord
from experiments.core.mechanism.storage import RecordStore
from experiments.mechanism.runtime import classify_error, contiguous_resume_prefix
from experiments.core.trust import extract_stance, majority_vote
from experiments.core.similarity import SimilarityCalculator


BEHAVIORS = ["independent", "evidence_update", "social_only", "copy_pressure", "counterevidence"]
_thread_local = threading.local()


def load_local_questions(dataset: str, n_samples: int, split: str) -> List[dict]:
    """Load the frozen local question cache; never contact a dataset host."""
    path = ROOT / "experiments" / "results" / "cache" / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"local question cache missing: {path}; prepare it explicitly before collection"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") or []
    clean = [
        {"question": x["question"], "ground_truth": x["ground_truth"],
         "question_id": str(x.get("question_id", i)), "source_cache": str(path.relative_to(ROOT))}
        for i, x in enumerate(questions)
        if x.get("question") and x.get("ground_truth") and not x.get("error")
    ]
    split_path = Path(__file__).with_name("splits.json")
    if not split_path.exists():
        raise FileNotFoundError("splits.json missing; run make_splits.py first")
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    allowed = split_data["datasets"][dataset]["splits"][split]
    by_id = {x["question_id"]: x for x in clean}
    selected = [by_id[qid] for qid in allowed if qid in by_id]
    if len(selected) < n_samples:
        raise ValueError(f"{dataset}/{split} has only {len(selected)} valid questions, need {n_samples}")
    return selected[:n_samples]


def client_for(backend: str) -> LLMClient:
    key = f"client_{backend}"
    if not hasattr(_thread_local, key):
        setattr(_thread_local, key, LLMClient(backend=backend, max_tokens=1200))
    return getattr(_thread_local, key)


def parse_aeic(raw: str) -> AEICReport:
    text = raw or ""
    start = text.find("{")
    if start < 0:
        raise ValueError("response contains no AEIC JSON object")
    data, _end = json.JSONDecoder().raw_decode(text[start:])
    evidence = data.get("E", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    report = AEICReport(
        A=str(data.get("A", "")), E=[str(x) for x in evidence],
        I=str(data.get("I", "")), C=str(data.get("C", "")),
    )
    errors = report.validate()
    if errors:
        raise ValueError("; ".join(errors))
    return report


def peer_context(behavior: str, previous: List[AgentRound], agent_id: int) -> Tuple[str, List[str]]:
    peers = [x for x in previous if x.agent_id != agent_id]
    if behavior == "independent" or not peers:
        return "No peer information is visible. Reason independently.", []
    conclusions = "\n".join(f"Agent {x.agent_id}: {x.report.C}" for x in peers)
    evidence = [e for x in peers for e in x.report.E]
    if behavior == "social_only":
        majority = majority_stance([x.stance for x in peers]) or "no clear majority"
        return f"Peer majority stance (no evidence is provided): {majority}", []
    if behavior == "copy_pressure":
        return ("Peer reports are visible. Matching the apparent majority may increase social acceptance, "
                "but your formal contract remains unchanged.\n" + conclusions +
                "\nPeer evidence:\n" + "\n".join(evidence)), evidence
    if behavior == "counterevidence":
        own_previous = next((x for x in previous if x.agent_id == agent_id), None)
        opposing = [
            e for x in peers
            if own_previous is not None and x.stance not in (None, "neutral", own_previous.stance)
            for e in x.report.E
        ]
        if not opposing:
            return "No stance-opposing peer evidence is available this round.", []
        return (
            "Stance-opposing peer evidence is supplied for critical evaluation. "
            "Change stance only if it actually rebuts your earlier evidence:\n"
            + "\n".join(opposing)
        ), opposing
    supplied = evidence[:1]
    return "One new peer evidence item is supplied:\n" + "\n".join(supplied), supplied


def build_choice_prompt(question: str, contract_text: str, context: str,
                        reputation: Optional[float]) -> str:
    reputation_text = (f"{reputation:.3f}" if reputation is not None
                       else "not disclosed by the mechanism")
    return f"""{contract_text}

Current accumulated reputation: {reputation_text}
Information available this round:
{context}

Question:
{question}

You are a resource-constrained, self-interested follower. Choose the action that
maximizes your expected net utility: expected credit under the visible rule minus
the action's stated cost. Credit is uncertain and is paid only when the listed
criterion is actually satisfied. Do not choose extra work merely because the
mechanism or an action mentions AEIC. Choose your action before solving. Output only:
{{"ACTION":"one action name"}}
"""


def execution_prompt(question: str, context: str, action, draft: str = "") -> str:
    common = f"Question:\n{question}\n\nAvailable information:\n{context}\n"
    schema = ('Return strict JSON: {"A":"assumptions","E":["specific evidence"],'
              '"I":"inference","C":"Yes or No: concise answer"}')
    if action.value == "low_effort":
        return common + "Give a deliberately brief, low-cost report. Use at most one evidence item.\n" + schema
    if action.value == "defer_imitate":
        return common + "Defer to the visible peer information and synthesize a short report.\n" + schema
    if action.value == "verify" and draft:
        return (common + "Audit the draft below for factual or logical errors. Correct it when needed, "
                "then return the final verified AEIC report.\nDraft:\n" + draft + "\n" + schema)
    return common + "Reason independently and produce a complete AEIC report.\n" + schema


def _call_json(client: LLMClient, messages: list, temperature: float,
               max_tokens: int, stage: str) -> Tuple[str, dict]:
    raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens,
                      response_format={"type": "json_object"})
    usage = dict(client.last_usage)
    return raw, {
        "stage": stage,
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "latency_s": float(client.last_latency_s or 0.0),
        "status": client.last_status_code,
        "backend": client.backend,
        "model": client.model,
    }


def generate_one(question: str, agent_id: int, backend: str, contract_text: str,
                 context: str, reputation: Optional[float], dry_run: bool,
                 resource_tiered: bool = False, fixed_resources: bool = False,
                 deepseek_only_tiers: bool = False,
                 matched_profile: Optional[dict] = None):
    if resource_tiered and fixed_resources:
        raise ValueError("resource_tiered and fixed_resources are mutually exclusive")
    if dry_run:
        choice = '{"ACTION":"aeic_effort"}'
        raw = ('{"A":"mock premise","E":["mock evidence"],'
               '"I":"mock inference","C":"Yes: mock conclusion"}')
        trace = [
            {"stage": "choose", "prompt_tokens": 5, "completion_tokens": 2,
             "latency_s": 0.001, "status": 200},
            {"stage": "aeic_effort", "prompt_tokens": 5, "completion_tokens": 8,
             "latency_s": 0.001, "status": 200},
        ]
        return parse_action(choice), raw, trace
    persona = AGENT_PERSONAS[agent_id % len(AGENT_PERSONAS)]
    choice_client = client_for(backend)
    system = {"role": "system", "content": persona["system"]}
    choice_raw, choice_meta = _call_json(
        choice_client, [system, {"role": "user", "content": build_choice_prompt(
            question, contract_text, context, reputation)}],
        persona["temperature"], 80, "choose",
    )
    action = parse_action(choice_raw)
    execution_backend = backend
    if matched_profile:
        execution_backend = str(matched_profile["backend"])
    elif resource_tiered and action.value == "low_effort" and not deepseek_only_tiers:
        execution_backend = "zhipu_flash"
    elif resource_tiered and action.value == "verify":
        execution_backend = "deepseek_reasoner"
    client = client_for(execution_backend)
    limit = (int(matched_profile["max_tokens_per_call"]) if matched_profile else
             800 if fixed_resources else
             {"low_effort": 350, "aeic_effort": 1000,
              "defer_imitate": 500, "verify": 1000}[action.value])
    execution_action = AgentAction.AEIC_EFFORT if matched_profile else action
    draft_raw, draft_meta = _call_json(
        client, [system, {"role": "user", "content": execution_prompt(
            question, context, execution_action)}], persona["temperature"], limit,
        "matched_compute_draft" if matched_profile else
        ("verify_draft" if action.value == "verify" else action.value),
    )
    trace = [choice_meta, draft_meta]
    final_raw = draft_raw
    repeat_revision = (int(matched_profile.get("execution_calls", 1)) > 1
                       if matched_profile else action.value == "verify" and not fixed_resources)
    if repeat_revision:
        final_raw, verify_meta = _call_json(
            client, [system, {"role": "user", "content": execution_prompt(
                question, context, AgentAction.VERIFY, draft_raw)}],
            max(0.1, persona["temperature"] - 0.1), limit,
            "matched_compute_revision" if matched_profile else "verify_revision",
        )
        trace.append(verify_meta)
    try:
        parse_aeic(final_raw)
    except (ValueError, json.JSONDecodeError):
        # Formatting recovery is a separately logged call and may not add or
        # revise substantive claims. It prevents an otherwise usable response
        # from breaking a longitudinal sequence solely because JSON was cut or
        # escaped incorrectly.
        repair_client = client_for("deepseek")
        repair_prompt = (
            "Reformat the malformed response below as one strict JSON object "
            "with exactly A, E, I, C. Preserve the original conclusion and the "
            "core recoverable reason; do not add facts. This is an extreme "
            "compression task: A <=12 words; E exactly one item <=20 words; "
            "I <=20 words; C <=12 words and must preserve the original Yes/No. "
            "E must be a JSON list. Output JSON only, under 100 words total.\n\n"
            + final_raw
        )
        final_raw, repair_meta = _call_json(
            repair_client,
            [{"role": "system", "content": "You only repair JSON formatting."},
             {"role": "user", "content": repair_prompt}],
            0.0, 180, "format_repair",
        )
        trace.append(repair_meta)
        try:
            parse_aeic(final_raw)
        except (ValueError, json.JSONDecodeError):
            retry_prompt = (
                "Compress the malformed attempted repair below into exactly one JSON object. "
                "Use exactly four keys: A, E, I, C. A, I, C must each be at most 6 words. "
                "E must contain exactly one string of at most 8 words. Preserve the existing "
                "Yes/No conclusion and do not add facts. Output JSON only, at most 50 words.\n\n"
                + final_raw
            )
            final_raw, retry_meta = _call_json(
                repair_client,
                [{"role": "system", "content": "Return one tiny valid JSON object only."},
                 {"role": "user", "content": retry_prompt}],
                0.0, 96, "format_repair_retry",
            )
            trace.append(retry_meta)
            parse_aeic(final_raw)
    return action, final_raw, trace


def run_record(question: str, ground_truth: str, dataset: str, question_id: str,
               behavior: str, mechanism: str, backends: List[str], n_agents: int,
               n_rounds: int, seed: int, controller: MechanismController,
               reward_scale: float, cost_scale: float,
               signal_mask: Optional[str],
               history_policy: str,
               variant_tag: Optional[str],
               dry_run: bool,
               resource_tiered: bool,
               experiment_id: str,
               fixed_resources: bool = False,
               deepseek_only_tiers: bool = False,
               resource_profiles: Optional[dict] = None) -> ExperimentRecord:
    reward_suffix = f"@{reward_scale:g}" if reward_scale != 1.0 else ""
    cost_suffix = f"__cost@{cost_scale:g}" if cost_scale != 1.0 else ""
    signal_suffix = f"__signals@{signal_mask}" if signal_mask else ""
    history_suffix = f"__history@{history_policy}" if history_policy != "cumulative" else ""
    variant_suffix = f"__variant@{variant_tag}" if variant_tag else ""
    condition = (f"{behavior}__{mechanism}{reward_suffix}{cost_suffix}"
                 f"{signal_suffix}{history_suffix}{variant_suffix}")
    if dry_run:
        experiment_id = experiment_id + "-dry"
    model_assignments = {str(i): backends[i % len(backends)] for i in range(n_agents)}
    all_rounds: List[AgentRound] = []
    previous: List[AgentRound] = []
    base_spec = scale_costs(scale_rewards(CONTRACTS[mechanism], reward_scale), cost_scale)
    if signal_mask:
        base_spec = apply_signal_mask(base_spec, signal_mask)
    reputation_snapshot = dict(controller.reputation)

    try:
        for round_id in range(1, n_rounds + 1):
            jobs = []
            contexts = {}
            with ThreadPoolExecutor(max_workers=n_agents) as executor:
                for agent_id in range(n_agents):
                    context, visible_evidence = peer_context(behavior, previous, agent_id)
                    spec = controller.adaptive_spec(base_spec, agent_id)
                    contract_text = render_contract(spec)
                    visible_reputation = (None if history_policy == "hidden"
                                          else controller.reputation[agent_id])
                    contexts[agent_id] = (context, visible_evidence, visible_reputation)
                    profile_key = f"{dataset}|{question_id}|{round_id}|{agent_id}"
                    matched_profile = ((resource_profiles or {}).get(profile_key)
                                       if resource_profiles is not None else None)
                    if resource_profiles is not None and matched_profile is None:
                        raise ValueError(f"matched compute profile missing: {profile_key}")
                    jobs.append((agent_id, spec, contract_text, executor.submit(
                        generate_one, question, agent_id, model_assignments[str(agent_id)],
                        contract_text, context, visible_reputation, dry_run,
                        resource_tiered, fixed_resources, deepseek_only_tiers,
                        matched_profile,
                    )))
                current = []
                for agent_id, spec, contract_text, future in jobs:
                    action, raw, action_trace = future.result()
                    report = parse_aeic(raw)
                    context, visible_evidence, visible_reputation = contexts[agent_id]
                    previous_item = next((x for x in previous if x.agent_id == agent_id), None)
                    offer = spec.to_offer(build_choice_prompt(
                        question, contract_text, context, visible_reputation
                    ))
                    reputation_before = controller.reputation[agent_id]
                    settlement = controller.settle(
                        agent_id, offer, action, report,
                        ground_truth if round_id == n_rounds else None,
                        visible_evidence,
                        previous_item.report if previous_item else None,
                        previous_item.stance if previous_item else None,
                        visible_evidence,
                    )
                    item = AgentRound(
                        round_id=round_id, agent_id=agent_id, offered=offer,
                        chosen_action=action, report=report,
                        visible_peer_ids=[x.agent_id for x in previous if x.agent_id != agent_id],
                        visible_peer_reports=[{"agent_id": x.agent_id, "C": x.report.C,
                                               "E": x.report.E if visible_evidence else []}
                                              for x in previous if x.agent_id != agent_id],
                        stance=extract_stance(report.C), reward=settlement["credit"],
                        realized_cost=settlement["cost"],
                        reputation_before=reputation_before,
                        reputation_after=settlement["reputation_after"],
                        reward_components={k: v for k, v in settlement.items()
                                           if k.startswith("component_")},
                        token_input=sum(x["prompt_tokens"] for x in action_trace),
                        token_output=sum(x["completion_tokens"] for x in action_trace),
                        latency_s=sum(x["latency_s"] for x in action_trace),
                        raw_response=raw, action_trace=action_trace,
                    )
                    current.append(item)
                    all_rounds.append(item)
            previous = current
        decision = majority_vote([x.report.__dict__ for x in previous])
        return ExperimentRecord(
            experiment_id, dataset, question_id, condition, seed,
            model_assignments, all_rounds, ground_truth, decision, completed=True,
        )
    except Exception as exc:
        # A failed unit is not part of the longitudinal game. Roll back all
        # partial settlements so a retry sees the same leader state.
        controller.reputation.clear()
        controller.reputation.update(reputation_snapshot)
        return ExperimentRecord(
            experiment_id, dataset, question_id, condition, seed,
            model_assignments, all_rounds, ground_truth=ground_truth,
            completed=False, error=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="StrategyQA",
                        choices=["StrategyQA", "MMLU-Hard", "TruthfulQA"])
    parser.add_argument("--behavior", default="independent", choices=BEHAVIORS)
    parser.add_argument("--mechanism", default="no_mechanism", choices=sorted(CONTRACTS))
    parser.add_argument("--backends", nargs="+", default=["deepseek"])
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--question-ids", nargs="+", default=None,
                        help="Optional subset of IDs within the selected split")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--cost-scale", type=float, default=1.0)
    parser.add_argument("--signal-mask", default=None,
                        help="StaticAEIC ablation mask containing a subset of AEIC")
    parser.add_argument("--history-policy", default="cumulative",
                        choices=["cumulative", "reset", "hidden", "shuffled"])
    parser.add_argument("--variant-tag", default=None,
                        help="Stable alphanumeric label for robustness variants")
    parser.add_argument("--flip-threshold", type=float, default=0.25)
    parser.add_argument("--similarity", default="embedding",
                        choices=["embedding", "tfidf", "lexical"])
    parser.add_argument("--resource-tiered", action="store_true",
                        help="Enable action-specific token, call, and backend tiers")
    parser.add_argument("--deepseek-only-tiers", action="store_true",
                        help="With --resource-tiered, use deepseek-chat for low/regular and deepseek-reasoner for verify")
    parser.add_argument("--fixed-resources", action="store_true",
                        help="all declared actions use one call, assigned backend, and max_tokens=800")
    parser.add_argument("--resource-schedule", type=Path,
                        help="Exogenous per-question/agent compute schedule from Full Mechanism traces")
    parser.add_argument("--experiment-id", default="aeic-md-v10-pilot",
                        help="Protocol/cohort identifier; change it for formal or ablation runs")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Maximum attempts per experimental unit in this invocation")
    parser.add_argument("--max-infra-attempts", type=int, default=10,
                        help="Connection/DNS retries not counted as experimental attempts")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/mechanism/pilot/records.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    if args.max_infra_attempts < 1:
        parser.error("--max-infra-attempts must be >= 1")
    if args.resource_tiered and args.fixed_resources:
        parser.error("--resource-tiered and --fixed-resources are mutually exclusive")
    if args.deepseek_only_tiers and not args.resource_tiered:
        parser.error("--deepseek-only-tiers requires --resource-tiered")
    if args.resource_schedule and (args.resource_tiered or args.fixed_resources
                                   or args.mechanism != "no_mechanism"):
        parser.error("--resource-schedule requires no_mechanism and cannot combine with other resource modes")
    if args.cost_scale <= 0:
        parser.error("--cost-scale must be > 0")
    if args.signal_mask:
        normalized = "".join(x for x in "AEIC" if x in set(args.signal_mask.upper()))
        if normalized != args.signal_mask.upper() or args.mechanism != "static_aeic":
            parser.error("--signal-mask must be an ordered nonempty subset of AEIC and requires static_aeic")
        args.signal_mask = normalized
    if args.variant_tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.variant_tag):
        parser.error("--variant-tag must contain only letters, digits, dot, underscore, or hyphen")
    if not 0.0 <= args.flip_threshold <= 1.0:
        parser.error("--flip-threshold must be between 0 and 1")

    # A whole-file run lock prevents two resumed collectors from generating the
    # same unit concurrently. RecordStore still uses its shorter atomic lock.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_lock_path = args.out.with_suffix(args.out.suffix + ".run.lock")
    run_lock = run_lock_path.open("a+")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another collector is already writing {args.out}")

    if args.dry_run:
        questions = [{"question": "Mock question?", "ground_truth": "yes"}]
    else:
        from urllib.parse import urlparse
        for backend in args.backends:
            host = urlparse(LLM_BACKENDS[backend]["base_url"]).hostname
            if host:
                install_dns_fallback(host)
        if args.resource_tiered:
            tier_backends = (("deepseek_reasoner",) if args.deepseek_only_tiers
                             else ("zhipu_flash", "deepseek_reasoner"))
            for backend in tier_backends:
                host = urlparse(LLM_BACKENDS[backend]["base_url"]).hostname
                if host:
                    install_dns_fallback(host)
        requested = max(args.n_samples, 50 if args.question_ids else args.n_samples)
        questions = load_local_questions(args.dataset, requested, args.split)
        if args.question_ids:
            wanted = {str(x) for x in args.question_ids}
            questions = [x for x in questions if x["question_id"] in wanted]
            missing = wanted - {x["question_id"] for x in questions}
            if missing:
                raise ValueError(f"question IDs not found in {args.dataset}/{args.split}: {sorted(missing)}")

    store = RecordStore(args.out)
    if args.dry_run or args.similarity == "lexical":
        similarity_fn = None
    else:
        similarity = SimilarityCalculator(method=args.similarity)
        similarity_fn = similarity.compute
    controller = MechanismController(
        args.n_agents, similarity_fn=similarity_fn,
        flip_threshold=args.flip_threshold,
    )
    resource_profiles = None
    if args.resource_schedule:
        payload = json.loads(args.resource_schedule.read_text(encoding="utf-8"))
        resource_profiles = payload.get("profiles")
        if not isinstance(resource_profiles, dict) or not resource_profiles:
            raise ValueError("resource schedule contains no profiles")
    limit = len(questions) if args.question_ids else args.n_samples
    reward_suffix = f"@{args.reward_scale:g}" if args.reward_scale != 1.0 else ""
    cost_suffix = f"__cost@{args.cost_scale:g}" if args.cost_scale != 1.0 else ""
    signal_suffix = f"__signals@{args.signal_mask}" if args.signal_mask else ""
    history_suffix = (f"__history@{args.history_policy}"
                      if args.history_policy != "cumulative" else "")
    variant_suffix = f"__variant@{args.variant_tag}" if args.variant_tag else ""
    current_condition = (f"{args.behavior}__{args.mechanism}{reward_suffix}{cost_suffix}"
                         f"{signal_suffix}{history_suffix}{variant_suffix}")
    effective_experiment_id = args.experiment_id + ("-dry" if args.dry_run else "")

    # Restore the longitudinal state only from a contiguous completed prefix.
    # A later checkpoint after a gap belongs to a different history and must
    # never be used to resume an earlier missing question.
    ordered_ids = [str(item.get("question_id", i))
                   for i, item in enumerate(questions[:limit])]
    stored_records = store.load()
    eligible = {
        str(item["question_id"]): item for item in stored_records.values()
        if item.get("completed") is True
        and item.get("experiment_id") == effective_experiment_id
        and item.get("dataset") == args.dataset
        and item.get("condition") == current_condition
        and int(item.get("seed", -1)) == args.seed
        and str(item.get("question_id")) in set(ordered_ids)
    }
    completed_prefix = contiguous_resume_prefix(ordered_ids, eligible)
    if completed_prefix:
        latest = eligible[completed_prefix[-1]]
        latest_round = max((x.get("round_id", 0) for x in latest.get("rounds", [])), default=0)
        for item in latest.get("rounds", []):
            if item.get("round_id") == latest_round and item.get("reputation_after") is not None:
                controller.reputation[int(item["agent_id"])] = float(item["reputation_after"])
        print(f"restored reputation after question {latest['question_id']}: "
              f"{controller.reputation_snapshot()}")

    completed_keys = store.completed_keys()
    for index, item in enumerate(questions[:limit]):
        qid = str(item.get("question_id", index))
        probe = (effective_experiment_id,
                 args.dataset, current_condition, qid, args.seed)
        if probe in completed_keys:
            continue
        if args.history_policy == "reset":
            for agent_id in range(args.n_agents):
                controller.reputation[agent_id] = 0.0
        elif args.history_policy == "shuffled":
            values = [controller.reputation[i] for i in range(args.n_agents)]
            random.Random(f"{args.seed}|{args.dataset}|{current_condition}|{qid}").shuffle(values)
            for agent_id, value in enumerate(values):
                controller.reputation[agent_id] = value
        attempt = 1
        infra_attempt = 0
        while attempt <= args.max_attempts:
            record = run_record(
                item["question"], item["ground_truth"], args.dataset, qid,
                args.behavior, args.mechanism, args.backends, args.n_agents,
                args.n_rounds, args.seed, controller,
                reward_scale=args.reward_scale, cost_scale=args.cost_scale,
                signal_mask=args.signal_mask,
                history_policy=args.history_policy,
                variant_tag=args.variant_tag,
                dry_run=args.dry_run,
                resource_tiered=args.resource_tiered,
                experiment_id=args.experiment_id,
                fixed_resources=args.fixed_resources,
                deepseek_only_tiers=args.deepseek_only_tiers,
                resource_profiles=resource_profiles,
            )
            error_class = classify_error(record.error)
            infrastructure_error = error_class == "transient"
            if infrastructure_error and infra_attempt < args.max_infra_attempts:
                infra_attempt += 1
                print(f"{qid} transient retry {infra_attempt}: {record.error}")
                time.sleep(min(2 ** min(infra_attempt - 1, 3), 8))
                continue
            store.upsert(record)
            print(f"{qid} attempt {attempt}: {'complete' if record.completed else record.error}")
            if record.completed:
                completed_keys.add(probe)
                break
            if error_class == "protocol":
                print(f"{qid} non-retryable protocol error; stopping immediately")
                break
            if attempt < args.max_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
            attempt += 1
        if not record.completed:
            raise SystemExit(
                f"stopping at incomplete longitudinal unit {args.dataset}/"
                f"{current_condition}/{qid}; resume after correction"
            )


if __name__ == "__main__":
    main()
