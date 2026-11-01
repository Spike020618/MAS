# Unified evidence-consensus E1

This package is an **engineering integration**, not a trained or validated final
mechanism. It joins native QA and recommendation adapters, AEIC reports, source
expansion, versioned updates, budget accounting, fixed/central/game allocation,
and actual model calls. Sources and accepted reports do not certify truth.

## Code

- `schema.py`: native option/candidate validation, AEIC claims/edges/patch schema,
  source scope/authority.
- `state.py`: immutable catalog identity, fetched evidence, registered claims,
  state versions, explicit decision initialization and local updates.
- `actions.py`: real catalog text expansion/reuse, source-authority inspection,
  task-wide reserve/settle budget (unknown usage remains reserved).
- `task_adapter.py`, `prepare_e1.py`: 12 native StrategyQA and 12 native ReDial
  development tasks, shared permissions, label access only after generation seal.
- `proposal.py`: multiple source candidates using metadata, not hidden body
  content. This is **not yet multiple claim-specific semantic doubts**. Gains are
  uncalibrated; no invented reviewer capability differences.
- `runner.py`: bounded real online source fetch -> review -> linked patch -> next
  current state -> deterministic readout. No final LLM rewriting accepted state.
- `evaluate.py`: post-seal scores and integration counters, not a confirmatory
  efficacy test.

All five arms have the same source permissions and 12-call task cap. Actual
calls and text processing differ. Independent aggregation currently has no
additional source expansion or review; do not attribute E1 score differences
solely to communication or game allocation. Current iteration makes proposals
once and executes sequential reviews, not a complete two-round reallocation.

## Recorded runs

`experiments/results/unified_e1_smoke_v2/`: two-task first live smoke, retained
as executed, including invalid outputs and one unknown request. Do not retry it.

`experiments/results/unified_e1_v3/`: sealed 24-task integration; 286 receipts,
2 unknown, 14 accepted patches; missing review operands retained as failures.

`experiments/results/unified_e1_v4/`: current 24-task review-operand revision.
Initial requests are identical to v3 and successful raw responses reused.
First QA/Rec gate passed; inspect seal/run lock for full completion.
Old E1 v1 was preflight only. Version names are not evidence of completion.

## Operations (run from repository root)

Read-only source/lock/checkpoint inspection:

```bash
.venv-native-benchmark/bin/python -m experiments.resume_unified_e1 \
  --output experiments/results/unified_e1_v4
```

If the original collector is stopped and recorded source hashes still match,
resume under the same root (no automatic resending of unknown intents):

```bash
.venv-native-benchmark/bin/python -m experiments.resume_unified_e1 \
  --output experiments/results/unified_e1_v4 --resume
```

A completed generation seal makes this entry score-only, with no new API calls.
The native v3 runner also uses a nonblocking run lock and stable first-start
metadata. Do not run two collectors or delete intents to bypass the ledger.
Changes in method/source require a new version, not overwriting frozen results.

After the full generation seal:

```bash
.venv-native-benchmark/bin/python -m experiments.evidence_consensus.runner \
  --inputs experiments/results/evidence_consensus_e1_tasks_v1/tasks.json \
  --output experiments/results/unified_e1_v4 --score-only
.venv-native-benchmark/bin/python -m experiments.audit_unified_e1 \
  --output experiments/results/unified_e1_v4
.venv-native-benchmark/bin/python -m experiments.audit_unified_e1_state \
  --output experiments/results/unified_e1_v4
```

For a *new* authorized live batch, use an unused output root and `--run`. Current
transport has a batch cap of 1000 physical attempts / configured 5 CNY. No paid
network calls happen without `--run` (or explicit `--resume` of incomplete work).
Load keys through `.env.local`; never print or commit credentials.

## Evidence levels

Operational acceptance means the patch uses fetched, permitted evidence and a
linked model support/refutation opinion; it is not semantic certification.
Explicit abstention/no-change and malformed/unauthorized patches are different
outcomes and must be separated in reporting. Model accuracy/Recall use native
labels only after all task arms are sealed. All tasks are reused development
materials; QA web snapshots can be historical rather than current truth.

Next research work: semantic doubt proposals, training-side action-contribution
and damage calibration, complete iterative reallocation, independent agents'
strong source-processing control, homogeneous/heterogeneous models, then the
precommitted QA/ReDial/INSPIRED experiments in the approved research plan.
