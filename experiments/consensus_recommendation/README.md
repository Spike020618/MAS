# Evidence-grounded multi-agent recommendation experiment

This is an isolated research implementation, not the running movie system and
not a faithful reproduction of MACRec, AgentCF or EyeCRS. Recommendations are
scored against future ReDial questionnaire-positive movie targets, not real-user
satisfaction or complete utility.

From `/Users/spike/code/MAS`:

```sh
python3 -m experiments.consensus_recommendation.data
python3 -m experiments.consensus_recommendation.accept
python3 -m experiments.consensus_recommendation.runner --stage engineering
```

The last command is preparation only. Network execution requires the explicit
`--execute-authorized` flag and an accepted data/code checkpoint. The collector
uses the official DeepSeek endpoint and the local `DEEPSEEK_API_KEY`,
`DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL` settings; never put keys in command arguments.
No Qwen model is used. The actual returned model identity is saved for every call.

```sh
python3 -m experiments.consensus_recommendation.runner --stage engineering --execute-authorized
python3 -m experiments.consensus_recommendation.evaluate --stage engineering
python3 -m experiments.consensus_recommendation.runner --stage pilot --execute-authorized
python3 -m experiments.consensus_recommendation.evaluate --stage pilot
python3 -m experiments.scripts.verify_consensus_recommendation
```

Do not execute these network commands unless the research run has been authorized.
No command here automatically evaluates reserved labels or launches a formal test.

## Inputs and evaluation

- Only the original ReDial **train** source is used for this experiment.
- Participant IDs are hashed to partitions, and cross-partition conversations
  are removed. Worker IDs are **not** sent to the model or interpreted as stable
  personal-preference identities across conversations.
- Cutoffs depend only on roles and movie-token positions, not future liking labels.
- Inputs and labels are separate files. The provider prompt uses a field whitelist.
- The common candidate pool uses training-only context transfer, neutral
  cooccurrence, prior frequency and lexical RRF. No target is injected into it.
- All paths have access to the same original evidence. Independent analyses are
  cached and shared by R3/R4/R5. Actual paid costs and full logical path costs differ.
- R0 (and lexical/example controls), R1 one-shot, R2 six serial calls, R3 independent
  aggregation, R4 generic review, R5 claim-targeted review are reported separately.

## Execution and process validity are different

`validate` strictly audits claims, citations and updates. `collect_parse` requires
an executable five-item ranking but records report errors separately. A fabricated
citation remains visible and erroneous; the parser does not repair it. Otherwise
field-format errors could turn a valid ranking into a zero and manufacture an
accuracy difference between protocols. Neither function proves semantic support.

## Immutable revisions

Results live under `experiments/results/consensus_recommendation_v1/`:

- `collection_dns_blocked/`: sandbox DNS failures; no successful response records.
- `collection/engineering/`: original engineering interface and source snapshot.
- `collection_interface_v2/engineering/`: one uniform interface repair and snapshot.
- `collection_execution_v3/`: ranking/process separation adopted before the pilot.

The v3 engineering replay can reuse v2 records only when the original request
messages, provider configuration and generation settings match exactly and the
source record passes integrity checks. Reuse does not count as another API call.
Different prompts cannot silently reuse previous responses. Unknown network/billing
intents are not retried automatically. Changed immutable artifacts require a new
version namespace, not deletion of the old evidence.

## Statistical and theoretical boundaries

The 12 engineering examples are not effect evidence. The 48-case pilot is
development-only, with seeker-cluster intervals and explicit remaining dependency.
Published work uses different candidate/evaluation protocols; raw scores cannot
be compared across them.

`game.py` solves a **declared finite matrix** randomized zero-sum game. Its bounds
are neither a full-catalog certificate nor evidence that LLMs are strategic,
incentive compatible, or in a Stackelberg equilibrium. Synthetic unit tests do
not certify recommendation quality.
