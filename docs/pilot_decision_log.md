# AEIC-MD Pilot Decision Log

## Evidence status

Only results produced by the two-stage follower-response collector are eligible
for the new study. Earlier one-call pilots and topology-era outputs are retained
as diagnostics or prior work, not pooled with the confirmatory experiment.

## Pilot revisions

### v6-v8: homogeneous execution

- The mechanism changed declared and executed effort, but did not improve
  question-level decision quality.
- Static and contract mechanisms often saturated the high-effort action.
- Dynamic adaptation over-treated copy pressure and was not superior to the
  static AEIC contract.
- Decision: retain the homogeneous setting as a mechanism-compliance result;
  do not use it as the primary quality claim.

### v9: resource-tiered execution, instruction-following action choice

- `low_effort` used `glm-4-flash`, ordinary AEIC used `deepseek-chat`, and
  `verify` used a draft and revision from `deepseek-reasoner`.
- On MMLU-Hard dev10, AEIC contract increased actual high effort from 0.58 to
  1.00 and accuracy from 0.80 to 0.90. The accuracy comparison was not
  significant at ten paired questions.
- Reward scales 0.05/0.2/0.5 produced high-effort rates 1.00/1.00/0.98.
- Decision: fail the registered saturation and monotonic-response gates. Keep
  v9 as a failed configuration; do not expand it to the formal test set.

### v10: resource-tiered execution, rational follower choice

The action-choice prompt was changed before any formal-test access. It now asks
the follower to maximize expected credit minus visible action cost and warns
against choosing work merely because the contract mentions AEIC. Execution and
settlement were unchanged.

MMLU-Hard dev10 dose calibration:

| Reward scale | High-effort rate | Majority accuracy | Mean tokens/agent-round |
|---:|---:|---:|---:|
| 0.05 | 0.44 | 0.80 | 842.32 |
| 0.20 | 0.50 | 0.70 | 886.14 |
| 0.50 | 0.60 | 0.80 | 891.24 |

The registered mechanical pilot checks pass: action choices are not saturated,
the dose response is monotone, completed parse rate is 1.00, completed-record
API failure rate is 0, and cost logging is complete. This is an identifiability
gate, not evidence of decision-quality improvement.

## Frozen decisions before expansion

1. v10 is the only action-choice prompt eligible for further pilot collection.
2. Use 0.20 as the central calibration level because it lies between two
   pre-registered held-out intensities; do not select a level by dev accuracy.
3. Resource-tiered and homogeneous execution are separate estimands and must
   never be pooled.
4. Expand pilot gates on dev splits before touching formal test splits.
5. The primary claim requires question-level decision quality or cost-adjusted
   quality; effort compliance alone is a manipulation check.
6. Adaptive/Stackelberg enters formal collection only if it improves a static
   contract under a pre-specified strategic or longitudinal condition. Current
   social/copy-pressure pilots do not satisfy that requirement.

## Next locked collection

- Complete v10 dev pilots at the registered sample size on StrategyQA,
  MMLU-Hard, and TruthfulQA.
- Compare no mechanism, outcome-only, static AEIC, AEIC contract at central
  intensity, and adaptive AEIC under the same real resource tiers.
- Apply the existing gates without configuration selection by accuracy.
- Only passing conditions advance to the frozen test split.

## v10 dev50 mechanism decision (three real resource tiers)

The registered three-domain pilot uses five agents, three rounds, and 50 paired
dev questions per domain. Low effort executes on GLM-4-Flash, ordinary AEIC on
DeepSeek Chat, and verification on DeepSeek Reasoner. The following decisions
were made without accessing the formal test split.

### Outcome-only: stop as a candidate method

Across 150 paired questions, outcome-only raised high-effort choice only from
0.58% to 3.16%, changed accuracy from 0.720 to 0.727 (paired p=1.0), and added
19.5 tokens per agent-round. Retain it as the moral-hazard baseline.

### Static AEIC: advance as the candidate method

Static AEIC is the only mechanism with a decision-quality signal that survives
a paired pilot comparison. On MMLU-Hard it raised accuracy from 0.64 to 0.88
versus no mechanism (Holm p=0.0037) and from 0.62 to 0.88 versus outcome-only
(Holm p=0.0029), while inducing 74.5% actual high-effort choices. StrategyQA
and TruthfulQA show directionally higher quality but are underpowered at 50
questions. Advance static AEIC to the frozen formal test split against no
mechanism and outcome-only. Its increased resource cost must be reported.

### Marginal-contribution AEIC contract: stop as a candidate method

The contract reliably changes actions relative to no mechanism, but loses to
static AEIC in all completed domains. Relative to static AEIC:

- StrategyQA: effort -20.9 pp, accuracy -4 pp, no token saving;
- TruthfulQA: effort -18.8 pp, accuracy -2 pp, +47.5 tokens;
- MMLU-Hard: effort -26.8 pp, accuracy -10 pp, -46.6 tokens.

The mechanism can reward low-cost novel phrasing and can overuse expensive
verification. Retain it as a preregistered failure/boundary ablation; do not
promote it to a primary formal method.

### Adaptive/Stackelberg: stop as a candidate method

Public offers genuinely change with reputation, so this is not a hidden-label
null treatment. Nevertheless, adaptive offers have not improved the fixed
marginal-contribution contract:

- StrategyQA: effort -0.9 pp, accuracy +2 pp (paired p=1.0), +12.9 tokens;
- TruthfulQA: effort -3.1 pp, accuracy -2 pp, +45.6 tokens.

The MMLU-Hard arm remains in collection only to complete the frozen pilot
matrix. Adaptive will not enter the confirmatory main test unless a separately
pre-specified strategic-information condition demonstrates a gain over static
AEIC; such a result would be reported as a boundary interaction, not used to
rewrite the independent-task primary endpoint.
