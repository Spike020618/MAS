# MovieLens slate-choice study

This is an independent study, with a fixed protocol in `protocol.json`. No
historical QA outputs, conversational labels, system rankings or LLM prompts are
used. No network access or model API is required.

## Task

- Global cutoff: 2016-01-01 UTC. Training priors and personal histories use only
  timestamps strictly before this instant; boundary ties go to future labels.
- Users partition deterministically: 80% training, 10% development, 10% reserved
  test. Item priors use training users only. Personal dev/test fitting uses only
  each user's permitted pre-cutoff history.
- Eligible catalog: known year <=2015, at least 20 pre-cutoff training-user
  ratings. Static genre/year metadata is not claimed to reconstruct historical
  commercial availability.
- Activity eligibility: at least 40 catalog-history ratings and at least five
  future observations; the future rating values do not determine eligibility.
- Sample: lowest deterministic user hashes, up to 1,000 dev and 2,000 reserved
  test users. This is a retrospective active-user sample, not population-random
  recommendation requests.
- Visibility budgets: last 5/20/40 historical ratings. All past seen item IDs
  remain known to every method, which isolates rating preference visibility
  rather than watched-item discovery.

## Shared information and actions

The point model fits ridge residuals over a training-only item-mean prior and
static genre features. Bootstrap resampling of the same visible history produces
32 alternative predictions; this is not a calibrated Bayesian posterior.

The same candidate pool is supplied to every method at each visibility budget:
top-200 point predictions, top-100 priors, and top-10 priors per genre, excluding
seen items. No target injection, future-item pool or label-assisted retrieval.

Four ordinary baselines: point Top-K, MMR, greedy expected maximum utility across
bootstrap models, and greedy finite-hypothesis minimax regret. Final display
order within each selected set is always the same point-score ordering.

This initial genre model is not a state-of-the-art collaborative recommender.
The first experiment diagnoses whether this common decision representation has
measurable effects. It does not establish superiority over matrix factorization
or modern collaborative models, and no equilibrium/new mechanism is claimed.
Greedy minimax regret is not an exact SMR solver; small exhaustive tests expose
this distinction rather than concealing it.

## Evaluation

Future ratings are stored separately and read only after predictions. The
primary development metric is paired recall@5 of observed future positives
(rating >=4), including all eligible sampled users. Users with no observed
positive are retained at zero and counted separately.

Future positives outside the time-restricted catalog remain in the primary
denominator; catalog and candidate ceilings must be reported. A zero observed
match does not establish dislike. Missing future ratings are not inferred user
utility, and observed-rating coverage is not a satisfaction metric. The future
observation window extends to dataset end; taste drift and exposure selection
remain limitations, not effects solved by this protocol.

Development paired bootstrap intervals are exploratory, unadjusted for multiple
comparisons, and not confirmatory significance evidence. Methods share users;
5/20/40-budget cases are not independent samples. The reserved test is prepared
but the development runner has no test-scoring option and never deserializes
test labels. Hashing a reserved file only verifies its integrity.

The timing in each checkpoint covers the whole four-method prediction pipeline,
not isolated per-method runtime; it cannot support an algorithm-speed claim.

## Run

```sh
python3 -m experiments.slate_choice.build
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -m experiments.slate_choice.run_dev --output experiments/results/slate_choice_v1/development_checked
python3 -m unittest experiments.tests.test_slate_choice -v
```

All outputs are under `experiments/results/slate_choice_v1/`, separate from other
experiments. Build has a completed-prior checkpoint. Development has one complete
checkpoint per user and history budget; incomplete/corrupt JSON is an error,
not silently accepted. Dependency changes require a new version directory. No
protocol or method is tuned against development results in this run.

## Collaborative and exact controls

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -W error -m experiments.slate_choice.train_cf
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -W error -m experiments.slate_choice.run_controls
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -W error -m experiments.slate_choice.verify_controls
```

`controls_protocol.json` fixes a 32-dimensional training-only residual SVD and
same-input four-method CF comparison. Exact diagnostics use the first 128 dev
users by the fixed sampling hash, all history budgets and both representations.
The threshold set-cover MILP certifies deterministic finite-hypothesis SMR only,
not a mixed game equilibrium or real-user optimality. Timeouts retain bounds.
Point-score sum breaks objective ties, so external changes at equal regret must
not be attributed to strict regret improvement. All control artifacts remain in
`experiments/results/slate_choice_v1/controls/`; no reserved-test scoring occurs.
