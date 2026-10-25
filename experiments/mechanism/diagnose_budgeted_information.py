"""No-call sensitivity checks; do not refit against evaluation outcomes."""
from collections import Counter
from dataclasses import replace
from itertools import product
import json
import random

from experiments.core.mechanism.budgeted_information_sim import Contract, digest
from experiments.mechanism.replay_budgeted_information import (
    ROOT, OUT, WEIGHTS, confidence_cell, fit_table, load_observations, plan, source,
)
from experiments.mechanism.run_upgrade_pilot import freeze


def random_same_role_counts(observations, upgrades, decisions, outcomes):
    """Exact expectation over independent, within-role uniform subset selections.

    Calls are matched by role. Costs are logged actual per-slot costs, not made
    equal by assumption. Dependence across questions doesn't affect expectation.
    """
    rates = {}
    for role in range(3):
        eligible = sum(confidence_cell(panel[role]) != 'invalid' for panel in observations.values())
        k = sum(ds[role].action == 'U' for ds in decisions.values())
        rates[role] = k/eligible if eligible else 0.
    correct = 0.
    tokens = sum(o.initial_tokens for panel in observations.values() for o in panel)
    expected_calls = 0.
    for cid, panel in observations.items():
        probabilities = [rates[o.agent] if confidence_cell(o) != 'invalid' else 0. for o in panel]
        for actions in product((0, 1), repeat=3):
            mass = 1.
            votes = []
            for o, action, rate in zip(panel, actions, probabilities):
                mass *= rate if action else 1-rate
                votes.append(upgrades[(cid, o.agent)][0].answer if action else o.initial.answer)
            answer = next((c for c in panel[0].choices if votes.count(c) >= 2), 'abstain')
            correct += mass*int(answer == outcomes[cid])
        for o, rate in zip(panel, probabilities):
            tokens += rate*upgrades[(cid, o.agent)][1]
            expected_calls += rate
    return {'correct_expectation_not_observed_count': correct,
            'tokens_expectation_including_initial': tokens, 'expected_high_calls': expected_calls,
            'role_sampling_rates': rates}


def posterior_consistency_diagnostic(observations, upgrades, outcomes):
    """Necessary-posterior martingale diagnostic, NOT a calibrated model proof.

    Align each event to the initial selected choice. If both probabilities were
    nested coherent Bayesian posteriors, E[q1(event)-q0(event)|q0] = 0.
    Coarse cells and tiny opened data cannot verify that equality statistically.
    """
    from experiments.core.mechanism.budgeted_information_sim import valid_probability
    cells = {}
    for cid, panel in observations.items():
        for o in panel:
            high = upgrades[(cid, o.agent)][0]
            if not o.initial.valid(o.choices) or not high.valid(o.choices):
                continue
            key = str((o.agent, confidence_cell(o)))
            slot = cells.setdefault(key, [])
            j = o.choices.index(o.initial.answer)
            slot.append({'low_probability_of_initial_answer': o.initial.probabilities[j],
                         'high_probability_of_initial_answer': high.probabilities[j],
                         'initial_answer_correct': int(o.initial.answer == outcomes[cid])})
    return {key: {'n': len(rs),
                  'mean_low_probability': sum(r['low_probability_of_initial_answer'] for r in rs)/len(rs),
                  'mean_high_probability_same_event': sum(r['high_probability_of_initial_answer'] for r in rs)/len(rs),
                  'posterior_drift_same_event': sum(r['high_probability_of_initial_answer']-r['low_probability_of_initial_answer'] for r in rs)/len(rs),
                  'empirical_initial_accuracy': sum(r['initial_answer_correct'] for r in rs)/len(rs)}
            for key, rs in cells.items()}


def bootstrap_calibration(calibration, train_upgrades, train_labels, test, reps=256):
    """Question-cluster bootstrap; ONLY calibration outcomes can affect plans.

    No confidence interval validity claim on previously inspected samples.
    Records uncertainty of plug-in actions, not a new experiment.
    """
    rng = random.Random(20261023)
    ids = list(calibration)
    counts = {}
    original = {}
    for conditional in (False, True):
        original_table = fit_table(calibration, train_upgrades, train_labels, conditional)
        for weight in WEIGHTS:
            tag = f'conditional={conditional}:weight={weight}'
            original[tag] = plan(test, original_table, Contract(weight))
            counts[tag] = {'upgrades': [], 'changed_vs_original': [], 'slot_upgrade_counts': Counter()}
    for _ in range(reps):
        sample, high, labels = {}, {}, {}
        for j, cid in enumerate(rng.choices(ids, k=len(ids))):
            fake = str(j)
            sample[fake] = [replace(o, case_id=fake) for o in calibration[cid]]
            labels[fake] = train_labels[cid]
            for i in range(3): high[(fake, i)] = train_upgrades[(cid, i)]
        for conditional in (False, True):
            table = fit_table(sample, high, labels, conditional)
            for weight in WEIGHTS:
                tag = f'conditional={conditional}:weight={weight}'
                ds = plan(test, table, Contract(weight))
                selected = [f'{cid}/{i}' for cid, panel in ds.items() for i, d in enumerate(panel) if d.action == 'U']
                counts[tag]['upgrades'].append(len(selected))
                counts[tag]['changed_vs_original'].append(sum(d.action != original[tag][cid][i].action for cid, panel in ds.items() for i, d in enumerate(panel)))
                counts[tag]['slot_upgrade_counts'].update(selected)
    result = {}
    for tag, record in counts.items():
        ordered = sorted(record['upgrades'])
        result[tag] = {'repetitions': reps, 'high_call_count_min': ordered[0], 'high_call_count_max': ordered[-1],
            'high_call_count_median': ordered[len(ordered)//2],
            'mean_action_changes_among_72': sum(record['changed_vs_original'])/reps,
            'unstable_slots': sum(0 < n < reps for n in record['slot_upgrade_counts'].values()),
            'slot_upgrade_fraction': {cid: n/reps for cid, n in sorted(record['slot_upgrade_counts'].items())}}
    return result


def billing_ratio_sensitivity(observations, upgrades, decisions):
    """Symbolic equal input/output rate proxy, NOT an actual billing estimate.

    Let r = weak price/token divided by strong price/token. Hybrid costs
    r*T_low + T_selected_high versus T_all_high. Return the break-even r.
    Cache, distinct input/output tariffs and free tiers need actual receipts.
    """
    low = sum(o.initial_tokens for panel in observations.values() for o in panel)
    high = sum(v[1] for v in upgrades.values())
    selected = sum(upgrades[(cid, i)][1] for cid, ds in decisions.items() for i, d in enumerate(ds) if d.action == 'U')
    return {'initial_weak_tokens': low, 'selected_high_tokens': selected, 'direct_high_tokens': high,
            'weak_to_strong_unit_price_break_even': (high-selected)/low,
            'cheaper_than_direct_high_iff': 'r < reported break-even under flat per-backend token-rate assumption',
            'actual_price_or_billing_claim': False}


def main():
    calibration, train_upgrades, _ = load_observations(source.OUT)
    test, upgrades, _ = load_observations(ROOT/'experiments/results/resource_allocation/routing_holdout_validation_v1')
    train_labels = {r['id']: r['label'] for r in json.loads((ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())}
    test_labels = {r['id']: r['label'] for r in json.loads((ROOT/'experiments/results/resource_allocation/routing_holdout_data_v1/private_labels.json').read_text())}
    bootstrap = bootstrap_calibration(calibration, train_upgrades, train_labels, test)
    frozen = json.loads((OUT/'plans_before_evaluation.json').read_text())
    from experiments.core.mechanism.budgeted_information_sim import Decision
    random_controls = {tag: random_same_role_counts(test, upgrades,
        {cid: [Decision(**d) for d in ds] for cid, ds in cases.items()}, test_labels)
        for tag, cases in frozen['plans'].items()}
    billing = {tag: billing_ratio_sensitivity(test, upgrades,
        {cid: [Decision(**d) for d in ds] for cid, ds in cases.items()})
        for tag, cases in frozen['plans'].items()}
    result = {'scope': 'opened-data no-call uncertainty/consistency diagnosis, no selection of winning contract',
        'new_api_calls': 0, 'bootstrap_seed': 20261023, 'question_cluster_resampling': True,
        'bootstrap_uses_evaluation_labels': False,
        'bootstrap_calibration': bootstrap, 'random_same_role_counts': random_controls,
        'symbolic_billing_sensitivity': billing,
        'posterior_consistency': {'calibration': posterior_consistency_diagnostic(calibration, train_upgrades, train_labels),
                                  'opened_evaluation': posterior_consistency_diagnostic(test, upgrades, test_labels)},
        'limits': ['bootstrap variability is not independent replication',
                  'nonzero posterior drift may reflect sampling, miscalibration or non-nested information; not a causal attribution',
                  'cannot replace outcome-based gain by squared differences of two uncalibrated LLM forecasts',
                  'data and this diagnostic already opened; no confirmatory p-values']}
    freeze(OUT/'diagnostics.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('bootstrap_calibration', 'random_same_role_counts')}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__': main()
