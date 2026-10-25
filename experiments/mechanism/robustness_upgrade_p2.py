"""Post-hoc sanity checks; do not rewrite the registered comparison or gate."""
import argparse
from itertools import product
import json
from pathlib import Path

from experiments.core.mechanism.upgrade_game import digest, improve
from experiments.core.mechanism.upgrade_value import contract_for, PROFILES
from experiments.mechanism.calibrate_upgrade_v2 import metric, rotation, scored_rows
from experiments.mechanism.run_upgrade_pilot import freeze, read_rows


def c_only(row, choices):
    body = json.loads(row['response']['body'])
    if row['response']['http_status'] != 200 or body['choices'][0]['finish_reason'] != 'stop':
        return 'abstain'
    try:
        value = json.loads(body['choices'][0]['message']['content'])
        vote = value.get('C')
    except (ValueError, TypeError, AttributeError):
        return 'abstain'
    return vote if vote in choices else 'abstain'


def majority(votes, choices):
    return next((choice for choice in choices if votes.count(choice) > len(votes) / 2), 'abstain')


def assess(directory, labels_path, model):
    rows = scored_rows(directory, labels_path)
    cases = json.loads((directory / 'public_cases.json').read_text())
    labels = {r['id']: r['label'] for r in json.loads(labels_path.read_text())}
    raw = {r['key']: r for r in read_rows(directory / 'responses.jsonl')}
    mechanism_profiles = [tuple(improve(contract_for(model, r['initial'], .02))['actions']) for r in rows]
    m = metric(rows, mechanism_profiles)
    simple_profiles = [rotation(r, 2 if len({p['C'] for p in r['initial']}) > 1 else 0) for r in rows]
    simple = metric(rows, simple_profiles)
    c_scores = []
    for case in cases:
        prefix = digest(case)[:20]
        votes = [c_only(raw[prefix + '/initial/' + str(i)], case['choices']) for i in range(3)]
        c_scores.append(int(majority(votes, case['choices']) == labels[case['id']]))
    changed_ids = [r['id'] for r, score in zip(rows, c_scores) if score > r['scores'][0]]
    mechanism_corrected = [r['id'] for r, score in zip(rows, m['question_scores']) if score > r['scores'][0]]
    # Evaluate the resource-count rule over every conclusion equality pattern,
    # not only over the observed development cases. No gold labels involved.
    exhaustive = []
    for votes in product(('A', 'B', 'C', 'abstain'), repeat=3):
        initial = [{'A': '', 'E': [], 'I': '', 'C': v} for v in votes]
        action = improve(contract_for(model, initial, .02))['actions']
        k_simple = 2 if len(set(votes)) > 1 else 0
        exhaustive.append(sum(action) == k_simple)
    c_baseline = {'correct': sum(c_scores), 'n': len(rows), 'upgrades': 0,
                  'deployed_calls': 3 * len(rows), 'tokens': sum(r['initial_tokens'] for r in rows),
                  'scope': 'initial valid C extraction only; does NOT pretend missing AEIC fields exist'}
    matches = c_scores == m['question_scores']
    review = {'registered_effect_gate': 'preserved_in_p2_comparison_v2.json_not_rewritten',
              'scope': 'post-hoc diagnostic, not an independent confirmatory comparison',
              'selected_lambda_02_is_counterfactual_replay': True,
              'actual_online_development_lambda': .01,
              'selected_replay': {k: m[k] for k in ('correct', 'n', 'upgrades', 'deployed_calls', 'tokens')},
              'initial_C_only_baseline': c_baseline,
              'same_question_outcomes_as_C_only': matches,
              'mechanism_corrected_ids': mechanism_corrected,
              'already_recoverable_from_initial_C_ids': changed_ids,
              'all_mechanism_corrections_already_present_in_initial_C': set(mechanism_corrected).issubset(changed_ids),
              'simple_disagreement_two_upgrade_baseline': {k: simple[k] for k in ('correct', 'n', 'upgrades', 'deployed_calls', 'tokens')},
              'selected_resource_counts_match_simple_rule_on_all_64_patterns': all(exhaustive),
              'core_semantic_increment': 'NOT_ESTABLISHED',
              'formal_300_decision': 'DO_NOT_START_pending_method_validity_not_just_numerical_gate',
              'reason': 'Observed two-question gain is already recoverable from initial C without upgrading; '
                        'same-k random reaches the same expected correctness. Do not promote parser recovery into semantic information production.'}
    return review


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.collection, args.labels, json.loads(args.model.read_text()))
    freeze(args.output, result)
    print(json.dumps(result))
