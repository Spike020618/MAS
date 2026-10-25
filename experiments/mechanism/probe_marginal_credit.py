"""Exploratory blocked leave-one-question-out replay of delayed difference credit.

72 old independent action reports, six already-exposed development questions.
Never an online equilibrium, held-out benchmark, or new accuracy evidence.
"""
from itertools import product
import json
from pathlib import Path

from experiments.core.mechanism.marginal_credit import seal, settle, VerifiedOutcome

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/results/mechanism/stackelberg/source_sharing_pilot_v3'
PROFILES = list(product((0, 1), repeat=3))
WEIGHTS = [0, .01, .1]


def normalized(row):
    if not row['schema_valid'] or row['finish_reason'] != 'stop':
        raise ValueError('pilot requires complete reports; no silent substitution')
    report = row['report']
    return {'A': report['A'], 'E': [report['E']], 'I': report['I'], 'C': report['C']}


def table_for(cases, lookup, weight, score_mode='accuracy'):
    if not cases: raise ValueError('training cases required')
    table = {}
    for profile in PROFILES:
        utilities = [0.] * 3; quality = tokens = potential = 0.
        for case in cases:
            qid = case['dataset']+'/'+case['question_id']
            for repeat in range(2):
                rows = [lookup[qid, i, 'R' if profile[i] else 'D', repeat] for i in range(3)]
                reports = {str(i): normalized(r) for i, r in enumerate(rows)}
                usage = [(r['usage'] or {})['total_tokens'] for r in rows]
                result = settle(reports, {str(i): weight*u/1000 for i, u in enumerate(usage)},
                    1., seal(reports), VerifiedOutcome(case['ground_truth'], 'development-benchmark-label'), score_mode)
                quality += result['decision'] == case['ground_truth']; tokens += sum(usage); potential += result['potential']
                for i in range(3): utilities[i] += result['utilities'][str(i)]
        n = 2 * len(cases)
        table[profile] = {'quality': quality/n, 'tokens': tokens/n, 'potential': potential/n,
                          'utilities': [u/n for u in utilities]}
    return table


def solve(table):
    profile = (0, 0, 0); events = []; visited = {profile}
    while True:
        changed = False
        for i in range(3):
            alternative = list(profile); alternative[i] = 1-alternative[i]; alternative = tuple(alternative)
            gain = table[alternative]['utilities'][i] - table[profile]['utilities'][i]
            if gain > 1e-10:
                if alternative in visited: raise ValueError('strict improvement cycle')
                events.append({'agent': i, 'before': profile, 'after': alternative, 'gain': gain})
                visited.add(alternative); profile = alternative; changed = True
        if not changed: break
    regrets = []
    identity_error = 0.
    equilibria = []
    for p in PROFILES:
        gains = []
        for i in range(3):
            other = list(p); other[i] = 1-other[i]; other = tuple(other)
            gain = table[other]['utilities'][i]-table[p]['utilities'][i]
            identity_error = max(identity_error, abs(gain-(table[other]['potential']-table[p]['potential'])))
            gains.append(max(0, gain))
        if max(gains) <= 1e-10: equilibria.append(list(p))
        if p == profile: regrets = gains
    return {'profile': list(profile), 'events': events, 'terminal_regrets': regrets,
            'pure_equilibria': equilibria, 'max_potential_identity_error': identity_error}


def run():
    cases = json.loads((BASE/'private_scoring.json').read_text())
    calls = [json.loads(line) for line in (BASE/'calls.jsonl').read_text().splitlines() if line.strip()]
    if len(calls) != 72: raise ValueError('expected complete 72-response development panel')
    lookup = {(r['question_id'], r['agent'], r['action'], r['repeat']): r for r in calls}
    if len(lookup) != 72: raise ValueError('duplicate reports')
    folds = []; summaries = []
    for score_mode, weight in product(['accuracy', 'brier'], WEIGHTS):
        quality = tokens = 0.
        controls = {name: {'quality_sum': 0., 'tokens_sum': 0.} for name in
                    ['all_D', 'all_R', 'only_R_0', 'only_R_1', 'only_R_2', 'uniform_profile_mean']}
        for index, case in enumerate(cases):
            training = [c for j, c in enumerate(cases) if j != index]
            selected = solve(table_for(training, lookup, weight, score_mode))
            # Only after selection: reveal this question's label to score replay.
            evaluation = table_for([case], lookup, weight, score_mode)
            result = evaluation[tuple(selected['profile'])]
            quality += result['quality']; tokens += result['tokens']
            train_ids = [c['dataset']+'/'+c['question_id'] for c in training]
            qid = case['dataset']+'/'+case['question_id']
            if qid in train_ids: raise ValueError('question split leakage')
            folds.append({'score_mode': score_mode, 'token_weight': weight, 'test_question': qid,
                'training_questions': train_ids, 'selection': selected,
                'test_quality': result['quality'], 'test_report_tokens': result['tokens']})
            profiles = {'all_D': (0, 0, 0), 'all_R': (1, 1, 1), 'only_R_0': (1, 0, 0),
                        'only_R_1': (0, 1, 0), 'only_R_2': (0, 0, 1)}
            for name, profile in profiles.items():
                controls[name]['quality_sum'] += evaluation[profile]['quality']
                controls[name]['tokens_sum'] += evaluation[profile]['tokens']
            controls['uniform_profile_mean']['quality_sum'] += sum(r['quality'] for r in evaluation.values())/8
            controls['uniform_profile_mean']['tokens_sum'] += sum(r['tokens'] for r in evaluation.values())/8
        summaries.append({'score_mode': score_mode, 'token_weight': weight, 'n_questions': len(cases),
            'selected_quality': quality/len(cases), 'selected_mean_report_tokens': tokens/len(cases),
            'controls': {name: {'quality': v['quality_sum']/len(cases),
                        'mean_report_tokens': v['tokens_sum']/len(cases)} for name, v in controls.items()}})
    return {'scope': 'exploratory old development potential-report replay; blocked crossfit is not new independent test',
        'no_new_api_calls': True, 'actual_old_api_responses': len(calls),
        'payment': 'B/(2n) * [1 + team_score - team_score_without_i]; both accuracy and Brier variants disclosed',
        'budget': 1, 'cost': 'listed weights * report_tokens/1000, excludes source preparation; not real payout',
        'policy': 'fixed-order explicit strict improvement on training means, never per-test-label optimal choice',
        'all_weights_reported_no_winner_selected': True, 'summary': summaries, 'folds': folds,
        'limitations': ['six previously exposed questions, two sampled outputs/action; sampling error unknown',
            'D/R are old source actions, not original model-tier actions',
            'trusted post-report labels required for settlement, never supplied as evidence',
            'payoff estimation uses labeled historical tasks; no LLM autonomous action adaptation tested',
            'classical difference-reward theory, not a new general theorem or unique AEIC advantage',
            'counterfactual deletes one sealed input only; no causal peer-reaction attribution']}


if __name__ == '__main__':
    result = run()
    out = ROOT/'experiments/results/mechanism/synthesis/marginal_credit_probe_2026-10-21.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
