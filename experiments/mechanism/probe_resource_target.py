"""Offline resource-necessity and collective-decision headroom diagnostics.

Uses public acquisition metadata for allocation. Labels are used only AFTER
recorded decisions, for descriptive diagnosis, never to allocate resources.
"""
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path

from experiments.core.mechanism.source_sharing import SourceSharingGame

ROOT = Path(__file__).resolve().parents[2]


def metrics(profile, agents, bundles, weights, costs):
    selected = [set(bundles[a]) if profile[i] else set() for i, a in enumerate(agents)]
    union = set().union(*selected)
    return {'coverage_weight': sum(weights[s] for s in union),
            'source_acquisitions': sum(map(len, selected)),
            'unique_sources': len(union),
            'duplicate_acquisitions': sum(map(len, selected)) - len(union),
            'modeled_cost': sum(costs[a][profile[i]] for i, a in enumerate(agents))}


def central_cover(agents, bundles, weights, costs, cost_cap):
    """Max public weighted coverage under same modeled cost cap; no labels."""
    candidates = []
    for p in product((0, 1), repeat=len(agents)):
        m = metrics(p, agents, bundles, weights, costs)
        if m['modeled_cost'] <= cost_cap + 1e-10:
            candidates.append((p, m))
    if not candidates:
        raise ValueError('infeasible public cost cap')
    return min(candidates, key=lambda item: (-item[1]['coverage_weight'],
        item[1]['modeled_cost'], item[1]['duplicate_acquisitions'], item[0]))


def greedy_cover(agents, bundles, weights, costs, cost_cap):
    profile = [0] * len(agents)
    if metrics(profile, agents, bundles, weights, costs)['modeled_cost'] > cost_cap + 1e-10:
        raise ValueError('infeasible baseline')
    while True:
        current = metrics(profile, agents, bundles, weights, costs)
        candidates = []
        for i, agent in enumerate(agents):
            if profile[i]: continue
            trial = profile.copy(); trial[i] = 1
            m = metrics(trial, agents, bundles, weights, costs)
            gain = m['coverage_weight'] - current['coverage_weight']
            fee = m['modeled_cost'] - current['modeled_cost']
            if gain > 1e-10 and m['modeled_cost'] <= cost_cap + 1e-10:
                score = gain / fee if fee > 0 else float('inf')
                candidates.append((-score, -gain, str(agent), trial))
        if not candidates: return profile
        profile = min(candidates, key=lambda item: item[:3])[3]


def allocation_probe():
    directory = ROOT / 'experiments/results/mechanism/stackelberg/locked_source_game_test_v1'
    protocol = json.loads((directory / 'protocol.json').read_text())
    agents, costs = protocol['agents'], protocol['costs']
    rows = []
    for path in sorted((directory / 'commitments').glob('*.json')):
        row = json.loads(path.read_text())
        bundles = row['bundles']
        ids = sorted(set().union(*(set(b) for b in bundles.values())))
        weights = {s: 1/len(ids) for s in ids}
        game = SourceSharingGame(agents, bundles, weights, costs, row['budget'])
        profile = row['profile']
        if game.run_local_updates(protocol['initial'], protocol['order'])['terminal']['profile'] != profile:
            raise ValueError('frozen allocation mismatch')
        baseline = metrics(profile, agents, bundles, weights, costs)
        cap = baseline['modeled_cost']
        greedy = greedy_cover(agents, bundles, weights, costs, cap)
        optimal, best = central_cover(agents, bundles, weights, costs, cap)
        fixed = [0, 0, 1] if costs['2'][1] <= cap + 1e-10 else [0, 0, 0]
        rows.append({'question_id': row['question_id'], 'budget': row['budget'],
            'game_profile': profile, 'game': baseline,
            'greedy_profile': greedy, 'greedy': metrics(greedy, agents, bundles, weights, costs),
            'central_profile': list(optimal), 'central': best,
            'fixed_agent_2_profile': fixed,
            'fixed_agent_2': metrics(fixed, agents, bundles, weights, costs),
            'greedy_exact_same_profile': greedy == profile,
            'central_exact_same_profile': list(optimal) == profile})
    summary = {}
    for budget in [0, 1]:
        subset = [r for r in rows if r['budget'] == budget]
        summary[str(budget)] = {'n': len(subset),
            'game_profiles': dict(Counter(str(r['game_profile']) for r in subset)),
            'greedy_exact_same_profiles': sum(r['greedy_exact_same_profile'] for r in subset),
            'central_exact_same_profiles': sum(r['central_exact_same_profile'] for r in subset),
            'fixed_agent_2_exact_same_profiles': sum(r['fixed_agent_2_profile'] == r['game_profile'] for r in subset),
            'game_mean_cost': sum(r['game']['modeled_cost'] for r in subset)/len(subset),
            'game_mean_coverage': sum(r['game']['coverage_weight'] for r in subset)/len(subset)}
    return {'scope': 'public metadata and modeled costs only; not API-cost or accuracy comparison',
            'cost_cap': 'per-question cost spent by the recorded game; no outcome labels',
            'summary': summary, 'rows': rows}


def decision_diagnosis(turns, truth):
    votes = [t['stance'] for t in turns]
    counts = Counter(v for v in votes if v in ('yes', 'no'))
    decision = None if not counts else counts.most_common()[0][0]
    if len(counts) == 2 and counts['yes'] == counts['no']: decision = None
    any_correct = truth in votes
    correct = decision == truth
    return {'majority_correct': correct, 'any_agent_correct': any_correct,
        'correct_answer_not_selected': any_correct and not correct,
        'no_correct_agent': not any_correct,
        'unanimously_wrong': len(set(votes)) == 1 and not any_correct,
        'different_stances': len(set(votes)) > 1}


def decision_probe():
    frozen = ROOT / 'experiments/results/mechanism/frozen'
    manifest = json.loads((frozen / 'manifest.json').read_text())
    aggregates = defaultdict(Counter)
    transitions = defaultdict(Counter)
    for suite in manifest['required_suites']:
        for line in (frozen / suite / 'question_level.jsonl').read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            rounds = sorted({t['round_id'] for t in row['rounds']})
            stages = {}
            for stage, rid in [('first', rounds[0]), ('last', rounds[-1])]:
                turns = [t for t in row['rounds'] if t['round_id'] == rid]
                result = decision_diagnosis(turns, row['ground_truth'])
                stages[stage] = result
                key = (suite, row['condition'], row['dataset'], stage)
                aggregates[key]['n'] += 1
                aggregates[key].update(result)
            before, after = stages['first']['majority_correct'], stages['last']['majority_correct']
            key = (suite, row['condition'])
            transitions[key][('correct' if before else 'wrong') + '_to_' +
                             ('correct' if after else 'wrong')] += 1
    return {'scope': 'descriptive same-round recorded-answer headroom; gold oracle is not deployable and not achievable accuracy',
        'rows': [{'suite': k[0], 'condition': k[1], 'dataset': k[2], 'stage': k[3], **dict(v)}
                 for k, v in sorted(aggregates.items())],
        'first_last_transitions': [{'suite': k[0], 'condition': k[1], **dict(v)}
                                  for k, v in sorted(transitions.items())]}


def main():
    allocation = allocation_probe()
    decisions = decision_probe()
    result = {'allocation': allocation, 'decisions': decisions,
        'no_api_calls': True, 'labels_used_for_allocation': False}
    out = ROOT / 'experiments/results/mechanism/synthesis/resource_target_probe_2026-10-21.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(allocation['summary'], indent=2))
    for name in ['independent__no_mechanism@0.2', 'independent__static_aeic@0.2']:
        for stage in ['first', 'last']:
            rows = [r for r in decisions['rows'] if r['suite'] == 'formal_v10' and r['condition'] == name and r['stage'] == stage]
            print(name, stage, dict(sum((Counter({k: v for k, v in r.items() if isinstance(v, int)}) for r in rows), Counter())))


if __name__ == '__main__': main()
