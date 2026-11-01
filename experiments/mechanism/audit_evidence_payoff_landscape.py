"""Exact description of OBSERVED action tables, not population game equilibria."""
from itertools import product
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism import repeat_open_evidence_solver as repeat
from experiments.mechanism.run_upgrade_pilot import freeze

OUT = p.ROOT / 'experiments/results/redesign_v1/payoff_landscape_v1'


def pure_best_response_profiles(table, n_actions=3):
    profiles = set(product(range(n_actions), repeat=2))
    if set(table) != profiles: raise ValueError('incomplete empirical table')
    # Common observed payoff; each unilateral comparison holds the other fixed.
    return [a for a in sorted(profiles) if all(table[a] >= table[(b, a[1])] and table[a] >= table[(a[0], b)] for b in range(n_actions))]


def main():
    p.verify(); repeat.verify(); OUT.mkdir(parents=True, exist_ok=True)
    labels = p.base.read(p.OUT / 'private_labels.json'); first = p.base.read(p.OUT / 'sealed_outputs.json')
    second = p.base.read(repeat.OUT / 'sealed_outputs.json')
    table = {}; by_arm = {}; synergy = {}; disagreements = []
    for i, j in product(range(3), repeat=2):
        arm = 's' + str(i) if i == j else 'p' + ''.join(map(str, sorted((i, j))))
        score = [p.answer_score(first[c][arm]['answer'], labels[c])['em'] for c in labels]
        table[i, j] = sum(score) / len(score)
        by_arm[arm] = {'em_correct': sum(score), 'n': len(score),
            'solver_tokens': sum(first[c][arm]['usage']['total_tokens'] for c in labels),
            'planner_tokens': 0 if arm == 's0' else sum(first[c]['plan']['usage']['total_tokens'] for c in labels)}
    for arm in repeat.ARMS:
        i, j = int(arm[1]), int(arm[2]); hits = []
        for c in labels:
            pair = p.answer_score(first[c][arm]['answer'], labels[c])['em']
            best_single = max(p.answer_score(first[c]['s' + str(k)]['answer'], labels[c])['em'] for k in (i, j))
            if pair > best_single: hits.append(c)
            if pair != p.answer_score(second[c][arm]['answer'], labels[c])['em']: disagreements.append([c, arm])
        synergy[arm] = hits
    frontier = []
    for arm, cell in by_arm.items():
        tokens = cell['solver_tokens'] + cell['planner_tokens']
        if not any(other['em_correct'] >= cell['em_correct'] and other['solver_tokens'] + other['planner_tokens'] <= tokens
            and (other['em_correct'] > cell['em_correct'] or other['solver_tokens'] + other['planner_tokens'] < tokens)
            for name, other in by_arm.items() if name != arm): frontier.append(arm)
    result = {'scope': 'opened development table, not learned policy, expected payoff or original theorem',
        'n': len(labels), 'joint_table_mean_em': [[table[i, j] for j in range(3)] for i in range(3)],
        'empirical_pure_best_response_profiles': pure_best_response_profiles(table),
        'pair_correct_where_both_constituent_singletons_wrong': synergy,
        'repeat_pair_em_disagreements': disagreements, 'arms': by_arm,
        'observed_em_token_pareto_arms': frontier,
        'limitations': ['diagonal repeats one query twice; not two independent source discoveries',
            'a single stochastic answer is not a true expected game payoff',
            'best fixed arm chosen after labels is descriptive, not test performance',
            'observed equilibrium existence does not establish benefit of game learning']}
    freeze(OUT / 'result.json', result)
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__': main()
