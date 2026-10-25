"""Offline theory/code alignment of the successful historical process contract.

No new contract, no model calls, no changes to frozen data. Bounds refer only
to the declared immediate credit-minus-cost objective, not unknown LLM goals.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.core.mechanism.controller import MechanismController
from experiments.core.mechanism.schema import AEICReport, AgentAction, MechanismOffer

ROOT = Path(__file__).resolve().parents[2]


def static_bounds(offer):
    """Valid for fixed_process_quality only: both reward signals lie in [0,1]."""
    if offer['credit_rule'] != 'fixed_process_quality':
        raise ValueError('bounds only cover fixed_process_quality')
    if set(offer['visible_reward']) - {'complete_aeic', 'consistency'}:
        raise ValueError('unknown signal')
    low_credit = sum(min(0., value) for value in offer['visible_reward'].values())
    high_credit = sum(max(0., value) for value in offer['visible_reward'].values())
    floor = low_credit - offer['visible_cost']['low_effort']
    return {a: {'min_utility': low_credit - cost, 'max_utility': high_credit - cost,
                'strictly_dominated_by_low_for_any_report': high_credit - cost < floor - 1e-10}
            for a, cost in offer['visible_cost'].items()}


def audit():
    path = ROOT / 'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    counts = Counter()
    bounds = {}
    mismatches = []
    rules = Counter()
    reput_errors = 0
    for row in rows:
        for turn in row['rounds']:
            offer = turn['offered']
            rules[offer['credit_rule']] += 1
            # Replay each local settlement from its logged pre-state. Do not
            # assume that file order is the original collection order.
            controller = MechanismController(5)
            agent = turn['agent_id']
            controller.reputation[agent] = turn['reputation_before']
            replay = controller.settle(agent, MechanismOffer(**offer),
                AgentAction(turn['chosen_action']), AEICReport(**turn['report']),
                row['ground_truth'] if turn['round_id'] == 3 else None)
            for key, actual in [('credit', turn['reward']), ('cost', turn['realized_cost']),
                                ('reputation_after', turn['reputation_after'])]:
                if abs(replay[key] - actual) > 2e-6:
                    mismatches.append({'question': [row['dataset'], row['question_id']],
                        'condition': row['condition'], 'agent': agent,
                        'round': turn['round_id'], 'field': key})
            if abs(.9 * turn['reputation_before'] + turn['reward'] -
                   turn['realized_cost'] - turn['reputation_after']) > 2e-6:
                reput_errors += 1
            if offer['credit_rule'] != 'fixed_process_quality':
                continue
            signature = json.dumps({'rewards': offer['visible_reward'],
                                    'costs': offer['visible_cost']}, sort_keys=True)
            table = static_bounds(offer)
            bounds[signature] = table
            action = turn['chosen_action']
            counts['agent_rounds'] += 1
            counts['dominated_chosen'] += table[action]['strictly_dominated_by_low_for_any_report']
            counts[action] += 1
    matrix_path = ROOT / 'experiments/mechanism/matrices/formal_v10.json'
    matrix = json.loads(matrix_path.read_text())
    checks = {'all_settlements_replay': not mismatches,
              'all_reputation_updates_match': reput_errors == 0,
              'static_agent_rounds_6750': counts['agent_rounds'] == 6750,
              'main_peer_behavior_independent': matrix['defaults']['behavior'] == 'independent'}
    return {'scope': 'historical static process contract, frozen immediate-payoff audit; not a new equilibrium or quality experiment',
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'checks': checks, 'passed_execution_replay': all(checks.values()),
        'record_count': len(rows), 'credit_rules': dict(rules),
        'static_action_counts': dict(counts), 'static_payoff_bounds': bounds,
        'settlement_mismatches': mismatches, 'reputation_mismatches': reput_errors,
        'conclusions': {
            'static_immediate_credit_has_peer_action_externality': False,
            'separable_stage_payoffs_have_trivial_exact_potential': True,
            'observed_static_choices_are_best_response_under_declared_immediate_objective': False,
            'dynamic_reputation_equilibrium_established': False,
            'main_independent_condition_is_peer_deliberation_evidence': False,
            'source_sharing_theorem_transfers_to_historical_static_contract': False},
        'limitations': ['own payoff function separable only conditional on frozen history and independent execution; not a general dynamic game result',
            'reported bounds are declared payoff units, not API prices or latent LLM preferences',
            'reputation enters later prompts; no specified discount/horizon/terminal utility justifies dynamic rationality',
            'recorded accuracy improvements are unchanged; equilibrium interpretation is what fails',
            'current-code replay agrees on frozen observations, not proof all historical source code was identical']}


if __name__ == '__main__':
    result = audit()
    out = ROOT / 'experiments/results/mechanism/synthesis/contract_game_alignment_2026-10-21.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed_execution_replay']:
        raise SystemExit(1)
