"""Development replay of fixed potential reports; never label this live equilibrium evidence."""
import argparse
from collections import Counter
from itertools import permutations, product
import json
from pathlib import Path
from statistics import mean

from experiments.core.mechanism.source_sharing import SourceSharingGame


def analyze(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    cases = json.loads((directory/'private_scoring.json').read_text())
    calls = [json.loads(line) for line in (directory/'calls.jsonl').read_text().splitlines() if line.strip()]
    if len(calls) != protocol['n_jobs'] or len({r['key'] for r in calls}) != len(calls):
        raise ValueError('complete unique action outcomes required')
    lookup = {(r['question_id'], r['agent'], r['action'], r['repeat']): r for r in calls}
    question_results = []
    for case in cases:
        qid = case['dataset']+'/'+case['question_id']
        bundles = {}
        for i in range(3):
            content = json.loads(lookup[qid, i, 'R', 0]['messages'][1]['content'])
            bundles[str(i)] = [s['source_hash'] for s in content['sources']]
        sources = set().union(*map(set, bundles.values()))
        weights = {s: 1/len(sources) for s in sources} if sources else {}
        profiles = {}
        for profile in product((0, 1), repeat=3):
            outcomes = []
            for repeat in range(2):
                selected = [lookup[qid, i, 'R' if profile[i] else 'D', repeat] for i in range(3)]
                stances = [r['report']['C'] if r['schema_valid'] else 'abstain' for r in selected]
                counts = Counter(stances)
                stance = next((s for s in ['yes', 'no'] if counts[s] >= 2), 'abstain')
                outcomes.append({'correct': stance == case['ground_truth'], 'stance': stance,
                    'abstention': stance == 'abstain',
                    'tokens': sum((r.get('usage') or {}).get('total_tokens', 0) for r in selected),
                    'report_calls': 3, 'report_wall_sum': sum(r['wall_seconds'] for r in selected),
                    'report_wall_parallel_idealized': max(r['wall_seconds'] for r in selected)})
            profiles[''.join(map(str, profile))] = {'profile': list(profile),
                'quality': mean(r['correct'] for r in outcomes),
                'tokens': mean(r['tokens'] for r in outcomes), 'outcomes': outcomes}
        contracts = []
        for budget in [0, .5, 1, 2, 4]:
            agents = ['0', '1', '2']
            game = SourceSharingGame(agents, bundles, weights, {a: [0, .75] for a in agents}, budget)
            equilibria = []
            for equilibrium in game.equilibria():
                key = ''.join(map(str, equilibrium['profile']))
                equilibria.append({**equilibrium, 'quality': profiles[key]['quality'], 'tokens': profiles[key]['tokens']})
            implementations = []
            for order in permutations(agents):
                run = game.run_local_updates(order=order)
                key = ''.join(map(str, run['terminal']['profile']))
                implementations.append({'order': order, 'profile': run['terminal']['profile'],
                                        'regrets': run['regrets'], 'quality': profiles[key]['quality'],
                                        'tokens': profiles[key]['tokens']})
            contracts.append({'budget': budget, 'equilibria': equilibria,
                              'quality_min_ne': min(e['quality'] for e in equilibria),
                              'quality_max_ne': max(e['quality'] for e in equilibria),
                              'implementations': implementations})
        question_results.append({'question_id': qid, 'profiles': profiles, 'contracts': contracts})
    response = []
    for j, budget in enumerate([0, .5, 1, 2, 4]):
        response.append({'budget': budget,
            'mean_quality_min_ne': mean(q['contracts'][j]['quality_min_ne'] for q in question_results),
            'mean_quality_max_ne': mean(q['contracts'][j]['quality_max_ne'] for q in question_results),
            'fixed_order_quality': mean(q['contracts'][j]['implementations'][0]['quality'] for q in question_results),
            'fixed_order_tokens': mean(q['contracts'][j]['implementations'][0]['tokens'] for q in question_results)})
    # Exhaustive sensitivity surface, not a post-hoc choice of one preferred weight.
    # One B across all questions; only equilibrium selection is pessimistic per task.
    leader_sensitivity = []
    for token_weight, payment_weight in product([0, .01, .1], repeat=2):
        candidates = []
        for j, budget in enumerate([0, .5, 1, 2, 4]):
            per_question = []
            for q in question_results:
                values = [e['quality']-token_weight*e['tokens']/1000-payment_weight*e['paid']
                          for e in q['contracts'][j]['equilibria']]
                per_question.append(min(values))
            candidates.append({'budget': budget, 'pessimistic_value': mean(per_question)})
        best = max(c['pessimistic_value'] for c in candidates)
        leader_sensitivity.append({'token_weight': token_weight, 'payment_weight': payment_weight,
            'candidates': candidates,
            'model_optimal_budgets': [c['budget'] for c in candidates if abs(c['pessimistic_value']-best)<1e-10]})
    return {'scope': 'development potential-report replay; not independent test or autonomous response evidence',
            'n_questions': len(cases), 'n_api_responses': len(calls),
            'schema_failures': sum(not r['schema_valid'] for r in calls),
            'valid_report_abstentions': sum(r['schema_valid'] and r['report']['C'] == 'abstain' for r in calls),
            'prompt_variants': sorted(set(r.get('prompt_variant', 'original-v2') for r in calls)),
            'truncated_responses': sum(r['finish_reason'] == 'length' for r in calls),
            'models_returned': sorted(set(str(r.get('returned_model')) for r in calls)),
            'usage_missing': sum(not r.get('usage') for r in calls),
            'actual_collection_tokens': sum((r.get('usage') or {}).get('total_tokens', 0) for r in calls),
            'actual_collection_wall_sum': sum(r['wall_seconds'] for r in calls),
            'cost_units': 'diagnostic D=0,R=.75; not calibrated API costs or enforced wallets',
            'leader_contract_selected': False,
            'leader_sensitivity_scope': 'exploratory finite-model exhaustive solutions for all listed weights; not a frozen deployment choice',
            'leader_sensitivity': leader_sensitivity,
            'latency_scope': 'measured report calls; excludes historical source construction and hypothetical policy execution',
            'limitations': ['six previously exposed questions', 'two samples per action',
                            'no formal statistical power', 'no payment account enforcement',
                            'no autonomous LLM action decisions'],
            'response_surface': response, 'questions': question_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'questions'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
