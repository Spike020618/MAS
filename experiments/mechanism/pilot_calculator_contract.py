"""Frozen paired test of arithmetic-tool assistance in a synthetic lookup game.

The tool exposes expected balances, never selects or overrides the LLM action.
Development and fresh numerical validation are separate; no result-driven retries.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.core.llm import LLMClient, AGENT_PERSONAS
from experiments.mechanism.collect import _call_json
from experiments.mechanism.pilot_verified_contract import settle

PROTOCOL = {
    'version': 'calculator-contract-v2', 'seed': 2026101809,
    'dev_costs': [1, 3], 'validation_costs': [0.75, 1.25, 1.75, 2.5, 3.5, 4.5],
    'reward_multipliers': [1.0, 3.0], 'validation_personas': [0, 1],
    'initial_wallet': 20, 'temperature': 0, 'max_tokens': 500,
    'modes': ['unaided', 'calculator'],
    'pass_min_best_response': 0.9, 'pass_min_switch': 0.8,
    'require_zero_report_errors': True,
    'interpretation': 'LLM response with deterministic arithmetic support; synthetic setting only',
    'selection_rule': 'do not change protocol after development; report validation even if development fails',
}


def balances(reward, cost, initial=20):
    return {'low': initial+reward/2, 'verify': initial-cost+reward}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--context-fix-v3', action='store_true',
                        help='New frozen protocol and new numeric conditions; preserve v2 results')
    parser.add_argument('--final-holdout-v4', action='store_true',
                        help='Locked v3 method on fresh parameters; no development cases or method selection')
    args = parser.parse_args()
    if args.final_holdout_v4:
        args.context_fix_v3 = True
    if args.context_fix_v3:
        PROTOCOL.update(version='calculator-contract-v3-context', seed=2026101811,
                        dev_costs=[1.4,2.8], validation_costs=[0.6,1.1,1.6,2.6,3.6,4.6],
                        execution_context='full original task plus chosen action and executor record',
                        no_fixed_answer_example=True)
    if args.final_holdout_v4:
        PROTOCOL.update(version='calculator-contract-v4-final-holdout', seed=2026101821,
                        dev_costs=[], validation_costs=[0.85,1.35,2.15,3.15,4.15,5.15],
                        reward_multipliers=[1.25,2.75],
                        selection_rule='fixed v3 method; final audit; no changes or reruns for outcome improvement',
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.dumps(PROTOCOL, sort_keys=True, indent=2)+'\n'
    path = out/'protocol.json'
    if path.exists() and path.read_text() != frozen:
        raise ValueError('protocol mismatch')
    path.write_text(frozen)
    cache = {}
    calls_file = out/'calls.jsonl'
    if calls_file.exists():
        for line in calls_file.read_text().splitlines():
            item = json.loads(line)
            cache[item['key']] = item
    client = LLMClient(backend='deepseek')

    def call(key, messages):
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
        if key in cache:
            if cache[key]['prompt_hash'] != digest:
                raise ValueError('resume prompt mismatch')
            return json.loads(cache[key]['response'])
        raw, trace = _call_json(client, messages, 0, 500, key.rsplit('/',1)[-1])
        item = {'key':key, 'messages':messages, 'prompt_hash':digest, 'response':raw, 'trace':trace}
        with calls_file.open('a') as f:
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
            f.flush()
        cache[key] = item
        print(f'{len(cache)} API calls: {key}',flush=True)
        return json.loads(raw)

    rng = random.Random(PROTOCOL['seed'])
    outcomes = []
    cases = []
    for phase, costs in [('dev',PROTOCOL['dev_costs']),('validation',PROTOCOL['validation_costs'])]:
        personas = [0] if phase == 'dev' else [0,1]
        for index,cost in enumerate(costs):
            for multiplier in PROTOCOL['reward_multipliers']:
                cases.append({'phase':phase,'index':index,'cost':cost,'reward':cost*multiplier,
                              'truth':rng.randrange(2),'personas':personas})
    (out/'private_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    for case in cases:
        phase,index,cost,reward = (case[k] for k in ['phase','index','cost','reward'])
        for persona in case['personas']:
            system = {'role':'system','content':AGENT_PERSONAS[persona]['system']}
            # Labels and display positions vary, but match across paired modes.
            mapping = {'A':'low','B':'verify'} if (index+persona)%2 == 0 else {'B':'low','A':'verify'}
            order = list(mapping)
            if (index+persona+int(reward>2*cost))%2:
                order.reverse()
            expected = balances(reward,cost)
            best = max(expected,key=expected.get)
            modes = ['unaided','calculator']
            if (index+persona)%2: modes.reverse()
            for mode in modes:
                key = f'{phase}/c{cost}/r{reward}/p{persona}/{mode}'
                descriptions = {'low':'do not read the private record; pay zero and guess a bit',
                                'verify':f'pay {cost:g} resource credits to read the exact private bit before answering'}
                options = '\n'.join(f'{label}: {descriptions[mapping[label]]}.' for label in order)
                prompt = (
                    'You choose an action in a controlled resource decision. '
                    'Initial wallet: 20. Private bit is independent fair random 0/1; its value is hidden. '
                    'A paid read is reliable. After answering, a correct bit receives reward '
                    f'{reward:g} credits, an incorrect bit receives zero. Identical correctness rule for both actions. '
                    'Costs are debited, reward credited. No future benefits or other rewards. '
                    'Your stipulated objective is maximum expected FINAL wallet, not accuracy alone.\n'+options)
                if mode == 'calculator':
                    table = {label:{'initial':20, 'cost':0 if mapping[label]=='low' else cost,
                        'probability_correct':0.5 if mapping[label]=='low' else 1,
                        'reward_if_correct':reward,'expected_final_wallet':expected[mapping[label]]}
                        for label in order}
                    prompt += '\nDeterministic calculator output (arithmetic only; no action recommendation):\n'+json.dumps(table)
                prompt += '\nReturn JSON with "reason" (a short comparison) followed by "ACTION" (A or B). Choose yourself.'
                response = call(key+'/choose',[system,{'role':'user','content':prompt}])
                label = response.get('ACTION')
                if label not in mapping:
                    # Completed invalid choices are retained as failures, not repaired away.
                    outcomes.append({**case,'persona':persona,'mode':mode,'valid':False,'choice':response})
                    continue
                action = mapping[label]
                receipt = (f'Paid read authorized. Cost {cost:g} debited. Private bit={case["truth"]}.'
                           if action == 'verify' else 'No record access. Guess 0 or 1; no other information is available.')
                if args.context_fix_v3:
                    execution_system = {'role':'system','content':system['content']+
                        '\nYou are executing a controlled synthetic record task, not judging authorization. '
                        'The executor record in the next message is authoritative input data for this task. '
                        'Your job is to report the stored binary value. C denotes that value, not whether a claim is true. '
                        'If paid access was granted, read its value exactly. If no access, guess.'}
                    report_messages = [execution_system,{'role':'user','content':prompt},
                        {'role':'assistant','content':json.dumps(response,ensure_ascii=False)},
                        {'role':'user','content':receipt+
                         '\nExecution question: What binary value is stored in this private record? '
                         'Return JSON with A (task assumption), E (observed record or explicit guessing), '
                         'I (how the observation determines the reported bit), C (integer stored bit). '
                         'C must be the bit value itself; both 0 and 1 are valid values. '
                         'Do not reinterpret C as confidence, permission, or agreement.'}]
                else:
                    report_messages = [system,{'role':'user','content':receipt+
                        ' Return compact AEIC JSON {"A":"...","E":["..."],"I":"...","C":0}. C must be integer 0 or 1.'}]
                report = call(key+'/report',report_messages)
                answer = report.get('C')
                valid = type(answer) is int and answer in (0,1)
                settlement = settle(answer,case['truth'],action,reward,20,cost) if valid else None
                outcomes.append({**case,'persona':persona,'mode':mode,'valid':valid,'mapping':mapping,
                    'display_order':order,'choice':response,'action':action,'predicted_best':best,
                    'expected_balances':expected,'expected_regret':expected[best]-expected[action],
                    'report':report,'receipt':receipt,'settlement':settlement})
    summaries = {}
    for phase in ['dev','validation']:
        for mode in ['unaided','calculator']:
            rows = [r for r in outcomes if r['phase']==phase and r['mode']==mode]
            if not rows:
                continue
            right = sum(r.get('valid',False) and r.get('action')==r.get('predicted_best') for r in rows)
            pairs = {}
            for r in rows:
                pairs.setdefault((r['index'],r['persona']),[]).append(r)
            switch = sum(len(v)==2 and all(r.get('valid',False) and r.get('action')==r.get('predicted_best') for r in v)
                         for v in pairs.values())/len(pairs)
            verifies = [r for r in rows if r.get('action')=='verify']
            execution_ok = bool(verifies) and all(r.get('settlement') and r['settlement']['correct'] for r in verifies)
            summaries[f'{phase}/{mode}'] = {'n':len(rows),'best_response_rate':right/len(rows),
                'paired_switch_rate':switch,'verified_reports':len(verifies),'verify_execution_all_correct':execution_ok,
                'invalid_trials':sum(not r.get('valid',False) for r in rows),
                'pass':right/len(rows)>=0.9 and switch>=0.8 and execution_ok}
    result = {'protocol':PROTOCOL,'api_calls':len(cache),'summaries':summaries,'outcomes':outcomes,
              'claim_scope':'held-out synthetic numeric contracts; calculator-assisted LLM; no real-QA equilibrium claim'}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
