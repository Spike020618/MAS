"""Independent arithmetic and artifact audit; does not call the online game evaluator."""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import random

from experiments.mechanism.solve_multifollower import solve_multifollower


def sha(data): return hashlib.sha256(data).hexdigest()
def digest(value): return sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def arithmetic(profile, agents, bundles, source_ids, costs, budget):
    sets = [set(bundles[a]) if profile[i] else set() for i, a in enumerate(agents)]
    counts = {s: sum(s in chosen for chosen in sets) for s in source_ids}
    credits = [sum(budget/len(source_ids)/counts[s] for s in chosen) if source_ids else 0.
               for chosen in sets]
    utilities = [credits[i]-costs[a][profile[i]] for i, a in enumerate(agents)]
    return credits, utilities


def verify_development(root, protocol):
    path = root/'experiments/results/mechanism/stackelberg/source_sharing_pilot_v3/analysis.json'
    if sha(path.read_bytes()) != protocol['development_analysis_sha256']:
        raise AssertionError('development result changed')
    data = json.loads(path.read_text())
    calls = {r['key']:r for r in map(json.loads,(path.parent/'calls.jsonl').read_text().splitlines())}
    gold = {'/'.join([r['dataset'],r['question_id']]):r['ground_truth']
            for r in json.loads((path.parent/'private_scoring.json').read_text())}
    # Reconstruct all action reports and all pure equilibria independently.
    for question in data['questions']:
        qid = question['question_id']; agents = ['0','1','2']
        bundles = {a:[s['source_hash'] for s in json.loads(calls[f'{qid}/{a}/R/0']['messages'][1]['content'])['sources']]
                   for a in agents}
        sources = set().union(*map(set,bundles.values()))
        profiles = {}
        for profile in product((0,1),repeat=3):
            qualities, tokens = [], []
            for repeat in range(2):
                selected = [calls[f'{qid}/{a}/{"R" if profile[i] else "D"}/{repeat}'] for i,a in enumerate(agents)]
                votes = Counter(r['report']['C'] if r['schema_valid'] else 'abstain' for r in selected)
                stance = next((s for s in ['yes','no'] if votes[s]>=2),'abstain')
                qualities.append(stance==gold[qid]); tokens.append(sum(r['usage']['total_tokens'] for r in selected))
            profiles[profile] = {'quality':sum(qualities)/2,'tokens':sum(tokens)/2}
        for contract in question['contracts']:
            rows = []
            for profile, outcome in profiles.items():
                credits, utilities = arithmetic(profile,agents,bundles,sources,protocol['costs'],contract['budget'])
                rows.append({'profile':list(profile),'followers':utilities,
                             'leader':outcome['quality']-.01*outcome['tokens']/1000-.01*sum(credits)})
            solved = solve_multifollower({'agents':agents,'actions':{a:[0,1] for a in agents},
                                         'contracts':[{'id':'dev','payoffs':rows}]})
            independent_ne = {tuple(r['profile']) for r in solved['contracts'][0]['pure_nash']}
            assert independent_ne == {tuple(e['profile']) for e in contract['equilibria']}
            for equilibrium in contract['equilibria']:
                outcome = profiles[tuple(equilibrium['profile'])]
                assert outcome['quality'] == equilibrium['quality']
                assert abs(outcome['tokens']-equilibrium['tokens'])<1e-10
    values = []
    for index, budget in enumerate(protocol['budget_grid']):
        worst, best = [], []
        for question in data['questions']:
            contract = question['contracts'][index]
            assert contract['budget'] == budget
            objectives = [e['quality']-.01*e['tokens']/1000-.01*e['paid'] for e in contract['equilibria']]
            worst.append(min(objectives)); best.append(max(objectives))
        values.append({'budget': budget, 'pessimistic': sum(worst)/len(worst),
                       'optimistic': sum(best)/len(best)})
    maximum = max(row['pessimistic'] for row in values)
    optima = [row['budget'] for row in values if abs(row['pessimistic']-maximum)<1e-10]
    assert optima == [protocol['selected_budget']]
    return {'checked': True, 'scope': 'finite development-estimated pure equilibrium tables only',
            'locked_budget': protocol['selected_budget'], 'values': values}


def audit(directory, root):
    protocol = json.loads((directory/'protocol.json').read_text())
    for file, expected in protocol['script_hashes'].items():
        assert sha((root/file).read_bytes()) == expected, 'frozen script changed'
    cases = json.loads((directory/'cases.json').read_text())
    certificate = json.loads((directory/'exposure_certificate.json').read_text())
    history_path = directory/'history_audit.json'
    history = json.loads(history_path.read_text())
    assert sha(history_path.read_bytes()) == certificate['historical_audit_sha256']
    assert digest(cases) == certificate['cases_sha256']
    from experiments.mechanism.validate_locked_source_game import canonical
    seen = set(history['canonical_question_sha256'])
    assert len({canonical(c['question']) for c in cases}) == len(cases)
    assert all(sha(canonical(c['question']).encode()) not in seen for c in cases)
    provenance = json.loads((directory/'dataset_provenance.json').read_text())
    assert sha((directory/'official_train.json').read_bytes()) == provenance['sha256']
    calls = [json.loads(line) for line in (directory/'calls.jsonl').read_text().splitlines()]
    by_key = {r['key']: r for r in calls}
    assert len(by_key) == len(calls)
    for row in calls:
        try: decoded = json.loads(row['response'])
        except ValueError: decoded = None
        assert decoded == row['parsed'], 'stored parse differs from raw API content'
        assert digest(row['messages']) == row['prompt_sha256']
    errors, outcomes = [], []
    for case in cases:
        qid = case['question_id']; short = qid.split('/')[-1]
        bank = json.loads((directory/'banks'/f'{short}.json').read_text())
        source_ids = [s['source_hash'] for s in bank['sources']]
        source_map = {s['source_hash']: s for s in bank['sources']}
        for source in bank['sources']:
            assert sha((directory/'snapshots'/f'{source["source_hash"]}.html').read_bytes()) == source['source_hash']
            assert sha(source['text'].encode()) == source['excerpt_sha256']
        query = by_key[qid+'/query']
        assert query['messages'] == [{'role':'system','content':protocol['query_prompt']},
                                     {'role':'user','content':case['question']}]
        for budget in [0, 1]:
            commitment = json.loads((directory/'commitments'/f'{short}_B{budget}.json').read_text())
            run = json.loads((directory/'runs'/f'{short}_B{budget}.json').read_text())
            assert digest(commitment) == run['commitment_sha256']
            assert commitment['protocol_sha256'] == digest(protocol)
            assert commitment['bank_sha256'] == digest(bank)
            bundles = {'0': source_ids[:1], '1': source_ids[1:2], '2': source_ids}
            assert commitment['bundles'] == bundles
            profile = list(protocol['initial'])
            for turn, event in enumerate(run['decision']['events']):
                assert event['agent'] == protocol['order'][turn % len(protocol['agents'])]
                assert event['before'] == profile
                i = protocol['agents'].index(event['agent'])
                alternative = list(profile); alternative[i] = 1-profile[i]
                assert event['alternative'] == alternative
                _, current = arithmetic(profile, protocol['agents'], bundles, source_ids, protocol['costs'], budget)
                _, other = arithmetic(alternative, protocol['agents'], bundles, source_ids, protocol['costs'], budget)
                gain = other[i]-current[i]
                assert abs(gain-event['gain'])<1e-10
                assert event['updated'] == (gain>1e-10)
                if event['updated']: profile = alternative
            assert profile == commitment['profile']
            assert run['decision']['terminal']['profile'] == profile
            assert commitment['events'] == run['decision']['events']
            assert all(not e['updated'] for e in run['decision']['events'][-3:])
            credits, utilities = arithmetic(profile, protocol['agents'], bundles, source_ids, protocol['costs'], budget)
            table = []
            for candidate in product((0,1), repeat=3):
                _, payoff = arithmetic(candidate, protocol['agents'], bundles, source_ids, protocol['costs'], budget)
                table.append({'profile': list(candidate), 'followers': payoff, 'leader': 0})
            result = solve_multifollower({'agents':protocol['agents'],
                'actions':{a:[0,1] for a in protocol['agents']},
                'contracts':[{'id':'locked','payoffs':table}]})
            assert tuple(profile) in {tuple(e['profile']) for e in result['contracts'][0]['pure_nash']}
            settlement = run['settlement']
            assert all(abs(a-b)<1e-10 for a,b in zip(credits,settlement['credits']))
            assert all(abs(a-b)<1e-10 for a,b in zip(utilities,settlement['followers']))
            assert abs(sum(credits)-settlement['paid'])<1e-10 and sum(credits)<=budget+1e-10
            required = {(agent, source) for i,agent in enumerate(protocol['agents']) if profile[i] for source in bundles[agent]}
            actual = {(r['agent'],r['source']) for r in settlement['receipts']}
            assert actual == required and len(actual) == len(settlement['receipts'])
            for receipt in settlement['receipts']:
                assert receipt['task_id'] == run['request_id']
                assert receipt['bytes'] == (directory/'snapshots'/f'{receipt["source"]}.html').stat().st_size
            votes = []
            for i, agent in enumerate(protocol['agents']):
                row = by_key[f'{qid}/B{budget}/{agent}']
                expected = [source_map[s] for s in bundles[agent]] if profile[i] else []
                assert row['messages'] == [{'role':'system','content':protocol['report_prompts'][i]},
                    {'role':'user','content':json.dumps({'question':case['question'],'sources':expected},ensure_ascii=False)}]
                assert digest(row['messages']) == row['prompt_sha256']
                parsed = row['parsed']
                valid = isinstance(parsed,dict) and parsed.get('C') in ['yes','no','abstain'] and all(isinstance(parsed.get(k),str) for k in ['A','E','I'])
                votes.append(parsed['C'] if valid else 'abstain')
            count = Counter(votes)
            assert run['stance'] == next((s for s in ['yes','no'] if count[s]>=2),'abstain')
        outcomes.append({'question_id':qid,'source_count':len(source_ids)})
    assert len(calls) == len(cases)*7
    final = json.loads((directory/'final_evaluation.json').read_text())
    gold = {'StrategyQA-official/'+r['qid']:'yes' if r['answer'] else 'no'
            for r in json.loads((directory/'official_train.json').read_text())}
    for row in final['rows']:
        for condition in ['no_credit','locked_contract']:
            assert row[condition]['correct'] == (row[condition]['stance']==gold[row['question_id']])
    deltas = [int(r['locked_contract']['correct'])-int(r['no_credit']['correct']) for r in final['rows']]
    preview_path = directory/'external_preview_disclosure.json'
    previewed = set(json.loads(preview_path.read_text())['selected_previewed_ids']) if preview_path.exists() else set()
    clean = [r for r in final['rows'] if r['question_id'] not in previewed]
    clean_accuracy = {c:sum(r[c]['correct'] for r in clean)/len(clean)
                      for c in ['no_credit','locked_contract']}
    improved = sum(not r['no_credit']['correct'] and r['locked_contract']['correct'] for r in clean)
    worsened = sum(r['no_credit']['correct'] and not r['locked_contract']['correct'] for r in clean)
    discordant = improved+worsened
    clean_p = min(1.,2*sum(math.comb(discordant,k) for k in range(min(improved,worsened)+1))/2**discordant) if discordant else 1.
    rng = random.Random(2026101907)
    boot = sorted(sum(rng.choice(deltas) for _ in deltas)/len(deltas) for _ in range(10000))
    # Question-level bootstrap is descriptive at this small n, not a new primary test.
    return {'audit_passed': True, 'n_online_runs':len(cases)*2,
        'n_independently_checked_joint_profiles':len(cases)*2*8,
        'prior_record_overlap':0, 'source_banks_nonempty':sum(r['source_count']>0 for r in outcomes),
        'external_previewed_selected_count':len(previewed),
        'nonpreviewed_cohort':{'n_questions':len(clean),'accuracy':clean_accuracy,
                              'improved':improved,'worsened':worsened,'exact_mcnemar_p':clean_p},
        'source_banks_two_documents':sum(r['source_count']==2 for r in outcomes),
        'bootstrap_accuracy_difference_95': [boot[250],boot[9749]],
        'leader_development_optimum':verify_development(root,protocol),
        'cost_semantics':'explicit model fees; actual API usage separately recorded, no economic-incentive claim',
        'autonomous_llm_choices_tested': False, 'test_contract_reselected':False,
        'limitations':['finite estimated development model optimum, not population optimum',
                       'repository question-field exposure audit, not semantic or pretraining contamination audit',
                       'online follower is an explicit utility policy, not an autonomous LLM'],
        'question_source_counts':outcomes}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.directory,Path(__file__).resolve().parents[2])
    (args.directory/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
