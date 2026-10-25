"""Offline diagnostic of recorded outcomes, not a causal attribution or source judge."""
from collections import Counter
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def vote(stances):
    count = Counter(stances)
    return next((s for s in ['yes','no'] if count[s]>=2),'abstain')


def diagnose(root=ROOT):
    base = root/'experiments/results/mechanism/stackelberg'
    directory = base/'source_sharing_pilot_v3'
    cases = json.loads((directory/'private_scoring.json').read_text())
    calls = {r['key']:r for r in map(json.loads,(directory/'calls.jsonl').read_text().splitlines())}
    rows = []
    for case in cases:
        qid = case['dataset']+'/'+case['question_id']; gold = case['ground_truth']
        for repeat in range(2):
            agents = []
            for i in range(3):
                pair = [calls[f'{qid}/{i}/{a}/{repeat}'] for a in ['D','R']]
                d,r = [p['report']['C'] if p['schema_valid'] else 'abstain' for p in pair]
                content = json.loads(pair[1]['messages'][1]['content'])
                agents.append({'agent':i,'D':d,'R':r,'D_correct':d==gold,'R_correct':r==gold,
                    'change':'corrected' if d!=gold and r==gold else 'harmed' if d==gold and r!=gold else 'unchanged_correctness',
                    'sources':[{'hash':s['source_hash'],'title':s['title'],'url':s['url'],'text':s['text']}
                               for s in content['sources']],
                    'retrieval_report':pair[1]['report'],'source_sufficiency':'unjudged_not_inferred_from_self_report'})
            profiles = []
            for profile in product((0,1),repeat=3):
                stances = [agents[i]['R' if action else 'D'] for i,action in enumerate(profile)]
                profiles.append({'profile':list(profile),'stance':vote(stances),'correct':vote(stances)==gold,
                    'some_individual_correct':any(s==gold for s in stances)})
            rows.append({'question_id':qid,'question':case['question'],'gold_offline_only':gold,'repeat':repeat,
                'agents':agents,'profiles':profiles,
                'potential_profile_correctness_varies':len({p['correct'] for p in profiles})>1})
    locked = json.loads((base/'locked_source_game_test_v1/final_evaluation.json').read_text())
    locked_calls = {r['key']:r for r in map(json.loads,(base/'locked_source_game_test_v1/calls.jsonl').read_text().splitlines())}
    locked_rows=[]
    for row in locked['rows']:
        outcomes={}
        for c in ['no_credit','locked_contract']:
            run=row[c]
            outcomes[c]={'profile':run['decision']['terminal']['profile'],'stance':run['stance'],
                         'correct':run['correct'],
                         'votes':[locked_calls[k]['parsed']['C'] for k in run['report_keys']]}
        changed_same_action=[i for i in range(3) if
            outcomes['no_credit']['profile'][i]==outcomes['locked_contract']['profile'][i]
            and outcomes['no_credit']['votes'][i]!=outcomes['locked_contract']['votes'][i]]
        locked_rows.append({'question_id':row['question_id'],'outcomes':outcomes,
                            'same_action_vote_changes':changed_same_action,
                            'missing_counterfactuals':'unexecuted actions, repeat variability, source semantic validity'})
    all_agents=[a for r in rows for a in r['agents']]
    profile_summary=[]
    for i in range(8):
        ps=[r['profiles'][i] for r in rows]
        profile_summary.append({'profile':ps[0]['profile'],'correct':sum(p['correct'] for p in ps),'n_replays':len(ps),
            'wrong_majorities_with_some_correct_report':sum(not p['correct'] and p['some_individual_correct'] for p in ps)})
    return {'scope':'old development and completed test diagnostics only; all become exposed for future tests',
        'no_new_api_calls':True,'n_dev_questions':len(cases),'n_dev_agent_pairs':len(all_agents),
        'dev_D_correct':sum(a['D_correct'] for a in all_agents),'dev_R_correct':sum(a['R_correct'] for a in all_agents),
        'dev_corrected':sum(a['change']=='corrected' for a in all_agents),
        'dev_harmed':sum(a['change']=='harmed' for a in all_agents),
        'dev_profile_summary':profile_summary,
        'locked_same_action_vote_changes':sum(len(r['same_action_vote_changes']) for r in locked_rows),
        'limitations':['paired D/R samples are stochastic potential outcomes, not same-sample causal proof',
                      'correct-report oracle gaps do not show a feasible aggregator can identify the correct report',
                      'source sufficiency requires content review; presence or model self-report is not truth',
                      'no independent reports can be composed after introducing peer-dependent generation'],
        'development_rows':rows,'locked_rows':locked_rows}


if __name__=='__main__':
    result=diagnose()
    path=ROOT/'experiments/results/mechanism/stackelberg/failure_chain_diagnostic_v1.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_rows')},ensure_ascii=False,indent=2))
