"""Independent public-record truth, branch visibility and resource audit."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
from itertools import product
import json
from math import comb
from pathlib import Path
import re

from experiments.mechanism.audit_locked_source_game import arithmetic
from experiments.mechanism.solve_multifollower import solve_multifollower
from experiments.mechanism.pilot_source_sharing import digest

ROOT=Path(__file__).resolve().parents[2]


def public_record_truth(case):
    roster=json.loads(case['paid_documents'][0]['text'])['entries']
    records=json.loads(case['paid_documents'][1]['text'])['entries']
    group=re.search(r'group_[a-f0-9]+',case['question']).group()
    targets=[r['entity'] for r in roster if r['group']==group]
    assert len(targets)==4
    current={r['entity']:r['status'] for r in records if r['snapshot']==2}
    unknown=[e for e in targets if current.get(e) is None]
    possible=set()
    for fill in product([False,True],repeat=len(unknown)):
        assignment={e:current.get(e)=='enabled' for e in targets}
        assignment.update(zip(unknown,fill))
        values=list(assignment.values())
        answer=(sum(values)>=2 if 'at least 2' in case['question'] else
                all(values) if 'all members' in case['question'] else any(values))
        possible.add(answer)
    return 'yes' if possible=={True} else 'no' if possible=={False} else 'abstain'


def report_or_empty(row):
    report=row['parsed']
    return report if (isinstance(report,dict) and report.get('C') in ['yes','no','abstain']
                     and all(isinstance(report.get(k),str) for k in ['A','E','I'])) else {'C':'abstain','A':'','E':'','I':''}


def valid_majority_support(rows,truth):
    return sum(isinstance(r['parsed'],dict) and r['parsed'].get('C')==truth
        and all(isinstance(r['parsed'].get(k),str) for k in ['A','E','I']) for r in rows)>=2


def audit(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    script=ROOT/'experiments/mechanism/pilot_abstract_conflicts.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest()==protocol['code_sha256']
    cases=json.loads((directory/'public_cases.json').read_text())
    private=json.loads((directory/'private_scoring.json').read_text())
    assert digest(cases)==protocol['generator_sha256'] and digest(private)==protocol['private_score_sha256']
    scoring={r['question_id']:r for r in private}
    calls=[json.loads(line) for line in (directory/'calls.jsonl').read_text().splitlines()]
    by_key={r['key']:r for r in calls};assert len(calls)==len(by_key)==108
    for row in calls:
        assert digest(row['messages'])==row['prompt_sha256']
        for forbidden in ['ground_truth','target_statuses','preferred_answer']:
            assert forbidden not in row['messages'][1]['content']
        try:parsed=json.loads(row['response'])
        except ValueError:parsed=None
        assert parsed==row['parsed']
    results=[];initial_disagreements=0;regrets=[]
    for case in cases:
        qid=case['question_id'];truth=public_record_truth(case)
        assert truth==scoring[qid]['ground_truth']
        run=json.loads((directory/'runs'/f'{qid}.json').read_text())
        commitment=json.loads((directory/'commitments'/f'{qid}.json').read_text())
        assert commitment['protocol_sha256']==digest(protocol)
        paid={hashlib.sha256(d['text'].encode()).hexdigest():d for d in case['paid_documents']}
        ids=list(paid);bundles={'0':ids[:1],'1':ids[1:2],'2':ids};agents=['0','1','2'];costs={a:[0,.75] for a in agents}
        table=[]
        for profile in product((0,1),repeat=3):
            _,u=arithmetic(profile,agents,bundles,ids,costs,1)
            table.append({'profile':list(profile),'followers':u,'leader':0})
        game=solve_multifollower({'agents':agents,'actions':{a:[0,1] for a in agents},'contracts':[{'id':'fixed','payoffs':table}]})
        assert tuple(commitment['profile']) in {tuple(r['profile']) for r in game['contracts'][0]['pure_nash']}
        credits,u=arithmetic(commitment['profile'],agents,bundles,ids,costs,1)
        assert run['settlement']['credits']==credits and run['settlement']['followers']==u
        required={(a,s) for i,a in enumerate(agents) if commitment['profile'][i] for s in bundles[a]}
        assert {(r['agent'],r['source']) for r in run['settlement']['receipts']}==required
        initial=[by_key[f'{qid}/initial/{i}'] for i in range(3)];own_sources=[]
        for i in range(3):
            sources=[case['local_documents'][i]]+([paid[s] for s in bundles[str(i)]] if commitment['profile'][i] else [])
            own_sources.append(sources)
            assert initial[i]['messages']==[{'role':'system','content':protocol['initial_prompt']},
                {'role':'user','content':json.dumps({'question':case['question'],'rules':case['rules'],'sources':sources},sort_keys=True)}]
        votes=[report_or_empty(r)['C'] for r in initial]
        assert votes==run['initial_votes'];initial_disagreements+=len(set(votes))>1
        branch_correct={};supported_correct={}
        for name in ['ordinary','structured']:
            revised=[by_key[f'{qid}/{name}/{i}'] for i in range(3)]
            assert max(datetime.fromisoformat(r['utc']) for r in initial)<=min(datetime.fromisoformat(r['utc']) for r in revised)
            for i,row in enumerate(revised):
                body={'question':case['question'],'rules':case['rules'],
                    'own':{'report':report_or_empty(initial[i]),'source_packets':own_sources[i]},
                    'peers':[{'agent_id':str(j),'report':report_or_empty(initial[j]),'source_packets':own_sources[j]} for j in range(3) if j!=i]}
                assert row['messages']==[{'role':'system','content':protocol['revision_prompts'][name]},
                    {'role':'user','content':json.dumps(body,sort_keys=True)}]
                other=by_key[f'{qid}/{"structured" if name=="ordinary" else "ordinary"}/{i}']
                assert row['messages'][1]==other['messages'][1], 'branches received different evidence'
            final=[report_or_empty(r)['C'] for r in revised]
            assert final==run['branches'][name]['votes']
            count=Counter(final);stance=next((v for v in ['yes','no'] if count[v]>=2),'abstain')
            branch_correct[name]=stance==truth
            supported_correct[name]=valid_majority_support(revised,truth)
        results.append({'question_id':qid,'split':scoring[qid]['split'],**branch_correct,
                        'valid_report_majority_supported_correct':supported_correct})
        regrets.extend(run['resource_regrets'])
    contrasts=[]
    for split in ['calibration','heldout_seed']:
        subset=[r for r in results if r['split']==split]
        plus=sum(r['structured'] and not r['ordinary'] for r in subset)
        minus=sum(r['ordinary'] and not r['structured'] for r in subset);n=plus+minus
        p=min(1.,2*sum(comb(n,k) for k in range(min(plus,minus)+1))/2**n) if n else 1.
        contrasts.append({'split':split,'structured_improved':plus,'structured_harmed':minus,'exploratory_exact_mcnemar_p':p})
    sensitivity={split:{name:sum(r['valid_report_majority_supported_correct'][name] for r in results if r['split']==split)
        for name in ['ordinary','structured']} for split in ['calibration','heldout_seed']}
    return {'audit_passed':True,'n_questions':len(cases),'fresh_calls':108,
        'public_record_truth_matches_private_scorer':True,'branch_inputs_identical':True,
        'answer_keys_absent_from_prompts':True,'initial_report_disagreement_cases':initial_disagreements,
        'max_resource_regret':max(regrets),'paired_comparisons':contrasts,
        'valid_report_majority_sensitivity_correct_counts':sensitivity,
        'sensitivity_scope':'post-run integrity sensitivity, not replacement of frozen fallback scoring; require two valid reports supporting the true label',
        'structured_truncated_calls':sum('/structured/' in r['key'] and r['finish_reason']=='length' for r in calls),
        'scope':'synthetic smoke only; not real-domain improvement, causal assumption alignment, or Leader optimality'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    result=audit(args.directory)
    (args.directory/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
