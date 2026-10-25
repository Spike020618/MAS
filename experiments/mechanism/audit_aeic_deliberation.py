"""Recompute visibility, stage ordering, action equilibrium and report lineage."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
from itertools import product
import json
from math import comb
from pathlib import Path

from experiments.mechanism.audit_locked_source_game import arithmetic
from experiments.mechanism.solve_multifollower import solve_multifollower
from experiments.mechanism.pilot_source_sharing import digest

ROOT=Path(__file__).resolve().parents[2]


def audit(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    for name,value in protocol['code_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,'frozen code changed'
    cases=json.loads((directory/'cases.json').read_text())
    calls=[json.loads(line) for line in (directory/'calls.jsonl').read_text().splitlines()]
    by_key={r['key']:r for r in calls}
    assert len(calls)==len(by_key)==192
    for row in calls:
        assert digest(row['messages'])==row['prompt_sha256']
        try:decoded=json.loads(row['response'])
        except ValueError:decoded=None
        assert decoded==row['parsed']
    gold={'StrategyQA-official/'+r['qid']:'yes' if r['answer'] else 'no' for r in json.loads(
        (ROOT/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json').read_text())}
    outcomes={};lineages=[];regrets=[];source_dir=ROOT/'experiments/results/mechanism/stackelberg/sprint_source_dev_v1'
    for case in cases:
        qid=case['question_id'];short=qid.split('/')[-1];outcomes[qid]={}
        raw=(source_dir/'banks'/f'{short}.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==protocol['bank_hashes'][short]
        for label,config in protocol['conditions'].items():
            run=json.loads((directory/'runs'/f'{short}_{label}.json').read_text())
            commitment=json.loads((directory/'commitments'/f'{short}_{label}.json').read_text())
            assert commitment==run['commitment'] and commitment['protocol_sha256']==digest(protocol)
            first=[by_key[f'{qid}/{label}/initial/{i}'] for i in range(3)]
            second=[by_key[f'{qid}/{label}/revision/{i}'] for i in range(3)]
            assert max(datetime.fromisoformat(r['utc']) for r in first)<=min(datetime.fromisoformat(r['utc']) for r in second)
            source_ids=[];bundles={};first_votes=[];final_votes=[]
            for i,row in enumerate(first):
                payload=json.loads(row['messages'][1]['content'])
                assert set(payload)=={'question','sources'} and payload['question']==case['question']
                assert row['messages'][0]['content']==protocol['report_prompts'][i]
                first_votes.append(row['parsed']['C'] if isinstance(row['parsed'],dict) and row['parsed'].get('C') in ['yes','no','abstain']
                    and all(isinstance(row['parsed'].get(k),str) for k in ['A','E','I']) else 'abstain')
                if commitment['profile'][i]==0:assert payload['sources']==[]
                for source in payload['sources']:
                    assert set(source)=={'source_hash','title','url','text'}
                    assert hashlib.sha256((source_dir/'snapshots'/f'{source["source_hash"]}.html').read_bytes()).hexdigest()==source['source_hash']
            # Reconstruct the complete source universe independently from repaired snapshots.
            from experiments.mechanism.sprint_source_pipeline import repair_sources
            bank=json.loads(raw)
            repaired=repair_sources(case['question'],[dict(s,snapshot_path=str(source_dir/'snapshots'/f'{s["source_hash"]}.html')) for s in bank['sources']],ROOT)
            source_ids=[s['source_hash'] for s in repaired]
            expected_sources={s['source_hash']:{k:s[k] for k in ['source_hash','title','url','text']} for s in repaired}
            assert digest(list(expected_sources.values()))==commitment['source_bank_sha256']
            bundles={'0':source_ids[:1],'1':source_ids[1:2],'2':source_ids}
            credits,utility=arithmetic(commitment['profile'],['0','1','2'],bundles,source_ids,protocol['costs'],config['budget'])
            rows=[]
            for profile in product((0,1),repeat=3):
                _,u=arithmetic(profile,['0','1','2'],bundles,source_ids,protocol['costs'],config['budget'])
                rows.append({'profile':list(profile),'followers':u,'leader':0})
            solved=solve_multifollower({'agents':['0','1','2'],'actions':{str(i):[0,1] for i in range(3)},
                'contracts':[{'id':label,'payoffs':rows}]})
            assert tuple(commitment['profile']) in {tuple(r['profile']) for r in solved['contracts'][0]['pure_nash']}
            assert all(abs(a-b)<1e-10 for a,b in zip(credits,run['settlement']['credits']))
            assert all(abs(a-b)<1e-10 for a,b in zip(utility,run['settlement']['followers']))
            assert run['settlement']['paid']<=config['budget']+1e-10
            required={(str(i),sid) for i in range(3) if commitment['profile'][i] for sid in bundles[str(i)]}
            actual={(r['agent'],r['source']) for r in run['settlement']['receipts']}
            assert required==actual and len(actual)==len(run['settlement']['receipts'])
            for receipt in run['settlement']['receipts']:
                assert receipt['task_id']==commitment['request_id']
                assert receipt['bytes']==(source_dir/'snapshots'/f'{receipt["source"]}.html').stat().st_size
            for i,row in enumerate(second):
                first_payload=json.loads(first[i]['messages'][1]['content'])
                expected=[expected_sources[s] for s in bundles[str(i)]] if commitment['profile'][i] else []
                assert first_payload['sources']==expected
                assert row['messages'][0]['content']==protocol['revision_prompt']
                payload=json.loads(row['messages'][1]['content'])
                assert set(payload)=={'question','own_initial_report','peers'} and payload['question']==case['question']
                own=first[i]['parsed'] if first_votes[i]!='abstain' or (isinstance(first[i]['parsed'],dict) and all(isinstance(first[i]['parsed'].get(k),str) for k in ['A','E','I']) and first[i]['parsed'].get('C')=='abstain') else {'C':'abstain','A':'','E':'','I':''}
                assert payload['own_initial_report']==own
                expected_peers=[]
                for j in range(3):
                    if j==i:continue
                    report=first[j]['parsed'] if isinstance(first[j]['parsed'],dict) and first[j]['parsed'].get('C') in ['yes','no','abstain'] and all(isinstance(first[j]['parsed'].get(k),str) for k in ['A','E','I']) else {'C':'abstain','A':'','E':'','I':''}
                    peer={'agent_id':str(j),'C':report['C']}
                    if config['visibility']=='AEIC':peer.update({k:report[k] for k in ['A','E','I']});peer['evidence_id']=f'{j}:E'
                    expected_peers.append(peer)
                assert payload['peers']==expected_peers,'peer visibility or round leakage'
                parsed=row['parsed']
                valid=isinstance(parsed,dict) and parsed.get('C') in ['yes','no','abstain'] and all(isinstance(parsed.get(k),str) for k in ['A','E','I'])
                final=parsed['C'] if valid else 'abstain';final_votes.append(final)
                refs=parsed.get('peer_evidence_used',[]) if isinstance(parsed,dict) else []
                known={p['evidence_id']:p['E'] for p in expected_peers if 'evidence_id' in p}
                literal=[]
                for ref in refs if isinstance(refs,list) else []:
                    if (isinstance(ref,dict) and ref.get('evidence_id') in known and ref.get('disposition') in ['accept','reject','uncertain']
                        and isinstance(ref.get('reason'),str) and ref['reason'].strip()
                        and isinstance(ref.get('quote'),str) and len(ref['quote'])>=12 and ref['quote'] in known[ref['evidence_id']]):literal.append(ref)
                assert literal==run['references'][i]['valid']
                lineages.append({'question_id':qid,'condition':label,'agent':str(i),'initial_C':first_votes[i],'revised_C':final,
                    'changed':first_votes[i]!=final,'initial_correct':first_votes[i]==gold[qid],'final_correct':final==gold[qid],
                    'valid_peer_report_references':literal,'semantic_support':'not_verified','causal_attribution':'not_established_by_reference_alone'})
            count=Counter(final_votes);stance=next((s for s in ['yes','no'] if count[s]>=2),'abstain')
            assert final_votes==run['revised_votes'] and first_votes==run['initial_votes'] and stance==run['revised_stance']
            outcomes[qid][label]=stance==gold[qid];regrets.extend(run['regrets'])
    paired=[]
    for left,right in [('B','A'),('C','A'),('D','C'),('D','B')]:
        improved=sum(outcomes[q][left] and not outcomes[q][right] for q in outcomes)
        harmed=sum(not outcomes[q][left] and outcomes[q][right] for q in outcomes);n=improved+harmed
        p=min(1.,2*sum(comb(n,k) for k in range(min(improved,harmed)+1))/2**n) if n else 1.
        paired.append({'comparison':left+'-'+right,'improved':improved,'harmed':harmed,'exploratory_exact_mcnemar_p':p})
    return {'audit_passed':True,'n_questions':8,'n_condition_runs':32,'fresh_calls':len(calls),
        'visibility_and_round_order_pass':True,'resource_equilibrium_pass':True,'max_resource_regret':max(regrets),
        'paired_comparisons':paired,'semantic_support_verified':False,'autonomous_LLM_choices_tested':False,
        'lineage_rows':lineages}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    result=audit(args.directory)
    (args.directory/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='lineage_rows'},ensure_ascii=False,indent=2))
