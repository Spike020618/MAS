"""Additional training-side conditional observations after a real first review."""
import copy,json
from pathlib import Path
from experiments import member_evidence_data as d
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_recommendation.common import create,fingerprint
from experiments.paired_deliberation.native_optimization_runtime import audit

def collect(root):
    tasks=[t for t in read(root/'tasks.json')if t['metadata']['acquisition_partition']!='test']
    original=[read(root/'train/cases'/f'case_{i}.json')for i in range(len(tasks))]
    if read(root/'train/generation_complete.json')!={'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in original]}:raise ValueError('training seal mismatch')
    dest=root/'sequence'
    with locked(dest):
        plan=d.freeze(dest,tasks,'conditional_training',.3,len(tasks),[Path(__file__)])
        api=Runtime(dest,plan,'qwen_flash')
        for i,(t,case)in enumerate(zip(tasks,original)):
            path=dest/'cases'/f'case_{i}.json'
            if path.exists():continue
            first=next(a for a in case['arms']if a['branch']=='A');second=next(a for a in case['arms']if a['branch']=='B')
            state=copy.deepcopy(case['initial']['state']);reports=copy.deepcopy(case['initial']['reports'])
            c.expose(t,state,first['source_ids'])
            if first['event']['valid']:
                report=first['event']['report'];refs=c.citations(t,state,report)
                state,_=c.commit(t,state,report,'review',100,refs);state['claims']['target']['active']=True
                if first['reviewer']<3:reports[first['reviewer']]={**first['event']['raw_report'],'contract_valid':True}
            member=(first['reviewer']+1)%4;ids=second['source_ids']
            features=d.features(t,state,reports,member,ids)
            after,event=d.review(t,state,reports,member,ids,api,f'{i}_conditional_{member}')
            create(path,{'task_id':t['id'],'arm':{'reviewer':member,'branch':'B_after_A','source_ids':ids,
               'decision':after['decision'],'event':event,'features':features},'baseline':state['decision']})
            print('conditional training',i+1,len(tasks),flush=True)
        cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(tasks))]
        create(dest/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]})
        create(dest/'physical.json',audit(dest,'qwen_flash'))
if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv(d.ROOT/'.env.local',override=False);collect(d.DEFAULT)
