"""Strong independent calibration and native single-agent controls, preregistered before scoring."""
import copy,json
from pathlib import Path
import numpy as np
import joblib
from experiments import member_evidence_data as d,member_evidence_value as v
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_recommendation.common import create,fingerprint
from experiments.namespaced_runtime import NamespacedAPI
from experiments.paired_deliberation.native_optimization_runtime import audit

def prepare(root):
    tasks=[t for t in read(root/'tasks.json')if t['metadata']['acquisition_partition']=='test']
    choices={t['id']:[{'member':i,'source_ids':[sid]}for i,sid in enumerate(sorted(t['metadata']['acquisition_ids'],key=lambda s:(-t['source_catalog'][s]['retrieval_score'],s))[:2])]for t in tasks}
    create(root/'controls/design.before_scoring.json',{'tasks':[t['id']for t in tasks],'choices':choices,
       'controls':['training-prior-only','independent same-state queries plus SAME learned fusion','single-retrieval raw native answer'],
       'reason':'separate aggregation prior, fusion, and adaptive selection; no heldout labels used to choose controls'})
    return tasks,choices

def run(root):
    tasks,choices=prepare(root);dest=root/'controls';model=joblib.load(root/'model/model.joblib')
    main=[read(root/'evaluation/cases'/f'case_{i}.json')for i in range(len(tasks))]
    if read(root/'evaluation/generation_complete.json')!={'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in main]}:raise ValueError('main seal mismatch')
    with locked(dest):
        plan=d.freeze(dest,tasks,'strong_controls',.3,len(tasks)*2,[Path(__file__),Path(v.__file__)],root/'model/model.joblib')
        api=NamespacedAPI(Runtime(dest,plan,'qwen_flash'),'independent_calibrated')
        for i,(t,case)in enumerate(zip(tasks,main)):
            path=dest/'cases'/f'case_{i}.json'
            if path.exists():continue
            initial=case['initial'];state=copy.deepcopy(initial['state']);belief=v.posterior(model,t,initial)
            state['decision']=v.ANSWERS[int(np.argmax(belief))];events=[]
            for action in choices[t['id']]:
                member=action['member'];ids=action['source_ids']
                q=v.response_channel(model,t,state,initial['reports'],member,ids)
                _,event=d.review(t,state,initial['reports'],member,ids,api,f'{i}_parallel_{member}')
                events.append({**event,'channel':q.tolist()})
            for event in events:belief=v.update(belief,np.array(event['channel']),event)
            create(path,{'task_id':t['id'],'decision':v.ANSWERS[int(np.argmax(belief))],'events':events})
            print('independent calibrated control',i+1,len(tasks),flush=True)
        cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(tasks))]
        create(dest/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]})
        create(dest/'physical.json',audit(dest,'qwen_flash'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--root',type=Path,default=d.DEFAULT);a=p.parse_args()
    if a.run:
        from dotenv import load_dotenv
        load_dotenv(d.ROOT/'.env.local',override=False);run(a.root)
    else:prepare(a.root);print('strong controls frozen; no API or test scores')
