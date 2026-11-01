"""Counterfactual actions for sampled member, plus same-state member contrasts."""
import argparse,itertools,json
from pathlib import Path
from experiments import member_evidence_data as d
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_recommendation.common import create,fingerprint
from experiments.paired_deliberation.native_optimization_runtime import audit

def collect(root):
    tasks=[t for t in read(root/'tasks.json')if t['metadata']['acquisition_partition']!='test'];dest=root/'train'
    with locked(dest):
        plan=d.freeze(dest,tasks,'training',1.,len(tasks)*7+12,[Path(__file__)])
        api=Runtime(dest,plan,'qwen_flash')
        def gate():
            paths=[dest/'cases'/f'case_{j}.json'for j in range(4)]
            if not all(p.exists()for p in paths):return
            first=[read(p)for p in paths]
            valid=sum(a['event']['valid']for x in first for a in x['arms']);total=sum(len(x['arms'])for x in first)
            initial_valid=sum(e['valid']for x in first for e in x['initial']['events'])
            if valid<.8*total or initial_valid<10:raise RuntimeError('training engineering gate failed; preserve responses, no expansion')
        gate()
        for i,t in enumerate(tasks):
            path=dest/'cases'/f'case_{i}.json'
            if path.exists():continue
            initial_path=dest/'initial'/f'{i}.json'
            if not initial_path.exists():create(initial_path,d.initialize(t,api,i))
            initial=read(initial_path);reviewer=i%4
            pairs=list(itertools.combinations(t['metadata']['acquisition_ids'],2))
            pair=min(pairs,key=lambda ids:fingerprint(['explore',t['id'],ids]))
            actions={'empty':[],'A':[pair[0]],'B':[pair[1]],'AB':list(pair)};arms=[]
            for name,ids in actions.items():
                after,event=d.review(t,initial['state'],initial['reports'],reviewer,ids,api,f'{i}_member{reviewer}_{name}')
                arms.append({'reviewer':reviewer,'branch':name,'source_ids':ids,'decision':after['decision'],
                  'event':event,'features':d.features(t,initial['state'],initial['reports'],reviewer,ids)})
            if i<4:
                for other in range(4):
                    if other==reviewer:continue
                    ids=list(pair);after,event=d.review(t,initial['state'],initial['reports'],other,ids,api,f'{i}_sentinel{other}')
                    arms.append({'reviewer':other,'branch':'AB','source_ids':ids,'decision':after['decision'],
                      'event':event,'features':d.features(t,initial['state'],initial['reports'],other,ids)})
            create(path,{'task_id':t['id'],'partition':t['metadata']['acquisition_partition'],'initial':initial,'arms':arms})
            print('member training',i+1,len(tasks),t['id'],flush=True)
            if i==3:gate()
        cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(tasks))]
        create(dest/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]})
        create(dest/'physical.json',audit(dest,'qwen_flash'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect']);p.add_argument('--root',type=Path,default=d.DEFAULT);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.stage=='prepare':print('prepared',len(d.prepare(a.root)),'native tasks; no API')
    elif a.run:
        from dotenv import load_dotenv
        load_dotenv(d.ROOT/'.env.local',override=False);collect(a.root)
    else:print('No API calls; use --run')
