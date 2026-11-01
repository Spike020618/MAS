"""Prospective state-union comparator on the fresh cohort, exact response reuse."""
import argparse,json
from pathlib import Path
from experiments import calibrated_action_value as v
from experiments import state_union_action_value as u
from experiments.calibrated_value_report import load_value_model
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.native_optimization_runtime import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'experiments/results/calibrated_action_value_v3');p.add_argument('--run',action='store_true');p.add_argument('--model-family',choices=['calibrated','ridge'],default='calibrated');a=p.parse_args()
    tasks=[t for t in read(a.root/'tasks.json')if t['metadata']['acquisition_partition']=='test']
    model=a.root/'models'/(a.model_family+'.joblib')
    bundle=load_value_model(model);dest=a.root/('state_union'if a.model_family=='calibrated'else'ridge_state_union')
    choices={t['id']:{m:u.select(t,c.fresh(t),c.fresh(t),bundle,m,2,2)for m in ('game','greedy','central')}for t in tasks}
    create(dest/'selection.before_generation.json',choices)
    create(dest/'rationale.json',{'reason':'initial-state panel estimator otherwise forgets previously read evidence on replanning',
       'formula':'conditional_gain=F(origin, observed union action)-F(origin, observed)',
       'outcomes_used_for_comparator_design':False,'scope':'shared value model; logical allocation, no physical decentralization claim'})
    if not a.run:print('state-union choices frozen without labels or API');return
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    with locked(dest):
        plan={'version':'state-union-control-1','tasks_hash':fingerprint(tasks),'physical_cap':len(tasks)*6,'max_cny':.3,
              'reuse_root':str(a.root/'evaluation'),'model_sha256':sha(model),
              'source_sha256':{'runner':sha(__file__),'actor':sha(v.__file__),'value':sha(u.__file__)}}
        create(dest/'plan.json',plan);api=Runtime(dest,plan,'qwen_flash')
        for i,t in enumerate(tasks):
            arms={}
            for mode in ('game','greedy','central'):
                path=dest/'arms'/f'{i}_{mode}.json'
                if not path.exists():create(path,u.execute(t,api,bundle,mode,i))
                arms[mode]=read(path)
            create(dest/'cases'/f'case_{i}.json',{'task_id':t['id'],'methods':arms})
        cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(tasks))]
        create(dest/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]})
    labels=read(a.root/'test.labels.private.json');result={'tasks':len(tasks),'methods':{},'physical':audit(dest,'qwen_flash')}
    for mode in ('game','greedy','central'):
        before=[c.quality(t['initial'],labels[t['id']])for t in tasks]
        after=[c.quality(x['methods'][mode]['decision'],labels[t['id']])for t,x in zip(tasks,cases)]
        events=[e for x in cases for e in x['methods'][mode]['events']]
        result['methods'][mode]={'correct':int(sum(after)),'initial_correct':int(sum(before)),
          'improved':sum(y>x for x,y in zip(before,after)),'harmed':sum(y<x for x,y in zip(before,after)),
          'calls':len(events),'invalid':sum(not e['valid']for e in events)}
    create(dest/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
