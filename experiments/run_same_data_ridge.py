"""Additional pre-scoring comparator, not selected from heldout quality."""
import argparse,json
from pathlib import Path
import joblib
from experiments import calibrated_action_value as v
from experiments.calibrated_value_report import load_evaluation,load_value_model
from experiments.action_value_ridge_control import fit_control
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.native_optimization_runtime import audit

def prepare(root):
    # Select partition from inputs only: never access heldout labels/outcomes.
    tasks=[t for t in read(root/'tasks.json')if t['metadata']['acquisition_partition']=='test']
    original=load_value_model(root/'model/model.joblib')
    observations=read(root/'model/observations.json')
    bundle={**original,'outcome':fit_control(observations)}
    dest=root/'same_data_ridge';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'model.joblib'
    if path.exists():bundle=load_value_model(path)
    else:joblib.dump(bundle,path)
    old=joblib.load(ROOT/'experiments/results/consensus_mechanism_paper_v2/model/model.joblib')
    choices={t['id']:v.select(t,c.fresh(t),bundle,old,'new_game',2,2)for t in tasks}
    create(dest/'selection.before_generation.json',choices)
    create(dest/'rationale.json',{'scope':'same training actions, valid-only labels, same features, same validity model, same costs and solver; only signed Ridge versus outcome probability model',
       'reason_added':'design review: historical old model also differs in training domain, so is not a sufficient architecture control',
       'test_labels_read_to_choose_comparator':False,'observations_sha256':sha(root/'model/observations.json')})
    return tasks,bundle,old,dest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'experiments/results/calibrated_action_value_v2');p.add_argument('--run',action='store_true');a=p.parse_args()
    tasks,bundle,old,dest=prepare(a.root)
    if not a.run:print('same-data Ridge prepared without test labels or API');return
    load_evaluation(a.root) # wait for primary evaluation seal before exact receipt reuse
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    with locked(dest):
        plan={'version':'same-data-ridge-1','tasks_hash':fingerprint(tasks),'model_sha256':sha(dest/'model.joblib'),
              'physical_cap':len(tasks)*2,'max_cny':.3,'reuse_root':str(a.root/'evaluation'),
              'source_sha256':{'runner':sha(__file__),'actor':sha(v.__file__),'model':sha(ROOT/'experiments/action_value_ridge_control.py')}}
        create(dest/'plan.json',plan);api=Runtime(dest,plan,'qwen_flash')
        for i,t in enumerate(tasks):
            path=dest/'cases'/f'case_{i}.json'
            if not path.exists():create(path,{'task_id':t['id'],'result':v.execute(t,api,bundle,old,'new_game',i)})
        cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(tasks))]
        create(dest/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]})
    labels=read(a.root/'test.labels.private.json')
    before=[c.quality(t['initial'],labels[t['id']])for t in tasks];after=[c.quality(x['result']['decision'],labels[t['id']])for t,x in zip(tasks,cases)]
    events=[e for x in cases for e in x['result']['events']]
    result={'tasks':len(tasks),'correct':int(sum(after)),'initial_correct':int(sum(before)),
       'improved':sum(y>x for x,y in zip(before,after)),'harmed':sum(y<x for x,y in zip(before,after)),
       'calls':len(events),'invalid':sum(not e['valid']for e in events),'physical':audit(dest,'qwen_flash')}
    create(dest/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
