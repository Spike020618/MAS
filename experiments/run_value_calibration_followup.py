"""Prospective fresh-claim test after v2 diagnostics; no tuning on the old 12 claims."""
import argparse,json,shutil
from pathlib import Path
import joblib
from experiments import calibrated_action_value as v
from experiments.calibrated_value_report import load_value_model,paired
from experiments.action_value_probability_calibration import calibrate
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.native_optimization_runtime import audit
from experiments.namespaced_runtime import NamespacedAPI

METHODS={'base_game':('base','new_game'),'calibrated_game':('calibrated','new_game'),
         'calibrated_greedy':('calibrated','new_greedy'),'calibrated_central':('calibrated','new_central'),
         'ridge_game':('ridge','new_game'),'prior_game':('calibrated','prior_game'),
         'old_game':('base','old_game'),'fixed':('calibrated','fixed'),
         'random':('calibrated','random'),'reflection':('calibrated','reflection')}

def prepare(root):
    original=ROOT/'experiments/results/calibrated_action_value_v2'
    pool=v.prepare(root/'candidate_pool',fit_count=16,cal_count=4,test_count=36)
    # First 20 = exact reused train/calibration. Skip the previous 12 test claims.
    tasks=pool[:20]+pool[32:56]
    all_labels=read(root/'candidate_pool/evaluation.private.json')
    test=[t for t in tasks if t['metadata']['acquisition_partition']=='test']
    create(root/'tasks.json',tasks);create(root/'test.labels.private.json',{t['id']:all_labels[t['id']]for t in test})
    base=load_value_model(original/'model/model.joblib')
    observations=read(original/'model/observations.json')
    calibrated,stats=calibrate(base,observations)
    ridge=load_value_model(original/'same_data_ridge/model.joblib')
    bundles={'base':base,'calibrated':calibrated,'ridge':ridge}
    root.mkdir(parents=True,exist_ok=True)
    for name,bundle in bundles.items():
        p=root/'models'/(name+'.joblib');p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():bundles[name]=load_value_model(p)
        else:joblib.dump(bundle,p)
    create(root/'calibration.json',stats)
    create(root/'design.json',{'previous_test_ids':[t['id']for t in read(original/'tasks.json')if t['metadata']['acquisition_partition']=='test'],
       'fresh_test_ids':[t['id']for t in test],'fit_calibration_ids':base['training_task_ids'],
       'selection':'next 24 claim hashes after previous cohort, no label/outcome selection',
       'test_labels_used_for_calibration':False,
       'reason':'Temperature adjusts confidence but not signed expected-gain bias. Calibrate utility on original four calibration claims, evaluate fresh claims.',
       'source_training_observations_sha256':sha(original/'model/observations.json')})
    old=joblib.load(ROOT/'experiments/results/consensus_mechanism_paper_v2/model/model.joblib')
    first={t['id']:{method:v.select(t,c.fresh(t),bundles[key],old,mode,2,2)
            for method,(key,mode)in METHODS.items()if method!='reflection'}for t in test}
    create(root/'selection.before_generation.json',first)
    return test,bundles,old

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'experiments/results/calibrated_action_value_v3');p.add_argument('--run',action='store_true');a=p.parse_args()
    test,bundles,old=prepare(a.root)
    if not a.run:print(json.dumps({'tasks':len(test),'network':False,'calibration':read(a.root/'calibration.json')}));return
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    dest=a.root/'evaluation_v2'
    with locked(dest):
        plan={'version':'signed-risk-followup-2-namespaced','tasks_hash':fingerprint(test),'physical_cap':len(test)*len(METHODS)*2,
          'max_cny':1.,'reuse_root':str(a.root/'evaluation')if(a.root/'evaluation/plan.json').exists()else None,'model_sha256':{k:sha(a.root/'models'/(k+'.joblib'))for k in bundles},
          'source_sha256':{'runner':sha(__file__),'actor':sha(v.__file__),
                          'calibrator':sha(ROOT/'experiments/action_value_probability_calibration.py'),
                          'namespace':sha(ROOT/'experiments/namespaced_runtime.py')}}
        create(dest/'plan.json',plan);api=Runtime(dest,plan,'qwen_flash')
        for i,t in enumerate(test):
            methods={}
            for method,(key,mode)in METHODS.items():
                path=dest/'arms'/f'{i}_{method}.json'
                if not path.exists():create(path,v.execute(t,NamespacedAPI(api,method),bundles[key],old,mode,i))
                methods[method]=read(path)
            create(dest/'cases'/f'case_{i}.json',{'task_id':t['id'],'methods':methods})
            print('fresh evaluation',i+1,len(test),flush=True)
        cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(test))]
        create(dest/'generation_complete.json',{'task_ids':[t['id']for t in test],'hashes':[fingerprint(x)for x in cases]})
    labels=read(a.root/'test.labels.private.json');quality={};summary={'tasks':len(test),'methods':{},'physical':audit(dest,'qwen_flash')}
    before=[c.quality(t['initial'],labels[t['id']])for t in test]
    for method in METHODS:
        after=[c.quality(x['methods'][method]['decision'],labels[t['id']])for t,x in zip(test,cases)];quality[method]=after
        events=[e for case in cases for e in case['methods'][method]['events']]
        summary['methods'][method]={'correct':int(sum(after)),'initial_correct':int(sum(before)),
          'improved':sum(y>x for x,y in zip(before,after)),'harmed':sum(y<x for x,y in zip(before,after)),
          'calls':len(events),'invalid':sum(not e['valid']for e in events)}
    summary['paired_calibrated']={m:paired(quality['calibrated_game'],quality[m])for m in METHODS if m!='calibrated_game'}
    summary['scope']='24 fresh Climate-FEVER claims with controlled provisional decisions; original fit16/cal4 reused; no tuning on this cohort.'
    create(dest/'summary.json',summary)
    lines=['# 贡献值校准：新主张前瞻验证','',summary['scope'],'',
        '| 方法 | 正确 | 纠正 | 损害 | 调用 | 无效 |','|---|---:|---:|---:|---:|---:|']
    for m,r in summary['methods'].items():lines.append('| '+' | '.join(map(str,[m,r['correct'],r['improved'],r['harmed'],r['calls'],r['invalid']]))+' |')
    lines.extend(['','## 配对结果','',json.dumps(summary['paired_calibrated'],ensure_ascii=False,indent=2)])
    (a.root/'report_v2.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
