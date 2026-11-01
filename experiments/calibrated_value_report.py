"""Sealed paired evaluation and optional independent heldout action panel."""
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
from experiments import calibrated_action_value as v
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.native_optimization_runtime import audit


def load_value_model(path):
    import sys,joblib
    # Models produced by python -m bind the class to __main__.
    # Map that known local class for cross-entry-point artifact loading.
    sys.modules['__main__'].__dict__.setdefault('OutcomeModel',v.OutcomeModel)
    return joblib.load(path)

def load_evaluation(root):
    test=[t for t in read(root/'tasks.json')if t['metadata']['acquisition_partition']=='test']
    cases=[read(root/'evaluation/cases'/f'case_{i}.json')for i in range(len(test))]
    expected={'task_ids':[t['id']for t in test],'hashes':[fingerprint(x)for x in cases]}
    if read(root/'evaluation/generation_complete.json')!=expected:raise ValueError('evaluation seal mismatch')
    return test,cases

def paired(a,b):
    d=np.array(a)-np.array(b);rng=np.random.default_rng(20261029)
    boot=np.mean(d[rng.integers(0,len(d),size=(5000,len(d)))],axis=1)
    win=int(np.sum(d>0));loss=int(np.sum(d<0));n=win+loss
    p=min(1.,2*sum(math.comb(n,i)for i in range(min(win,loss)+1))/2**n)if n else 1.
    return {'difference':float(np.mean(d)),'wins':win,'losses':loss,
            'bootstrap_95':[float(x)for x in np.quantile(boot,[.025,.975])],
            'exact_paired_sign_p':p,'scope':'claim-level diagnostic interval; small sample and multiple comparisons'}

def heldout_panel(root,max_cny=.3):
    # All selectors and evaluation outputs have already frozen and sealed.
    test,_=load_evaluation(root);dest=root/'heldout_panel'
    with locked(dest):
        plan={'version':'heldout-action-panel-1','task_hash':fingerprint(test),'physical_cap':len(test)*4,
              'max_cny':max_cny,'reuse_root':str(root/'evaluation'),
              'source_sha256':{'collector':sha(__file__),'actor':sha(v.__file__)}}
        create(dest/'plan.json',plan);api=Runtime(dest,plan,'qwen_flash')
        for i,t in enumerate(test):
            path=dest/'cases'/f'case_{i}.json'
            if path.exists():continue
            state=c.fresh(t);arms={}
            for name,ids in v.BRANCHES.items():
                after,event=v.actor(api,f'{i}_{name}',t,state,ids)
                arms[name]={'decision':after['decision'],'event':event,'features':v.features(t,state,ids)}
            create(path,{'task_id':t['id'],'branches':arms});print('heldout panel',i+1,len(test),flush=True)
        cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(test))]
        create(dest/'generation_complete.json',{'task_ids':[t['id']for t in test],'hashes':[fingerprint(x)for x in cases]})
        create(dest/'physical.json',audit(dest,'qwen_flash'))

def report(root):
    test,cases=load_evaluation(root);labels=read(root/'test.labels.private.json')
    quality={m:[c.quality(case['methods'][m]['decision'],labels[t['id']])for t,case in zip(test,cases)]for m in v.METHODS}
    summary=read(root/'evaluation/summary.json')
    recomputed={m:int(sum(q))for m,q in quality.items()}
    if any(summary['methods'][m]['correct']!=score for m,score in recomputed.items()):raise ValueError('independent summary mismatch')
    ridge=root/'same_data_ridge'
    methods=list(v.METHODS)
    if (ridge/'generation_complete.json').exists():
        ridge_cases=[read(ridge/'cases'/f'case_{i}.json')for i in range(len(test))]
        if read(ridge/'generation_complete.json')!={'task_ids':[t['id']for t in test],'hashes':[fingerprint(x)for x in ridge_cases]}:raise ValueError('Ridge control seal mismatch')
        quality['same_data_ridge']=[c.quality(x['result']['decision'],labels[t['id']])for t,x in zip(test,ridge_cases)]
        r=read(ridge/'summary.json');summary['methods']['same_data_ridge']={k:r[k]for k in ('correct','initial_correct','improved','harmed','calls','invalid')}
        methods.append('same_data_ridge')
    result={'tasks':len(test),'methods':summary['methods'],
            'paired_new_game':{m:paired(quality['new_game'],quality[m])for m in methods if m!='new_game'},
            'training':read(root/'model/summary.json'),
            'scope':'Natural Climate-FEVER claim evidence, controlled provisional decisions, shared single-reviewer value model. Not original leaderboard or natural initial-model errors.'}
    panel=root/'heldout_panel'
    if (panel/'generation_complete.json').exists():
        import joblib
        bundle=load_value_model(root/'model/model.joblib')
        old=joblib.load(ROOT/'experiments/results/consensus_mechanism_paper_v2/model/model.joblib')
        arms=[read(panel/'cases'/f'case_{i}.json')for i in range(len(test))]
        if read(panel/'generation_complete.json')!={'task_ids':[t['id']for t in test],'hashes':[fingerprint(x)for x in arms]}:raise ValueError('heldout panel seal mismatch')
        errors={'old':[],'new':[]};miss={'old':0,'new':0};valid=0;positive=0;rows=[]
        for t,case in zip(test,arms):
            baseline=c.quality(t['initial'],labels[t['id']]);actions=c.actions(t,c.fresh(t))
            oldvalues=v.old_estimate(t,c.fresh(t),actions,[],old)
            newvalues=v.action_values(t,c.fresh(t),actions,bundle)
            by_sources={tuple(sorted(ids)):name for name,ids in v.BRANCHES.items()}
            observation={}
            for a,oldr,newr in zip(actions,oldvalues,newvalues):
                name=by_sources[tuple(sorted(a['source_ids']))];arm=case['branches'][name]
                if not arm['event']['valid']:continue
                reward=c.quality(arm['decision'],labels[t['id']])-baseline
                valid+=1;positive+=reward>0
                for tag,value in [('old',oldr['gains'][0]),('new',newr['gains'][0])]:
                    errors[tag].append(abs(value-reward));miss[tag]+=reward>0 and value<=0
                observation[name]={'actual':reward,'old':oldr['gains'][0],'new':newr['gains'][0]}
            rows.append({'task_id':t['id'],'action_observations':observation})
        result['heldout_value_prediction']={'valid_actions':valid,'positive_actions':positive,
            'mae':{k:float(np.mean(e))if e else None for k,e in errors.items()},
            'missed_positive':miss,'rows':rows,
            'scope':'Initial-state action prediction; model frozen before heldout panel acquisition, invalid actions excluded'}
    create(root/'report/summary.json',result)
    lines=['# 贡献估计修订：独立留出真实验证','',result['scope'],'',
      '| 方法 | 正确/总数 | 新增纠正 | 损害 | 核查调用 | 无效 |', '|---|---:|---:|---:|---:|---:|']
    names={'old_game':'旧估计＋博弈','new_game':'新估计＋博弈','new_greedy':'新估计＋贪心','new_central':'新估计＋中央','prior_game':'总体均值＋博弈','fixed':'固定顺序','random':'随机可行选择','reflection':'仅反思，不读新材料','same_data_ridge':'同数据同特征Ridge＋博弈'}
    for m in methods:
        row=result['methods'][m]
        lines.append('| '+ ' | '.join([names[m],str(row['correct'])+'/'+str(len(test)),str(row['improved']),str(row['harmed']),str(row['calls']),str(row['invalid'])])+' |')
    lines.extend(['','## 配对结果',''])
    for m,stats in result['paired_new_game'].items():
        lines.append('- 相对'+names[m]+'：差值'+str(stats['difference'])+'，新增/损失'+str(stats['wins'])+'/'+str(stats['losses'])+'，95%区间'+str(stats['bootstrap_95']))
    if 'heldout_value_prediction'in result:
        value=result['heldout_value_prediction'];lines.extend(['','## 留出动作预测','',json.dumps({k:x for k,x in value.items()if k!='rows'},ensure_ascii=False,indent=2)])
    lines.extend(['','不同主张分组留出；训练不读取测试标签文件。共享维基背景，因此不是来源或主题完全隔离。费用见各阶段physical记录，所有方法调用上限相同，实际使用与复用分开披露。'])
    (root/'report/report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:value for k,value in result.items()if k!='heldout_value_prediction'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['panel','report']);p.add_argument('--root',type=Path,default=ROOT/'experiments/results/calibrated_action_value_v2');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.stage=='panel':
        if not a.run:print('No API calls. Add --run for the heldout panel.')
        else:
            from dotenv import load_dotenv
            load_dotenv(ROOT/'.env.local',override=False);heldout_panel(a.root)
    else:report(a.root)
