"""Real scarce-evidence consensus: local reports -> response games -> posterior updates."""
import argparse,copy,itertools,json,math,random,time
from pathlib import Path
import numpy as np
import joblib
from experiments import member_evidence_data as d
from experiments import member_evidence_value as value
from experiments.member_evidence_game import EvidenceGame
from experiments.namespaced_runtime import NamespacedAPI
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_recommendation.common import create,fingerprint,sha
from experiments.paired_deliberation.native_optimization_runtime import audit

METHODS=('single_controller','single_retrieval','independent','fixed','random','greedy','central','game','exchange','shared_game')

def context_key(task,state,reports,member,ids):
    seeds=task['metadata']['seed_ids'];visible=set(state['visible'])|set(ids)
    if member<3:visible={s for s in visible if s not in seeds or s==seeds[member]}
    peers=[r for i,r in enumerate(reports)if i!=member]
    return member,tuple(sorted(visible)),fingerprint(peers)

def planner(task,state,reports,belief,model,calls,sources,history,shared=False,players=3):
    candidates=[s for s in task['metadata']['acquisition_ids']if s not in state['visible']]
    cache={}
    def full_channel(member,ids):
        key=member,tuple(ids)
        if key not in cache:cache[key]=value.response_channel(model,task,state,reports,member,ids)
        return cache[key]
    def channel(member,ids):
        return np.mean([full_channel(i,ids)for i in range(3)],axis=0)if shared else full_channel(member,ids)
    def cost(member,ids):
        seeds=task['metadata']['seed_ids'];visible=set(state['visible'])|set(ids)
        if member<3:visible={s for s in visible if s not in seeds or s==seeds[member]}
        chars=sum(len(task['source_catalog'][s]['text'])for s in visible)+sum(len(r['claim'])for r in reports)
        return .1*(.0005+chars*.15/3e6)
    options=[()]+[tuple(x)for n in range(1,min(2,sources)+1)for x in itertools.combinations(sorted(candidates),n)]
    blocked={(i,ids)for i in range(players)for ids in options if context_key(task,state,reports,i,ids)in history}
    return EvidenceGame(belief,candidates,channel,cost,calls,sources,players,blocked),full_channel

def initial_choice(task,initial,model,method):
    state=copy.deepcopy(initial['state']);reports=initial['reports'];belief=value.posterior(model,task,initial)
    state['decision']=value.ANSWERS[int(np.argmax(belief))]
    if method in ('single_retrieval','independent','fixed','random'):return {'rule':method}
    g,_=planner(task,state,reports,belief,model,2,2,set(),method=='shared_game',4 if method=='single_controller'else 3)
    result=g.solve('central'if method in ('central','single_controller')else'game'if method=='shared_game'else method)
    result.pop('solver_seconds',None);return result

def execute(task,initial,api,model,method,index):
    state=copy.deepcopy(initial['state']);reports=copy.deepcopy(initial['reports']);belief=value.posterior(model,task,initial)
    state['decision']=value.ANSWERS[int(np.argmax(belief))]
    start=state['decision'];events=[];plans=[];history=set();left_calls=2;left_sources=2
    candidates=task['metadata']['acquisition_ids']
    ranked=sorted(candidates,key=lambda s:(-task['source_catalog'][s]['retrieval_score'],s))
    if method=='independent':
        finals=[r['decision']for r in reports]
        for member,sid in enumerate(ranked[:2]):
            local=c.fresh(task);local['decision']=reports[member]['decision'];c.expose(task,local,[task['metadata']['seed_ids'][member]])
            after,event=d.review(task,local,[],member,[sid],api,f'{index}_independent_{member}',True)
            events.append(event);finals[member]=after['decision']
        return {'decision':c.aggregate(task,finals),'initial_decision':initial['state']['decision'],
          'events':events,'plans':[],'calls':len(events),'new_sources':2,'belief':None}
    while left_calls>0:
        selected=[];game=None;full_channel=None
        if method=='single_retrieval':
            if events:break
            selected=[{'member':3,'source_ids':ranked[:2]}]
        elif method=='fixed':
            unread=[s for s in ranked if s not in state['visible']]
            if not unread or left_sources<=0:break
            selected=[{'member':len(events)%3,'source_ids':[unread[0]]}]
        elif method=='random':
            game,full_channel=planner(task,state,reports,belief,model,left_calls,left_sources,history)
            feasible=[]
            for i in range(3):
                for q in range(1,len(game.options)):
                    s=[0]*3;s[i]=q
                    if game.feasible(tuple(s)):feasible.append({'member':i,'source_ids':list(game.options[q])})
            if not feasible:break
            selected=[random.Random(int(fingerprint([task['id'],len(events),'random'])[:16],16)).choice(feasible)]
        else:
            game,full_channel=planner(task,state,reports,belief,model,left_calls,left_sources,history,method=='shared_game',4 if method=='single_controller'else 3)
            result=game.solve('central'if method in ('central','single_controller')else'game'if method=='shared_game'else method)
            plans.append(result);selected=result['selected']
        if not selected:break
        # A batch uses one frozen public state. Reports are shared only after all
        # planned calls complete, matching the modeled response channels.
        before=copy.deepcopy(state);previous_reports=copy.deepcopy(reports);new_sources=set();batch=[]
        for k,action in enumerate(selected):
            member=action['member'];ids=action['source_ids']
            channel=value.response_channel(model,task,before,previous_reports,member,ids)
            after,event=d.review(task,before,previous_reports,member,ids,api,f'{index}_round{len(events)}_member{member}')
            event.update(channel=channel.tolist(),public_belief_before=belief.tolist(),batch_state_hash=fingerprint(before))
            history.add(context_key(task,before,previous_reports,member,ids));new_sources.update(ids)
            batch.append((member,after,event,channel))
        for member,after,event,channel in batch:
            belief=value.update(belief,channel,event)
            if event['valid']:
                state['assessments'].append({'member':member,'report':event['report']})
                if member<3:reports[member]={**event['raw_report'],'contract_valid':True}
            event['public_belief_after']=belief.tolist();events.append(event)
        c.expose(task,state,sorted(new_sources));state['decision']=value.ANSWERS[int(np.argmax(belief))]
        state['version']+=1;left_calls-=len(batch);left_sources-=len(new_sources)
    return {'decision':state['decision'],'initial_decision':start,'events':events,'plans':plans,
      'calls':len(events),'new_sources':2-left_sources,'belief':belief.tolist()}

def evaluate(root):
    tasks=[t for t in read(root/'tasks.json')if t['metadata']['acquisition_partition']=='test']
    model_path=root/'model/model.joblib';model=joblib.load(model_path)
    if set(t['id']for t in tasks)&set(model['training_claim_ids']):raise ValueError('claim split overlap')
    dest=root/'evaluation'
    with locked(dest):
        plan=d.freeze(dest,tasks,'evaluation',1.,len(tasks)*(3+len(METHODS)*2),[Path(__file__),Path(value.__file__),Path(d.ROOT/'experiments/member_evidence_game.py'),Path(d.ROOT/'experiments/namespaced_runtime.py')],model_path)
        api=Runtime(dest,plan,'qwen_flash');initials=[]
        for i,t in enumerate(tasks):
            path=dest/'initial'/f'{i}.json'
            if not path.exists():create(path,d.initialize(t,NamespacedAPI(api,'shared_initial'),i))
            initials.append(read(path));print('test initialization',i+1,len(tasks),flush=True)
        choices={t['id']:{m:initial_choice(t,init,model,m)for m in METHODS}for t,init in zip(tasks,initials)}
        create(dest/'selection.before_generation.json',choices)
        for i,(t,initial)in enumerate(zip(tasks,initials)):
            arms={}
            for method in METHODS:
                path=dest/'arms'/f'{i}_{method}.json'
                if not path.exists():create(path,execute(t,initial,NamespacedAPI(api,method),model,method,i))
                arms[method]=read(path)
            create(dest/'cases'/f'case_{i}.json',{'task_id':t['id'],'initial':initial,'methods':arms})
            print('member evaluation',i+1,len(tasks),flush=True)
        cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(tasks))]
        create(dest/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]})
    return report(root)

def report(root):
    from experiments.calibrated_value_report import paired
    tasks=[t for t in read(root/'tasks.json')if t['metadata']['acquisition_partition']=='test'];dest=root/'evaluation'
    cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(tasks))]
    if read(dest/'generation_complete.json')!={'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]}:raise ValueError('evaluation seal mismatch')
    labels=read(root/'test.labels.private.json');results={'tasks':len(tasks),'methods':{},'physical':audit(dest,'qwen_flash')};quality={}
    raw=[c.quality(x['initial']['state']['decision'],labels[t['id']])for t,x in zip(tasks,cases)]
    bayes=[c.quality(value.ANSWERS[int(np.argmax(value.posterior(joblib.load(root/'model/model.joblib'),t,x['initial'])))],labels[t['id']])for t,x in zip(tasks,cases)]
    results['initial_majority_correct']=int(sum(raw));results['initial_bayesian_correct']=int(sum(bayes))
    for method in METHODS:
        after=[c.quality(x['methods'][method]['decision'],labels[t['id']])for t,x in zip(tasks,cases)];quality[method]=after
        events=[e for case in cases for e in case['methods'][method]['events']]
        plans=[p for case in cases for p in case['methods'][method]['plans']]
        results['methods'][method]={'correct':int(sum(after)),'improved_vs_bayesian':sum(y>x for x,y in zip(bayes,after)),
          'harmed_vs_bayesian':sum(y<x for x,y in zip(bayes,after)),'review_calls':len(events),
          'invalid_reviews':sum(not e['valid']for e in events),'new_sources':sum(x['methods'][method]['new_sources']for x in cases),
          'pair_exchanges':sum(tr['kind']=='pair_exchange'for p in plans for tr in p['trace']),
          'max_unilateral_deviation':max([0.]+[p['max_unilateral_deviation']for p in plans])}
    results['paired_exchange']={m:paired(quality['exchange'],q)for m,q in quality.items()if m!='exchange'}
    results['scope']='Three real initial contexts; at least five candidate passages, additional source budget two; learned response channels and fixed-state potential games. Shared corpus and small diagnostic claim sample.'
    create(dest/'summary.json',results)
    names={'single_controller':'单Agent同权限规划（可模拟各成员上下文）','single_retrieval':'单Agent读检索前两份','independent':'独立核查后投票','fixed':'固定顺序＋同一共识更新',
      'random':'随机动作＋同一共识更新','greedy':'贪心分配','central':'中央穷举','game':'单边最佳响应博弈',
      'exchange':'允许互利交换的博弈','shared_game':'忽略成员差异的博弈'}
    lines=['# 材料稀缺与成员条件贡献：实测报告','',results['scope'],'',
       '初始多数汇总：'+str(results['initial_majority_correct'])+'/'+str(len(tasks))+'；预算为零的贝叶斯汇总：'+str(results['initial_bayesian_correct'])+'/'+str(len(tasks))+'。','',
       '| 方法 | 正确 | 新增/损害（对零预算贝叶斯汇总） | 核查调用 | 无效 | 新资料读取 |','|---|---:|---:|---:|---:|---:|']
    for m,r in results['methods'].items():lines.append('| '+' | '.join(map(str,[names[m],str(r['correct'])+'/'+str(len(tasks)),str(r['improved_vs_bayesian'])+'/'+str(r['harmed_vs_bayesian']),r['review_calls'],r['invalid_reviews'],r['new_sources']]))+' |')
    lines.extend(['','所有方法共享每题3次初始调用；表中核查调用为额外部分，每题最多2次、最多2份新资料。单Agent同权限规划可选择所有成员上下文与中性完整上下文，选择空间更强，未通过限制它来制造多Agent优势。',
       '研究性后验采用条件独立响应近似，不是语义真实性认证；强负例、来源相关性与更大样本仍须检验。','',json.dumps(results['paired_exchange'],ensure_ascii=False,indent=2)])
    (root/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2));return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','evaluate','report']);p.add_argument('--root',type=Path,default=d.DEFAULT);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.stage=='fit':
        if not(a.root/'model/model.joblib').exists():value.train(a.root)
    elif a.stage=='report':report(a.root)
    elif a.run:
        from dotenv import load_dotenv
        load_dotenv(d.ROOT/'.env.local',override=False);evaluate(a.root)
    else:print('No API calls; use --run')
