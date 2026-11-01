"""Prospective frozen allocation, real API potential-outcome review collection.

Each group has four distinct queries. Three reviewer players may review at most
one query each. Policies are frozen before the 192 potential outcomes are called.
This is not an online adaptive interaction or a semantic-truth evaluation.
"""
import argparse,json,copy,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_reviewer_calibration import request,normalize
from experiments.paired_deliberation.native_candidate_challenge_focused import apply_response
from experiments.paired_deliberation.native_optimization_runtime import Runtime,audit
from experiments.paired_deliberation.review_model_profiles import PROFILES
from experiments.paired_deliberation.expected_review_value import asymmetric_review_value
from experiments.paired_deliberation.native_quality_review_game import QualityReviewGame

PACKETS=ROOT/'experiments/results/native_review_diagnostic_packets_v1'
BLIND=ROOT/'experiments/results/native_review_diagnostic_v1'
OUT=ROOT/'experiments/results/native_review_diagnostic_live_v1'
MODELS=('glm_flash','qwen_flash','qwen_plus')


def load(path):return json.loads(Path(path).read_text())


def policy_commitment(rows,rates,costs):
    groups=[]
    for offset in range(0,len(rows),4):
        group=rows[offset:offset+4]
        gains=[[asymmetric_review_value(float(r['target']['predicted_b_count'])-float(r['target']['predicted_a_count']),float(r['target']['delta_member_sd']),rates[m]['recovery_if_b_observed_more'],rates[m]['safe_retention_if_a_observed_more'])['incremental_review_gain'] for r in group]for m in MODELS]
        matrix_cost=[[costs[m]]*len(group)for m in MODELS]
        game=QualityReviewGame(gains,matrix_cost,cost_weight=1.)
        solved=game.solve()
        def assigned(state):
            return [{'provider':MODELS[i],'task_id':group[q]['task_id'],'query':q,'gain':gains[i][q],'forecast_cost_cny':costs[MODELS[i]]}for i,q in enumerate(state)if q is not None]
        # Fixed single-reviewer controls spend the same maximum three requests,
        # on the greatest frozen predicted review utility, with no winner labels.
        fixed={}
        for m in ('qwen_flash','qwen_plus'):
            i=MODELS.index(m);indices=sorted(range(len(group)),key=lambda q:(-gains[i][q],group[q]['task_id']))[:3]
            fixed['fixed_'+m]=[{'provider':m,'task_id':group[q]['task_id'],'query':q,'gain':gains[i][q],'forecast_cost_cny':costs[m]}for q in indices]
        groups.append({'group':len(groups),'task_ids':[r['task_id']for r in group],'gains':gains,'forecast_costs_cny':matrix_cost,
            'game_solution':solved,'policies':{'base':[],'game':assigned(solved['actions']),'central':assigned(solved['central_actions']),**fixed}})
    return {'grouping':'blind serving order, consecutive groups of four distinct original conversations','reviewers':MODELS,'max_calls_per_group':3,
       'fixed_single_model_scope':'same full per-query information; at most three independent targeted reviews, not sequential self-reflection',
       'gain_baseline':'best direct mean-count decision; Normal and member-SD assumptions unverified','cost_weight_count_per_cny':1.,'groups':groups,'labels_available':False}


def prepare(root=OUT,packets=PACKETS):
    root=Path(root);packets=Path(packets)
    rows=load(packets/'pair64.inputs.json');states=load(packets/'pair_states.json')
    if len(rows)!=64 or len({r['case_id'].rsplit(':',1)[0]for r in rows})!=64:raise ValueError('64 distinct conversations required')
    for r in rows:
        if fingerprint(states[r['task_id']])!=r['target']['base_ranking_hash']:raise ValueError('state hash mismatch')
        if r.get('query_labels_available') is not False:raise ValueError('query labels must be unavailable')
    frozen=load(BLIND/'frozen_fit_rates.json');fit=ROOT/'experiments/results/native_reviewer_calibration_v1/summary.json'
    if sha(fit)!=frozen['summary_sha256']:raise ValueError('fit summary changed')
    prior=load(fit);costs={m:prior['models'][m]['physical']['configured_known_cny']/prior['models'][m]['physical']['physical_attempts']for m in MODELS}
    policies=policy_commitment(rows,frozen['rates'],costs)
    names=('native_review_diagnostic_live.py','native_reviewer_calibration.py','native_candidate_challenge_focused.py','native_candidate_challenge.py','native_optimization_runtime.py','transport_d2.py','native_quality_review_game.py','expected_review_value.py','review_model_profiles.py')
    plan={'version':'prospective-quality-allocation-potential-outcomes-v1','packets':str(packets.resolve()),'tasks':[r['task_id']for r in rows],
        'inputs_sha256':sha(packets/'pair64.inputs.json'),'states_sha256':sha(packets/'pair_states.json'),'frozen_rates_sha256':sha(BLIND/'frozen_fit_rates.json'),
        'policies_hash':fingerprint(policies),'models':list(MODELS),'profile_caps_cny':{m:1.5 for m in MODELS},'total_configured_upper_cny':4.5,
        'physical_cap_per_provider':70,'potential_outcomes':192,'scope':'real model calls; precommitted policy replay, not adaptive online cooperation',
        'label_stage':'only after all provider generation seals','source_sha256':{n:sha(ROOT/'experiments/paired_deliberation'/n)for n in names}}
    create(root/'plan.json',plan);create(root/'policy_commitment.json',policies)
    for n,h in plan['source_sha256'].items():
        p=root/'source_snapshot'/n
        create(p.parent/'hashes.json',plan['source_sha256'])
        if p.exists() and sha(p)!=h:raise ValueError('snapshot conflict')
        p.write_bytes((ROOT/'experiments/paired_deliberation'/n).read_bytes())
    return rows,states,plan


def collect_provider(model,rows,states,root,master):
    scope=root/model
    plan={'version':'native-review-diagnostic-potential-outcome','provider':model,'master_hash':fingerprint(master),'tasks':master['tasks'],'no_labels':True}
    create(scope/'plan.json',plan);api=Runtime(scope,plan,cap=70,max_cny=1.5,provider=model);results={}
    def one(i,row):
        path=scope/'cases'/('case_'+str(i)+'.json')
        if path.exists():return load(path)
        req=request(row);state=states[row['task_id']];event={'task_id':row['task_id'],'id':row['case_id'],'pair':row['pair'],'labels_available':False,'request_hash':fingerprint(req)}
        try:
            rec=api.call('review_'+str(i),req);answer,ops=normalize(rec['report'],row['pair'])
            sources=json.loads(req['messages'][1]['content'])['sources']
            bound=apply_response(state,row['target'],answer,sources,row['case_id'],fingerprint(state))
            event.update(report=answer,representation_operations=ops,bound=bound,valid=bound['valid'],effective_swap=bound.get('changed',False),abstained=bound.get('abstained',False),usage=rec['usage'],model=rec['model'])
        except (ValueError,RuntimeError,KeyError,TypeError,OSError) as exc:event.update(valid=False,effective_swap=False,abstained=False,error=str(exc))
        create(path,event);print(model,'completed',i+1,'/64','valid',event['valid'],'swap',event['effective_swap'],flush=True);return event
    with ThreadPoolExecutor(max_workers=2)as pool:
        jobs={pool.submit(one,i,r):i for i,r in enumerate(rows)}
        for f in as_completed(jobs):results[jobs[f]]=f.result()
    create(scope/'generation_complete.json',{'hashes':[fingerprint(results[i])for i in range(len(rows))],'labels_available':False})
    return [results[i]for i in range(len(rows))]


def run(root=OUT,packets=PACKETS):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    rows,states,plan=prepare(root,packets);root=Path(root)
    with ThreadPoolExecutor(max_workers=3)as pool:
        jobs=[pool.submit(collect_provider,m,rows,states,root,plan)for m in MODELS]
        for f in jobs:f.result()
    create(root/'generation_complete.json',{'models':MODELS,'tasks':plan['tasks'],'policies_hash':plan['policies_hash'],'labels_available':False})
    return score(root)


def score(root=OUT):
    root=Path(root);plan=load(root/'plan.json');seal=load(root/'generation_complete.json');packets=Path(plan['packets'])
    rows=load(packets/'pair64.inputs.json');states=load(packets/'pair_states.json');policies=load(root/'policy_commitment.json')
    if sha(packets/'pair64.inputs.json')!=plan['inputs_sha256'] or sha(packets/'pair_states.json')!=plan['states_sha256'] or fingerprint(policies)!=plan['policies_hash']:raise ValueError('frozen inputs/strategy changed')
    records={}
    for m in MODELS:
        rr=[load(root/m/'cases'/('case_'+str(i)+'.json'))for i in range(len(rows))]
        if load(root/m/'generation_complete.json')['hashes']!=[fingerprint(r)for r in rr]:raise ValueError('response seal mismatch')
        records[m]={r['task_id']:r for r in rr}
    # The first diagnostic-label read occurs only after every outcome is sealed.
    from experiments.paired_deliberation.native_review_calibration_data import read_fit_targets,OFFICIAL
    raw=load(ROOT/'experiments/results/native_review_calibration_v2/fold-serving-data/valid_data.json')
    labels=read_fit_targets(raw,[r['case_id']for r in rows],load(OFFICIAL/'entity2id.json'))
    create(root/'evaluation_labels.after_generation.json',labels)
    methods={};per_case={};total=sum(len(labels[r['case_id']])for r in rows)
    names=list(policies['groups'][0]['policies'])
    for name in names:
        chosen=[a for g in policies['groups']for a in g['policies'][name]]
        applied={};logical_cost=0.
        for a in chosen:
            logical_cost+=a['forecast_cost_cny'];old=applied.get(a['task_id'])
            if old is None or (a['gain'],-MODELS.index(a['provider']))>(old['gain'],-MODELS.index(old['provider'])):applied[a['task_id']]=a
        hits=[];basehits=[];gains=losses=0;valid=swaps=0;cost_actual_est=0.
        for a in chosen:
            event=records[a['provider']][a['task_id']];u=event.get('usage',{});p=PROFILES[a['provider']]
            # Physical receipts remain authoritative; profile rates for logical usage are in summary below.
        for row in rows:
            task=row['task_id'];rank=list(states[task]);base=list(rank);a=applied.get(task)
            if a:
                event=records[a['provider']][task];valid+=event['valid']
                if event['effective_swap']:
                    x,y=row['pair'];i,j=rank.index(x),rank.index(y);rank[i],rank[j]=rank[j],rank[i];swaps+=1
            targets=list(map(str,labels[row['case_id']]))
            h=sum(t in rank[:10]for t in targets);b=sum(t in base[:10]for t in targets)
            hits.append(h);basehits.append(b);gains+=sum(t in rank[:10]and t not in base[:10]for t in targets);losses+=sum(t not in rank[:10]and t in base[:10]for t in targets)
        methods[name]={'hits_at_10':sum(hits),'item_targets':total,'recall_at_10':sum(hits)/total,'new_targets_vs_base':gains,'lost_targets_vs_base':losses,
            'logical_calls':len(chosen),'distinct_queries_reviewed':len(applied),'valid_applied_reviews':int(valid),'effective_swaps':int(swaps),'forecast_logical_cost_cny':logical_cost,
            'cost_note':'forecast from frozen prior fit unit costs; not actual new physical bill'}
        per_case[name]=hits
    rng=np.random.default_rng(20261028);den=np.array([len(labels[r['case_id']])for r in rows]);boot=rng.integers(0,len(rows),size=(20000,len(rows)))
    comparisons={}
    for left,right in [('game','base'),('game','central'),('game','fixed_qwen_plus'),('game','fixed_qwen_flash')]:
        d=np.array(per_case[left])-np.array(per_case[right]);vals=d[boot].sum(axis=1)/den[boot].sum(axis=1)
        comparisons[left+'_vs_'+right]={'delta_recall_at_10':float(d.sum()/total),'conversation_cluster_bootstrap95':np.quantile(vals,[.025,.975]).tolist(),'positive_case_deltas':int((d>0).sum()),'negative_case_deltas':int((d<0).sum())}
    summary={'scope':plan['scope'],'methods':methods,'comparisons':comparisons,'physical':{m:audit(root/m,m)for m in MODELS},
        'reviewer_outcomes':{m:{'valid':sum(e['valid']for e in records[m].values()),'effective_swaps':sum(e['effective_swap']for e in records[m].values())}for m in MODELS},
        'warning':'Full192 outcomes paid for evaluation; each replayed policy at most48calls. No semantic-truth certification; count targets are next-recommender mentions.',
        'policy_seal_before_API':plan['policies_hash'],'generation_complete_hash':sha(root/'generation_complete.json')}
    create(root/'per_case_hits.json',per_case);create(root/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--packets',type=Path,default=PACKETS);p.add_argument('--run',action='store_true');p.add_argument('--score-only',action='store_true');a=p.parse_args()
    if a.score_only:score(a.output)
    elif a.run:run(a.output,a.packets)
    else:
        rows,states,plan=prepare(a.output,a.packets);print(json.dumps(plan,ensure_ascii=False,indent=2))
