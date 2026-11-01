"""Independent read-only replay and usage audit of real diagnostic outcomes."""
import argparse,json
import numpy as np
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_redial_benchmark import evaluate
from experiments.paired_deliberation.review_model_profiles import PROFILES
from experiments.paired_deliberation.native_review_diagnostic_live import OUT,MODELS,load


def known_usage(root,provider,task,index):
    scope=root/provider
    event=load(scope/'cases'/('case_'+str(index)+'.json'))
    if event['task_id']!=task:raise ValueError('event task mismatch')
    h=event['request_hash'];raw=scope/'slots'/('request_'+h[:32]+'.raw.json')
    usage={'prompt_tokens':0,'completion_tokens':0};status='unknown';known=False
    if raw.exists():
        wrapper=load(raw)
        if wrapper['integrity']!=fingerprint(wrapper['raw']) or fingerprint(wrapper['request'])!=h:raise ValueError('paid raw integrity mismatch')
        status=wrapper['raw']['status_code']
        try:
            u=json.loads(wrapper['raw']['body']).get('usage',{})
            known=bool(u)
            for k in usage:usage[k]=int(u.get(k,0))
        except (ValueError,TypeError):pass
    p=PROFILES[provider]
    cost=(usage['prompt_tokens']*p['input_cny_per_million']+usage['completion_tokens']*p['output_cny_per_million'])/1e6
    return {'usage':usage,'known_usage':known,'http_status':status,'configured_known_cny':cost,'event':event}


def audit(root=OUT):
    root=Path(root);plan=load(root/'plan.json');seal=load(root/'generation_complete.json');packets=Path(plan['packets'])
    policies=load(root/'policy_commitment.json');rows=load(packets/'pair64.inputs.json');states=load(packets/'pair_states.json')
    if seal['tasks']!=plan['tasks'] or seal['models']!=list(MODELS) or seal['policies_hash']!=plan['policies_hash'] or fingerprint(policies)!=plan['policies_hash']:raise ValueError('global seal mismatch')
    if sha(packets/'pair64.inputs.json')!=plan['inputs_sha256'] or sha(packets/'pair_states.json')!=plan['states_sha256']:raise ValueError('input mismatch')
    if sha(ROOT/'experiments/paired_deliberation/review_model_profiles.py')!=plan['source_sha256']['review_model_profiles.py']:raise ValueError('price profiles changed; audit with original source')
    labels=load(root/'evaluation_labels.after_generation.json');lookup={r['task_id']:i for i,r in enumerate(rows)}
    receipts={m:{r['task_id']:known_usage(root,m,r['task_id'],i)for i,r in enumerate(rows)}for m in MODELS}
    for m in MODELS:
        expected=load(root/m/'generation_complete.json')['hashes']
        if expected!=[fingerprint(receipts[m][r['task_id']]['event'])for r in rows]:raise ValueError('provider seal mismatch')
    out={'scope':'independent full-ranking replay plus actual response token accounting; no API calls',
         'labels_sha256':sha(root/'evaluation_labels.after_generation.json'),'master_plan_sha256':sha(root/'plan.json'),'methods':{}}
    actionsets={method:[a for g in policies['groups']for a in g['policies'][method]]for method in policies['groups'][0]['policies']}
    secondary_path=root/'secondary_cost_sensitivity.prelabel.json'
    if secondary_path.exists():
        secondary=load(secondary_path)
        if secondary['labels_exist_at_commit'] or secondary['outcome_data_read_for_selection'] or secondary['parent_policy_sha256']!=sha(root/'policy_commitment.json'):raise ValueError('secondary commitment invalid')
        out['secondary_prelabel_commitment_sha256']=sha(secondary_path)
        out['secondary_scope']='Prelabel sensitivity only; never select best result as primary.'
        for weight,groups in secondary['variants'].items():
            for method in ('game','central'):
                actionsets['secondary_lambda_'+weight+'_'+method]=[a for g in groups for a in g[method]]
    per_case_hits={}
    for method,actions in actionsets.items():
        chosen={}
        total={'prompt_tokens':0,'completion_tokens':0};known_cost=0.;unknown=[]
        for a in actions:
            m,t=a['provider'],a['task_id'];receipt=receipts[m][t]
            for k in total:total[k]+=receipt['usage'][k]
            known_cost+=receipt['configured_known_cny']
            if not receipt['known_usage']:unknown.append({'provider':m,'task_id':t,'http_status':receipt['http_status']})
            old=chosen.get(t)
            if old is None or (a['gain'],-MODELS.index(m))>(old['gain'],-MODELS.index(old['provider'])):chosen[t]=a
        rankings=[];new_targets=lost_targets=0
        for r in rows:
            t=r['task_id'];rank=list(states[t])
            if t in chosen:
                event=receipts[chosen[t]['provider']][t]['event']
                if event['effective_swap']:
                    a,b=r['pair'];i,j=rank.index(a),rank.index(b);rank[i],rank[j]=rank[j],rank[i]
            targets=list(map(str,labels[r['case_id']]))
            base=states[t]
            new_targets+=sum(x in rank[:10]and x not in base[:10]for x in targets)
            lost_targets+=sum(x not in rank[:10]and x in base[:10]for x in targets)
            rankings.append([int(x)for x in rank])
        metrics=evaluate([{'id':r['case_id'],'targets':labels[r['case_id']]}for r in rows],rankings)
        per_case_hits[method]=[sum(int(t)in rank[:10]for t in labels[r['case_id']])for r,rank in zip(rows,rankings)]
        out['methods'][method]={'metrics':metrics,'logical_calls':len(actions),'new_targets_vs_base':new_targets,'lost_targets_vs_base':lost_targets,'actual_response_tokens':total,'configured_known_logical_cny':known_cost,
            'calls_with_unknown_usage':unknown,'price_scope':'configured model rates applied to actual raw returned token counts; not actual platform invoice'}
    rng=np.random.default_rng(20261028)
    draws=rng.integers(0,len(rows),size=(20000,len(rows)))
    denominator=np.array([len(labels[r['case_id']])for r in rows])
    out['paired_vs_base']={}
    for method,casehits in per_case_hits.items():
        delta=np.asarray(casehits)-np.asarray(per_case_hits['base'])
        boot=delta[draws].sum(axis=1)/denominator[draws].sum(axis=1)
        out['paired_vs_base'][method]={'delta_recall_at_10':float(delta.sum()/denominator.sum()),'conversation_cluster_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_replicates':20000}
    physical={'prompt_tokens':0,'completion_tokens':0,'configured_known_cny':0.,'unknown_usage_calls':0}
    for m in MODELS:
        for v in receipts[m].values():
            for k in ('prompt_tokens','completion_tokens'):physical[k]+=v['usage'][k]
            physical['configured_known_cny']+=v['configured_known_cny'];physical['unknown_usage_calls']+=not v['known_usage']
    out['physical_collection_192_potential_actions']=physical
    create(root/'independent_receipt_and_ranking_audit.json',out);print(json.dumps(out,ensure_ascii=False,indent=2));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);args=p.parse_args();audit(args.output)
