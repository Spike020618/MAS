"""Heterogeneous native experts, four-API-call controls, default dry preflight.

Two trained numerical predictors and one context LM form a low-cost reference.
Every LM control sees BOTH predictor score tables from its first call, plus the
same union pool and context/training/KG material. This is development research;
heterogeneity/Borda/game selection does not imply quality gain.
"""
import argparse,json,math
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.native_predictor_fusion import RESULT as FUSION,PrecomputedPredictorFusion
from experiments.paired_deliberation.native_entity_evidence import NativeEvidence
from experiments.paired_deliberation.native_candidate_challenge import consensus_ranking,build_targets,allocate,apply_response
from experiments.paired_deliberation.native_candidate_challenge_focused import challenge_request
from experiments.paired_deliberation.native_optimization_runtime import Runtime,audit
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.native_redial_benchmark import evaluate

METHODS=('sparse','kbrd','rrf2','single1','single4','independent4','hetero_base','central_challenge','game_challenge')
FOCI=('Use the current dialogue to reconcile the TWO train-fitted predictor score tables; neither predictor is automatically correct.',
      'Recheck current seeker statements and preserve supported choices; both predictor tables and all evidence remain available.',
      'Examine Sparse-only and KBRD-only disagreements against the same contextual evidence, without rewarding agreement.',
      'Make the final next-recommendation prediction from the complete same material; do not treat another user outcome as current preference.')
SOURCE_NAMES=('native_heterogeneous_consensus.py','native_candidate_challenge.py','native_candidate_challenge_focused.py','native_optimization_runtime.py',
 'native_entity_evidence.py','native_kbrd_scores.py','native_predictor_fusion.py','review_allocation_game.py','transport_d2.py')


def _source(name,case_id,mid,columns,values,provenance):
    return {'source_id':'predictor:'+name+':'+mid,'kind':'model_prediction','allowed_use':'prediction_basis',
        'origin_case_id':'released_train_fitted_predictor','query_case_id':case_id,'movie_id':mid,
        'text':json.dumps({'predictor':name,'movie_id':mid,'columns':columns,'values':values,
           'interpretation':'train-fitted uncertain forecast for the visible prefix, not current-user statement or semantic truth'},separators=(',',':')),
        'provenance':provenance,'current_user_fact_authority':False,'semantic_truth_certified':False}


def prepare_row(original,entities,predictors):
    row=entities.augment(original,history_limit=3,kg_limit=3)
    ids=[str(m['movie_id']) for m in row['candidates']]
    values=predictors.score_pool(row['id'],ids)
    sparse={r[0]:r for r in values['rows']};kb={r[0]:r for r in values['kbrd']['rows']}
    # Prior is the fixed equal-weight RRF60, not the small-dev selected mixture.
    rrf={m:.5/(60+sparse[m][2])+.5/(60+kb[m][3]) for m in ids}
    ordered=sorted(ids,key=lambda m:(-rrf[m],sparse[m][2]))
    by_id={str(m['movie_id']):m for m in row['candidates']}
    row['candidates']=[{'movie_id':m,'title':by_id[m]['title'],'score':rrf[m]} for m in ordered]
    row['predictor_scores']={m:{'sparse_score':sparse[m][1],'sparse_native_rank':sparse[m][2],
       'kbrd_logit':kb[m][1],'kbrd_native_z':kb[m][2],'kbrd_native_rank':kb[m][3],'rrf60_equal_score':rrf[m]} for m in ordered}
    # Candidate-scoped cards keep pair requests small and preserve original roles.
    for m in ordered:
        row['sources']['predictor:sparse:'+m]=_source('sparse',row['id'],m,['score','native_rank'],[sparse[m][1],sparse[m][2]],{'fit_split':'train','sparse_config':'native_retrieval_inner_v1'})
        row['sources']['predictor:kbrd:'+m]=_source('kbrd',row['id'],m,['logit','native_z','native_rank'],[kb[m][1],kb[m][2],kb[m][3]],
            {'fit_split':'train','checkpoint_epoch':values['kbrd']['checkpoint_epoch'],'selection_metric':values['kbrd']['selection_metric']})
    row['expert_rankings']={'sparse':sorted(ordered,key=lambda m:sparse[m][2]),'kbrd':sorted(ordered,key=lambda m:kb[m][3]),'rrf2':ordered}
    row['predictor_policy']='equal weight RRF k=60; no fitting on current outcomes'
    return row


def payload(row):
    ids=[m['movie_id'] for m in row['candidates']]
    return {'id':row['id'],'text':row['text'],'candidates':row['candidates'],
        'candidate_evidence':row['candidate_evidence'],'training_neighbors':row['training_neighbors'],
        'predictor_columns':['movie_id','sparse_score','sparse_native_rank','kbrd_logit','kbrd_native_z','kbrd_native_rank'],
        'predictor_rows':[[m,*[row['predictor_scores'][m][k] for k in ('sparse_score','sparse_native_rank','kbrd_logit','kbrd_native_z','kbrd_native_rank')]] for m in ids],
        'score_authority':'both predictors trained on train; these are uncertain forecasts, NOT current user facts'}


def target_features(row):
    ids=[m['movie_id'] for m in row['candidates']];scores=row['predictor_scores']
    support={m:.5/(60+scores[m]['sparse_native_rank'])+.5/(60+scores[m]['kbrd_native_rank']) for m in ids}
    peak=max(support.values()) or 1
    z=np.array([scores[m]['kbrd_native_z'] for m in ids]);low=float(z.min());span=max(float(z.max())-low,1e-12)
    return {m:{**row['candidate_evidence'][m],'evidence_score':support[m]/peak,
        'kg_score':max(0.,(scores[m]['kbrd_native_z']-low)/span)} for m in ids}


def rank_preflight(payload_value,previous=None):
    """Reproduce Runtime.rank's request shape without constructing a provider."""
    content={**payload_value,'task':FOCI[0],'schema':{'ranking':['ten supplied movie ID strings']}}
    features=content.pop('candidate_evidence',{})
    content['candidate_evidence_columns']=['movie_id','neighbor_score','association_score','training_count','KG_edges_as_relation_id_and_entity_name_NOT_inferred_relation']
    content['candidate_evidence_rows']=[[m,f.get('neighbor_score',0),f.get('association_score',0),f.get('training_count',0),[[e['relation_id'],e['entity_name']] for e in f.get('kg_neighbors',[])]] for m,f in features.items()]
    if previous is not None:content['your_previous_ranking']=previous[:10]
    # Upper bound the actual fixed Runtime system message by a generous margin.
    return RequestsV5.envelope('ranking_v6',content,'x'*2048,max_tokens=512)


def preflight_rows(rows):
    records=[]
    for row in rows:
        pool=[m['movie_id'] for m in row['candidates']];initial=[row['expert_rankings']['sparse'][:10],row['expert_rankings']['kbrd'][:10],pool[:10]]
        targets=build_targets(initial,pool,target_features(row),max_targets=6)
        # Full predictor tables already in every ranking prompt. Pair requests
        # retain full same pool/order plus pair-specific numerical source cards.
        request_row={'id':row['id'],'text':row['text'],'candidates':row['candidates']}
        requests=[[challenge_request(request_row,initial,t,i,row['sources']) for t in targets] for i in range(3)]
        rank_upper=max(AttemptClient.upper(rank_preflight(payload(row))),AttemptClient.upper(rank_preflight(payload(row),pool[:10])))
        pair_upper=max((AttemptClient.upper(req) for player in requests for req in player),default=0)
        costs=[[.1*(sum(len(m['content'].encode())+512 for m in req['messages'])/64000+req['max_tokens']/4096) for req in player] for player in requests]
        central=allocate(targets,costs,'central');game=allocate(targets,costs,'game')
        if rank_upper>64000 or pair_upper>64000:raise ValueError('request upper exceeds transport limit before any network dispatch')
        if len(pool)>200 or len(pool)!=len(set(pool)):raise ValueError('unique shared predictor union <=200 required')
        records.append({'id':row['id'],'pool_size':len(pool),'rank_upper_conservative':rank_upper,'pair_upper':pair_upper,
            'targets':len(targets),'synthetic_central_calls':len(central['allocation']),'synthetic_game_calls':len(game['allocation'])})
    return {'pass':True,'no_api_calls':True,'no_evaluation_labels_read':True,'case_count':len(rows),'records':records,
        'max_rank_upper_conservative':max(r['rank_upper_conservative'] for r in records),'max_pair_upper':max(r['pair_upper'] for r in records),
        'scope':'synthetic LM ranking equals RRF prior for shape check only, not performance'}


def prepare(input_path,label_path,root,count=8,reuse_root=None,budget_cny=2.):
    if not math.isfinite(budget_cny) or budget_cny<=0:raise ValueError('finite positive collection budget required')
    input_path=Path(input_path);root=Path(root);raw=json.loads(input_path.read_text())[:count]
    if len(raw)!=count:raise ValueError('requested input count unavailable')
    if (root/'inputs.json').exists():rows=json.loads((root/'inputs.json').read_text())
    else:
        entities=NativeEvidence();predictors=PrecomputedPredictorFusion();rows=[prepare_row(r,entities,predictors) for r in raw]
        create(root/'inputs.json',rows)
    if [r['id'] for r in rows]!=[r['id'] for r in raw]:raise ValueError('frozen input IDs differ')
    plan={'version':'native-heterogeneous-consensus-v2-focused-offered-source','count':count,'ids':[r['id'] for r in rows],
        'inputs_fingerprint':fingerprint(rows),'input_source_sha256':sha(input_path),'input_source':str(input_path.resolve()),
        'label_source':str(Path(label_path).resolve()),'methods':list(METHODS),'max_api_calls_main_arm':4,
        'shared_initial':'one same full-information LM response shared; Sparse/KBRD are train-fitted numerical experts',
        'same_information':'all LM calls from FIRST call see both predictor score tables, union pool, context, public KG and training material',
        'predictor_costs':'training and CPU inference shared and disclosed; not free private information for multi-agent arms',
        'rank_readout':'Borda top10 over Sparse, KBRD, context-LM; may dilute a stronger individual predictor',
        'rrf2':'fixed equal weights, k60, within shared union; no new64 label selection',
        'allocation_utility':'fixed public retrieval/evidence/disagreement heuristic, not true accuracy reward',
        'fallback':'bad rank keeps shared RRF prior or previous state; bad challenge keeps exact prior state, same policy across controls',
        'source_sha256':{n:sha(ROOT/'experiments/paired_deliberation'/n) for n in SOURCE_NAMES},
        'challenge_contract':'pair-focused request; response source ID must be in this exact request offered sources',
        'generation_labels_available':False,'scope':'reused public development; no untouched confirmation claim',
        'reuse_root':str(Path(reuse_root).resolve()) if reuse_root else None,'workers':2,'physical_cap':13*count+16,
        'configured_max_cny':budget_cny,'global_qwen_dispatch_interval_seconds':3,'no_automatic_timeout_retry':True}
    create(root/'plan.json',plan)
    for n in SOURCE_NAMES:
        dst=root/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and sha(dst)!=plan['source_sha256'][n]:raise ValueError('source changed after freeze')
        dst.write_bytes((ROOT/'experiments/paired_deliberation'/n).read_bytes())
    check=preflight_rows(rows);create(root/'preflight.json',check)
    return plan,rows,check


def case(index,row,api,root):
    path=root/'cases'/('case_'+str(index)+'.json')
    if path.exists():return json.loads(path.read_text())
    pool=[m['movie_id'] for m in row['candidates']];data=payload(row);prefix='case'+str(index)
    common=api.rank(prefix+'_common_lm',data,FOCI[0],fallback=pool)
    sparse=row['expert_rankings']['sparse'];kbrd=row['expert_rankings']['kbrd'];initial=[sparse[:10],kbrd[:10],common['ranking']]
    base=consensus_ranking(initial,pool)
    full=lambda r:list(r)+[m for m in pool if m not in r]
    arms={m:{'ranking':row['expert_rankings'][m],'calls':[]} for m in ('sparse','kbrd','rrf2')}
    arms['single1']={'ranking':full(common['ranking']),'calls':[common]}
    arms['hetero_base']={'ranking':base,'calls':[common],'numerical_experts':['sparse','kbrd']}
    own=[common];previous=common['ranking']
    for i in range(1,4):
        call=api.rank(prefix+'_single'+str(i),data,FOCI[i],previous=previous,fallback=previous);own.append(call);previous=call['ranking']
    arms['single4']={'ranking':full(previous),'calls':own}
    independent=[common]+[api.rank(prefix+'_independent'+str(i),data,FOCI[i],fallback=pool) for i in range(1,4)]
    arms['independent4']={'ranking':consensus_ranking([r['ranking'] for r in independent],pool),'calls':independent}
    features=target_features(row);targets=build_targets(initial,pool,features,max_targets=6)
    request_row={'id':row['id'],'text':row['text'],'candidates':row['candidates']}
    requests=[[challenge_request(request_row,initial,t,i,row['sources']) for t in targets] for i in range(3)]
    costs=[[.1*(sum(len(m['content'].encode())+512 for m in req['messages'])/64000+req['max_tokens']/4096) for req in player] for player in requests]
    for mode in ('central','game'):
        allocation=allocate(targets,costs,mode);method=mode+'_challenge';state=list(base);events=[]
        commit={'targets':targets,'costs':costs,'allocation':allocation,'initial_hash':fingerprint(base),'requests_hash':fingerprint(requests),'labels_available':False}
        create(root/'commitments'/(prefix+'_'+mode+'.json'),commit)
        for reviewer,q in allocation['allocation']:
            req=requests[reviewer][q];slot=prefix+'_'+mode+'_p'+str(reviewer)+'_q'+str(q)
            try:
                record=api.call(slot,req);offered=json.loads(req['messages'][1]['content'])['sources']
                event=apply_response(state,targets[q],record['report'],offered,row['id'],fingerprint(state))
                event.update(slot=slot,usage=record['usage'],reviewer=reviewer,target=targets[q])
            except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:
                event={'ranking':list(state),'valid':False,'accepted':False,'changed':False,'slot':slot,'error':str(exc),'reviewer':reviewer,'target':targets[q]}
            state=event['ranking'];events.append(event)
        arms[method]={'ranking':state,'calls':[common]+events,'events':events,'allocation':allocation}
    for arm in arms.values():
        if len(arm['ranking'])!=len(pool) or set(arm['ranking'])!=set(pool):raise AssertionError('full shared pool permutation required')
        arm['all_calls_valid']=all(c.get('valid',True) for c in arm['calls'])
        if len(arm['calls'])>4:raise AssertionError('four API-call arm cap')
    result={'id':row['id'],'initial_experts':{'sparse':sparse[:10],'kbrd':kbrd[:10],'context_lm':common},
        'methods':arms,'generation_labels_available':False}
    create(path,result);print('completed',index,row['id'],{m:a['all_calls_valid'] for m,a in arms.items()},flush=True);return result


def score(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());seal=json.loads((root/'generation_complete.json').read_text())
    rows=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text()) for i in range(plan['count'])]
    if seal['hashes']!=[fingerprint(r) for r in rows] or [r['id'] for r in rows]!=plan['ids']:raise ValueError('sealed generation identity mismatch')
    all_labels={r['id']:r for r in json.loads(Path(plan['label_source']).read_text())};labels=[all_labels[r['id']] for r in rows]
    summary={'methods':{},'physical':audit(root),'scope':plan['scope'],'counterfactual':'all controls have both predictor score tables from first call',
        'attribution':'numerical fusion and richer material are not multi-agent/game gains; compare same-info single4/independent4',
        'fallback':'uniform prior/preserved state; invalid outputs separately shown and strict-zero diagnostic included'}
    for method in METHODS:
        arms=[r['methods'][method] for r in rows];rankings=[[int(m) for m in a['ranking']] for a in arms]
        metrics=evaluate(labels,rankings);metrics.update(logical_api_calls=sum(len(a['calls']) for a in arms),
            all_calls_valid_contexts=sum(a['all_calls_valid'] for a in arms),failed_calls=sum(not c.get('valid',True) for a in arms for c in a['calls']),
            strict_zero_diagnostic=evaluate(labels,[rank if arm['all_calls_valid'] else [] for rank,arm in zip(rankings,arms)]))
        if method.endswith('challenge'):
            events=[e for a in arms for e in a['events']];metrics.update(reviews=len(events),accepted=sum(e['valid'] for e in events),changed=sum(e.get('changed',False) for e in events),
                max_proxy_unilateral_gain=max((a['allocation']['selected_max_unilateral_gain'] for a in arms),default=0),
                max_game_central_proxy_gap=max((a['allocation']['central_potential_gap'] for a in arms),default=0))
        summary['methods'][method]=metrics
    create(root/'summary.json',summary);print(json.dumps(summary,indent=2));return summary


def run(plan,rows,root):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    for n,h in plan['source_sha256'].items():
        if sha(ROOT/'experiments/paired_deliberation'/n)!=h:raise ValueError('source changed since preflight')
    api=Runtime(root,plan,cap=plan['physical_cap'],max_cny=plan['configured_max_cny']);out={}
    with ThreadPoolExecutor(max_workers=plan['workers']) as workers:
        jobs={workers.submit(case,i,row,api,root):i for i,row in enumerate(rows)}
        for future in as_completed(jobs):out[jobs[future]]=future.result()
    create(root/'generation_complete.json',{'hashes':[fingerprint(out[i]) for i in range(len(rows))],'labels_available':False})
    return score(root)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',default=str(FUSION/'old32.union.inputs.json'))
    parser.add_argument('--labels',default=str(ROOT/'experiments/results/native_retrieval_inner_v1/old32.labels.json'))
    parser.add_argument('--output',default=str(ROOT/'experiments/results/native_heterogeneous_old8_v2'))
    parser.add_argument('--count',type=int,default=8);parser.add_argument('--budget-cny',type=float,default=2.); parser.add_argument('--reuse-root');parser.add_argument('--run',action='store_true');parser.add_argument('--score-only',action='store_true')
    args=parser.parse_args();root=Path(args.output)
    if args.score_only:score(root)
    else:
        plan,rows,check=prepare(args.inputs,args.labels,root,args.count,args.reuse_root,args.budget_cny)
        print(json.dumps(check,indent=2))
        if args.run:run(plan,rows,root)
