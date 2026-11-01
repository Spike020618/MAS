"""Train-only paired reviewer calibration, not semantic-truth certification.

Selected training strata (24 a-win,24 b-win,16 ties) do not estimate natural
winner prevalence. Effective swap/safe-retain and raw choice accuracy are separate.
"""
import argparse,copy,json,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_optimization_runtime import Runtime,audit
from experiments.paired_deliberation.review_model_profiles import PROFILES
from experiments.paired_deliberation.native_candidate_challenge_focused import challenge_request,apply_response
from experiments.paired_deliberation.native_calibrated_challenge import CalibratedChallengeModel
from experiments.paired_deliberation.budget_v6 import AttemptClient

DATA=ROOT/'experiments/results/native_review_calibration_v2'
FOLD=ROOT/'experiments/results/native_gain_calibration_v1'
OUT=ROOT/'experiments/results/native_reviewer_calibration_v1'
MODELS=('glm_flash','qwen_flash','qwen_plus')


def wilson(success,n,z=1.959963984540054):
    if not n:return [0.,1.]
    p=success/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-radius),min(1.,center+radius)]


def prepare_states(rows,root):
    ids=json.loads((FOLD/'hold20.ids.json').read_text());indices={c:i for i,c in enumerate(ids)}
    items=json.loads((FOLD/'native_item_ids.json').read_text());inputs=json.loads((FOLD/'hold20.inputs.json').read_text())
    model=CalibratedChallengeModel()
    arrays=[np.load(FOLD/path,mmap_mode='r') for path in ('sparse/hold20.native_scores.npy','sparse/hold20.native_ranks.npy','kbrd/hold20.native_logits.npy','kbrd/hold20.native_ranks.npy')]
    states={}
    for row in rows:
        i=indices[row['case_id']];pred=model.score_pool(inputs[i],items,*[matrix[i] for matrix in arrays]);state=pred['ranking']
        assert fingerprint(state)==row['target']['base_ranking_hash']
        assert state.index(row['target']['a'])==row['target']['a_position'] and state.index(row['target']['b'])==row['target']['b_position']
        states[row['task_id']]=state
    create(root/'pair_states.json',states);return states


def request(row):
    minimal={'id':row['case_id'],'text':row['text'],'candidates':row['pair_candidates']}
    req=challenge_request(minimal,[],row['target'],0,row['sources'])
    p=json.loads(req['messages'][1]['content']);p.pop('reviewer',None)
    p['calibrated_prior_positions']={'incumbent':row['target']['a_position']+1,'challenger':row['target']['b_position']+1}
    p['task']='Predict which of ONLY these two movies is more likely in the next recommender turn, using supplied evidence. Other movies are outside this task.'
    req['kind']='native_independent_evidence_review'
    req['messages'][1]['content']=json.dumps(p,ensure_ascii=False,sort_keys=True)
    return req


def normalize(answer,pair):
    obj=copy.deepcopy(answer);ops=[]
    if isinstance(obj,dict) and set(obj)=={'answer'} and isinstance(obj['answer'],dict) and set(obj['answer'])=={'preferred','source_id'}:
        obj=obj['answer'];ops.append('unwrap_exact_answer_object')
    if isinstance(obj,dict) and type(obj.get('preferred')) is int and str(obj['preferred']) in pair:
        obj['preferred']=str(obj['preferred']);ops.append('exact_integer_pair_id_to_string')
    return obj,ops


def run_provider(provider,rows,states,root,master):
    scope=root/provider
    plan={'version':'reviewer-count-calibration-v1','provider':provider,'profile':PROFILES[provider],
      'master_plan_hash':fingerprint(master),'tasks':[r['task_id'] for r in rows],'no_labels_in_generation':True,
      'output_normalization':'exact answer object wrapper/int pair ID only, no source substitution','unknown_retry':False}
    create(scope/'plan.json',plan);api=Runtime(scope,plan,cap=70,max_cny=1.5 if provider=='qwen_plus' else .5,provider=provider)
    results={}
    def collect(i,row):
        path=scope/'cases'/('case_'+str(i)+'.json')
        if path.exists():return json.loads(path.read_text())
        req=request(row);state=states[row['task_id']]
        event={'task_id':row['task_id'],'id':row['case_id'],'pair':row['pair'],'request_hash':fingerprint(req),'labels_available':False}
        try:
            rec=api.call('review_'+str(i),req);answer,ops=normalize(rec['report'],row['pair'])
            offered=json.loads(req['messages'][1]['content'])['sources'];bound=apply_response(state,row['target'],answer,offered,row['case_id'],fingerprint(state))
            event.update(report=answer,representation_operations=ops,bound=bound,valid=bound['valid'],effective_swap=bound.get('changed',False),
                abstained=bound.get('abstained',False),usage=rec['usage'],model=rec['model'])
        except (ValueError,RuntimeError,KeyError,TypeError,OSError) as exc:event.update(valid=False,effective_swap=False,abstained=False,error=str(exc))
        create(path,event);print(provider,i,'valid',event['valid'],'swap',event['effective_swap'],flush=True);return event
    with ThreadPoolExecutor(max_workers=2) as workers:
        jobs={workers.submit(collect,i,row):i for i,row in enumerate(rows)}
        for f in as_completed(jobs):results[jobs[f]]=f.result()
    create(scope/'generation_complete.json',{'hashes':[fingerprint(results[i]) for i in range(len(rows))],'labels_available':False})
    return [results[i] for i in range(len(rows))]


def score(root=OUT):
    plan=json.loads((root/'plan.json').read_text());inputs=json.loads((DATA/'pair64.inputs.json').read_text());labels=json.loads((DATA/'pair64.labels.json').read_text())
    assert [r['task_id'] for r in inputs]==[r['task_id'] for r in labels]
    out={'scope':'selected balanced train-derived fit strata, not natural prevalence, not semantic truth; separate natural-distribution diagnostic required','models':{}}
    all_results={}
    for provider in plan['models']:
        scope=root/provider;records=[json.loads((scope/'cases'/('case_'+str(i)+'.json')).read_text()) for i in range(len(inputs))]
        seal=json.loads((scope/'generation_complete.json').read_text());assert seal['hashes']==[fingerprint(r) for r in records]
        all_results[provider]=records;out['models'][provider]={'physical':audit(scope,provider),'strata':{}}
        for winner in ('a','b','tie'):
            pairs=[(r,y) for r,y in zip(records,labels) if y['winner']==winner];n=len(pairs)
            valid=sum(r['valid'] for r,y in pairs);swaps=sum(r['effective_swap'] for r,y in pairs)
            raw_correct=sum(r.get('report',{}).get('preferred')==y['pair'][1 if winner=='b' else 0] for r,y in pairs) if winner!='tie' else None
            count=swaps if winner=='b' else n-swaps if winner=='a' else None
            out['models'][provider]['strata'][winner]={'n':n,'valid':valid,'effective_swaps':swaps,'raw_winner_choice':raw_correct,
              'effective_correct_action':count,'effective_rate':count/n if count is not None else None,
              'effective_wilson95':wilson(count,n) if count is not None else None,'abstained':sum(r['abstained'] for r,y in pairs)}
        s=out['models'][provider]['strata'];out['models'][provider]['effective_parameters']={
          'recovery_if_b_observed_more':s['b']['effective_rate'],'safe_retention_if_a_observed_more':s['a']['effective_rate'],
          'warning':'safe retention includes rejected/no-op outputs; not claimed verifier accuracy; fit-stratum uncertainty and bias remain'}
    dual={}
    for left,right in [('glm_flash','qwen_flash'),('glm_flash','qwen_plus'),('qwen_flash','qwen_plus')]:
        if left not in all_results or right not in all_results:continue
        stats={}
        for winner in ('a','b','tie'):
            idx=[i for i,y in enumerate(labels) if y['winner']==winner];swaps=sum(all_results[left][i]['effective_swap'] and all_results[right][i]['effective_swap'] for i in idx)
            stats[winner]={'n':len(idx),'both_effective_swap':swaps,'rate':(swaps if winner=='b' else len(idx)-swaps)/len(idx) if winner!='tie' else None}
        dual[left+'_AND_'+right]={'rule':'swap only if both source-valid independent reviews prefer b; no independence assumption','strata':stats}
    out['paired_two_review_gates']=dual;create(root/'summary.json',out);print(json.dumps(out,indent=2));return out


def run(root=OUT):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    rows=json.loads((DATA/'pair64.inputs.json').read_text());models=list(MODELS)
    for model in models:
        probe=ROOT/'experiments/results/native_reviewer_transport_probe_v1'/model/'probe_result.json'
        if model!='qwen_flash' and (not probe.exists() or not json.loads(probe.read_text())['success']):raise RuntimeError('bounded provider probe required')
    root=Path(root);states=prepare_states(rows,root)
    reqs=[request(r) for r in rows];upper=max(AttemptClient.upper(r) for r in reqs)
    if upper>64000:raise ValueError('review material exceeds preflight upper')
    names=('native_reviewer_calibration.py','review_model_profiles.py','native_optimization_runtime.py','native_candidate_challenge_focused.py','native_candidate_challenge.py','transport_d2.py','native_calibrated_challenge.py')
    plan={'version':'reviewer-calibration-count-v1','models':models,'profiles':{m:PROFILES[m] for m in models},
       'input_sha256':sha(DATA/'pair64.inputs.json'),'tasks':[r['task_id'] for r in rows],'query_labels_available':False,
       'data':'fold80 evidence and teacher; source queries only review-cal-fit half of ranker diagnostic',
       'selection':'24 b/24 a/16 tie supervised fit strata, all64 source conversations distinct; not natural accuracy estimate',
       'max_prompt_upper':upper,'new_physical_cap':210,'configured_caps_cny':{'glm_flash':.5,'qwen_flash':.5,'qwen_plus':1.5},
       'source_sha256':{n:sha(ROOT/'experiments/paired_deliberation'/n) for n in names}}
    create(root/'plan.json',plan)
    for n in names:
        dst=root/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/'experiments/paired_deliberation'/n).read_bytes())
    with ThreadPoolExecutor(max_workers=3) as workers:
        jobs=[workers.submit(run_provider,m,rows,states,root,plan) for m in models]
        for job in jobs:job.result()
    create(root/'generation_complete.json',{'providers':models,'labels_available':False});return score(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT));p.add_argument('--score-only',action='store_true');a=p.parse_args()
    if a.score_only:score(Path(a.output))
    else:run(Path(a.output))
