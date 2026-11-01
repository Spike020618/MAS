"""No-label, no-API action reachability audit; synthetic identical initials."""
import json
from pathlib import Path
from experiments.paired_deliberation.native_candidate_challenge import build_targets,consensus_ranking,allocate,challenge_request,apply_response
from experiments.paired_deliberation.native_entity_evidence import NativeEvidence
from experiments.consensus_recommendation.common import fingerprint,sha,ROOT
from experiments.paired_deliberation.budget_v6 import AttemptClient


def run():
    source=ROOT/'experiments/results/native_retrieval_optimization_v1/old32.inputs.json'
    raw=json.loads(source.read_text());model=NativeEvidence();out=[]
    for original in raw:
        row=model.augment(original,history_limit=3,kg_limit=3)
        pool=[str(m['movie_id']) for m in row['candidates']];initial=[pool[:10] for _ in range(3)]
        base=consensus_ranking(initial,pool);targets=build_targets(initial,pool,row['candidate_evidence']);state=list(base);events=[];max_prompt=0
        compact={'id':row['id'],'text':row['text'],'candidates':[{k:m[k] for k in ('movie_id','title','score') if k in m} for m in row['candidates']]}
        requests=[[challenge_request(compact,initial,t,i,row['sources']) for t in targets] for i in range(3)]
        costs=[[.1*(sum(len(m['content'].encode())+512 for m in req['messages'])/64000+req['max_tokens']/4096) for req in rs] for rs in requests]
        for t in targets:
            req=challenge_request(compact,initial,t,0,row['sources']);max_prompt=max(max_prompt,AttemptClient.upper(req));sid='train-score:'+t['b']
            event=apply_response(state,t,{'preferred':t['b'],'source_id':sid},row['sources'],row['id'],fingerprint(state));state=event['ranking']
            events.append({'b':t['b'],'new_to_all_initial':all(t['b'] not in r for r in initial),'valid':event['valid'],
                'enters_top10':t['b'] in state[:10],'protected_first3_unchanged':state[:3]==base[:3],
                'nonzero_predictive_features':row['candidate_evidence'][t['b']]['neighbor_score']>0 or row['candidate_evidence'][t['b']]['association_score']>0})
        game=allocate(targets,costs,'game');central=allocate(targets,costs,'central')
        out.append({'id':row['id'],'targets':len(targets),'events':events,'max_prompt_upper':max_prompt,'game_calls':len(game['allocation']),
            'central_calls':len(central['allocation']),'game_proxy_gap':game['central_potential_gap'],'game_max_unilateral_gain':game['max_unilateral_gain']})
    summary={'scope':'label-free synthetic action reachability, NOT model performance; all3 synthetic initials equal retrieval first10',
        'input_path':str(source),'input_sha256':sha(source),'source_sha256':sha(ROOT/'experiments/paired_deliberation/native_candidate_challenge.py'),
        'cases':len(out),'total_targets':sum(r['targets'] for r in out),'new_to_all_initial_targets':sum(e['new_to_all_initial'] for r in out for e in r['events']),
        'source_bound_updates_valid':sum(e['valid'] for r in out for e in r['events']),'promoted_into_top10':sum(e['enters_top10'] for r in out for e in r['events']),
        'protected_prefix_always_unchanged':all(e['protected_first3_unchanged'] for r in out for e in r['events']),
        'targets_with_nonzero_predictive_features':sum(e['nonzero_predictive_features'] for r in out for e in r['events']),
        'max_prompt_upper':max(r['max_prompt_upper'] for r in out),'game_max_unilateral_gain':max(r['game_max_unilateral_gain'] for r in out),
        'max_central_proxy_gap':max(r['game_proxy_gap'] for r in out),'no_labels_read':True,'no_api_calls':True}
    root=ROOT/'experiments/results/native_candidate_challenge_offline_v1'
    root.mkdir(parents=True,exist_ok=True)
    for name,obj in [('cases.json',out),('summary.json',summary)]:
        (root/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':run()
