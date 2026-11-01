"""Evidence-linked open-candidate optimization; same-info six-call controls."""
import argparse,json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_entity_evidence import NativeEvidence
from experiments.paired_deliberation.native_candidate_challenge_focused import consensus_ranking,build_targets,allocate,challenge_request,apply_response
from experiments.paired_deliberation.native_optimization_runtime import Runtime,audit
from experiments.paired_deliberation.native_redial_benchmark import evaluate

FOCI=('Process current user statements and linked film attributes.',
      'Process the OTHER training examples and evidence supporting candidates.',
      'Process training associations and identify likely shared omissions.',
      'Check which previously overlooked candidate has stronger evidence.',
      'Check chronology, explicit dislikes and evidence quality.',
      'Preserve good choices and make the final evidence-based prediction.')
METHODS=('retrieval','single1','single6','independent3','independent6','central_challenge','game_challenge')


def compact(row):
    # Same available evidence for every method; trim only deterministic KG list length.
    features={m:{k:v for k,v in f.items() if k in ('neighbor_score','association_score','training_count','kg_neighbors')}
              for m,f in row['candidate_evidence'].items()}
    return {'id':row['id'],'text':row['text'],
        'candidates':[{k:m[k] for k in ('movie_id','title','score') if k in m} for m in row['candidates']],
        'candidate_evidence':features,'training_neighbors':row['training_neighbors']}


def view(payload,i):
    out=dict(payload)
    if i==0:out['training_neighbors']=[]
    else:
        out['training_neighbors']=payload['training_neighbors'][i-1::2]
        out['candidate_evidence']={m:{k:v for k,v in f.items() if k!='kg_neighbors'} for m,f in payload['candidate_evidence'].items()}
    return out


def full_rank(top,pool):return list(top)+[m for m in pool if m not in top]


def case(index,row,api,root):
    path=root/'cases'/('case_'+str(index)+'.json')
    if path.exists():return json.loads(path.read_text())
    payload=compact(row);pool=[m['movie_id'] for m in row['candidates']];prefix='case'+str(index)
    initials=[api.rank(prefix+'_initial'+str(i),view(payload,i),FOCI[i],fallback=pool) for i in range(3)]
    ranks=[r['ranking'] for r in initials];base=consensus_ranking(ranks,pool)
    out={'id':row['id'],'initial':initials,'methods':{'retrieval':{'ranking':pool,'calls':[]},
        'independent3':{'ranking':base,'calls':initials}},'generation_labels_available':False}
    one=api.rank(prefix+'_single0',payload,FOCI[0],fallback=pool)
    out['methods']['single1']={'ranking':full_rank(one['ranking'],pool),'calls':[one]}
    own=[one];prev=one['ranking']
    for i in range(1,6):
        call=api.rank(prefix+'_single'+str(i),payload,FOCI[i],previous=prev,fallback=prev);own.append(call);prev=call['ranking']
    out['methods']['single6']={'ranking':full_rank(prev,pool),'calls':own}
    extra=[api.rank(prefix+'_independent'+str(i),payload,FOCI[i],fallback=pool) for i in (3,4,5)]
    out['methods']['independent6']={'ranking':consensus_ranking(ranks+[r['ranking'] for r in extra],pool),'calls':initials+extra}
    features={m:{**f,'evidence_score':f.get('association_score',0)} for m,f in row['candidate_evidence'].items()}
    targets=build_targets(ranks,pool,features,max_targets=6)
    requests=[[challenge_request(payload,ranks,t,i,row['sources']) for t in targets] for i in range(3)]
    costs=[[.1*(sum(len(m['content'].encode())+512 for m in req['messages'])/64000+req['max_tokens']/4096) for req in player] for player in requests]
    for mode in ('central','game'):
        allocation=allocate(targets,costs,mode=mode);method=mode+'_challenge';state=list(base);events=[]
        commit={'targets':targets,'allocation':allocation,'costs':costs,'request_hash':fingerprint(requests),'initial_hash':fingerprint(base),'labels_available':False}
        create(root/'commitments'/(prefix+'_'+mode+'.json'),commit)
        for reviewer,q in allocation['allocation']:
            req=requests[reviewer][q];slot=prefix+'_'+mode+'_p'+str(reviewer)+'_q'+str(q)
            try:
                record=api.call(slot,req)
                offered=json.loads(req['messages'][1]['content'])['sources']
                event=apply_response(state,targets[q],record['report'],offered,row['id'],expected_hash=fingerprint(state))
                event.update(slot=slot,usage=record['usage'],target=targets[q],reviewer=reviewer)
            except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:
                event={'ranking':list(state),'valid':False,'changed':False,'error':str(exc),'slot':slot,'target':targets[q],'reviewer':reviewer}
            state=event['ranking'];events.append(event)
        out['methods'][method]={'ranking':state,'calls':initials+events,'events':events,'allocation':allocation,
            'new_to_union_selected':sum(targets[q]['novel_to_initial_union'] for i,q in allocation['allocation'])}
    for arm in out['methods'].values():
        assert len(arm['ranking'])==len(pool) and set(arm['ranking'])==set(pool)
        arm['all_calls_valid']=all(c.get('valid',True) for c in arm['calls'])
    create(path,out);print('completed',index,row['id'],{m:a['all_calls_valid'] for m,a in out['methods'].items()},flush=True)
    return out


def score(root):
    plan=json.loads((root/'plan.json').read_text());sealed=json.loads((root/'generation_complete.json').read_text())
    results=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text()) for i in range(plan['count'])]
    assert sealed['hashes']==[fingerprint(r) for r in results]
    label_file=Path(plan['label_source']);alllabels={r['id']:r for r in json.loads(label_file.read_text())};labels=[alllabels[r['id']] for r in results]
    summary={'methods':{},'physical':audit(root),'scope':plan['scope'],'fallback':'same prior or preserved state in all methods; invalid model outputs separately counted',
             'no_test_read':True,'input_fingerprint':plan['inputs_fingerprint']}
    for method in METHODS:
        arms=[r['methods'][method] for r in results];ranks=[[int(m) for m in a['ranking']] for a in arms]
        s=evaluate(labels,ranks);s.update(all_calls_valid_contexts=sum(a['all_calls_valid'] for a in arms),
            logical_calls=sum(len(a['calls']) for a in arms),failed_calls=sum(not c.get('valid',True) for a in arms for c in a['calls']))
        if method.endswith('challenge'):
            events=[e for a in arms for e in a['events']];s.update(reviews=len(events),accepted=sum(e['valid'] for e in events),
                changed=sum(e.get('changed',False) for e in events),new_to_union_selected=sum(a['new_to_union_selected'] for a in arms))
        summary['methods'][method]=s
    create(root/'summary.json',summary);print(json.dumps(summary,indent=2));return summary


def run(input_path,label_path,root,count,reuse=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    raw=json.loads(Path(input_path).read_text())[:count]
    if (root/'inputs.json').exists():rows=json.loads((root/'inputs.json').read_text())
    else:
        evidence=NativeEvidence();rows=[evidence.augment(row,history_limit=3,kg_limit=3) for row in raw]
        create(root/'inputs.json',rows)
    assert [r['id'] for r in rows]==[r['id'] for r in raw]
    paths=['native_evidence_optimization_focused.py','native_candidate_challenge_focused.py','native_optimization_runtime.py','native_entity_evidence.py','native_candidate_challenge.py','review_allocation_game.py','transport_d2.py']
    plan={'version':'native-open-evidence-focused-v2','reuse_root':str(Path(reuse).resolve()) if reuse else None,'count':count,'ids':[r['id'] for r in rows],
      'source_sha256':{n:sha(ROOT/'experiments/paired_deliberation'/n) for n in paths},'inputs_fingerprint':fingerprint(rows),
      'input_source':str(Path(input_path).resolve()),'input_source_sha256':sha(input_path),'label_source':str(Path(label_path).resolve()),
      'scope':'development optimization; same input hash selection, not confirmatory; retriever config developed on valid',
      'methods':list(METHODS),'max_logical_calls_main_arm':6,'workers':4,'dispatch_interval_seconds':3.0,'shared_provider_pacing':True,
      'engineering_release':'first case at least10 of12 ranking calls valid before remaining inputs',
      'physical_cap':1200,'configured_max_cny':4,'same_request_reuse':'exact successful request hash; known429 may retry once in this revised batch, original errors retained; unknown parent intents never resent; logical costs not amortized',
      'same_information':'full union for single and additional independent; initial agents split KG vs disjoint history subsets; all reviews access same sources',
      'fallback':'invalid ranking keeps same prior/own state; invalid candidate update keeps common state; no gold-dependent fallback',
      'game_value':'train-only evidence/retrieval/disagreement proxy; not actual hit gain',
      'generation_labels_available':False,'no_automatic_timeout_retry':True}
    create(root/'plan.json',plan)
    for n in paths:
        dst=root/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and sha(dst)!=plan['source_sha256'][n]:raise ValueError('snapshot changed')
        dst.write_bytes((ROOT/'experiments/paired_deliberation'/n).read_bytes())
    api=Runtime(root,plan);out={0:case(0,rows[0],api,root)}
    rank_calls=out[0]['initial']+out[0]['methods']['single6']['calls']+out[0]['methods']['independent6']['calls'][3:]
    valid=sum(c['valid'] for c in rank_calls)
    released=valid>=10
    create(root/'engineering_release.json',{'rank_calls':len(rank_calls),'valid':valid,'pass':released,'no_labels_read':True})
    if not released:
        print('engineering stop before remaining cases; no labels read',flush=True);return None
    with ThreadPoolExecutor(max_workers=4) as workers:
        jobs={workers.submit(case,i,row,api,root):i for i,row in enumerate(rows) if i>0}
        for future in as_completed(jobs):out[jobs[future]]=future.result()
    create(root/'generation_complete.json',{'hashes':[fingerprint(out[i]) for i in range(count)],'labels_available':False})
    return score(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--labels',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=8);p.add_argument('--reuse')
    args=p.parse_args();run(args.inputs,args.labels,Path(args.output),args.count,args.reuse)
