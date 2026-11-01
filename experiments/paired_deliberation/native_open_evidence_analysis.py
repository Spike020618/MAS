"""Read-only sealed-generation audit and paired analysis of open-evidence runs.

No provider construction or API calls. Labels are read only AFTER the root's
complete marker, input IDs, case hashes and source snapshots are verified.
"""
import argparse,collections,importlib.util,json,time
from pathlib import Path
import numpy as np
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.benchmark_evaluation import native_item_metrics
from experiments.paired_deliberation.client_v5 import parse_raw

DEFAULTS=('native_open_evidence_screen_v2','native_open_evidence_focused_screen_v1',
          'native_open_evidence_transfer_v1','native_open_evidence_focused_transfer_v1')


def paired(labels,left,right,replicates=10000,seed=281028):
    groups=collections.defaultdict(lambda:np.zeros(3));gained=lost=0
    for row,a,b in zip(labels,left,right):
        a=list(map(int,a));b=list(map(int,b));group=groups[row['id'].rsplit(':',1)[0]]
        for target in row['targets']:
            ma=native_item_metrics(a,target);mb=native_item_metrics(b,target)
            group[0]+=ma['recall@10']-mb['recall@10'];group[1]+=ma['ndcg@10']-mb['ndcg@10'];group[2]+=1
            gained+=target in a[:10] and target not in b[:10];lost+=target not in a[:10] and target in b[:10]
    values=np.array(list(groups.values()));n=len(values)
    if not n:return {'targets':0,'gained_hits@10':0,'lost_hits@10':0}
    draw=np.random.default_rng(seed).integers(0,n,size=(replicates,n));totals=values[draw].sum(axis=1)
    point=values[:,:2].sum(axis=0)/values[:,2].sum();interval=np.quantile(totals[:,:2]/totals[:,2,None],[.025,.975],axis=0)
    return {'contexts':len(labels),'dialogue_clusters':n,'targets':int(values[:,2].sum()),'gained_hits@10':int(gained),'lost_hits@10':int(lost),
            'recall@10_difference':float(point[0]),'recall@10_cluster95_interval':interval[:,0].tolist(),
            'ndcg@10_difference':float(point[1]),'ndcg@10_cluster95_interval':interval[:,1].tolist(),'replicates':replicates,'seed':seed,
            'interpretation':'exploratory dialogue-cluster percentile interval; no multiplicity adjustment or confirmatory claim'}


def cost_usage(raw):
    try:body=json.loads(raw['body'])
    except (ValueError,KeyError,TypeError):return {},False
    u=body.get('usage',{});known=all(type(u.get(k)) is int for k in ('prompt_tokens','completion_tokens'))
    return ({k:u.get(k,0) for k in ('prompt_tokens','completion_tokens')},known)


def price(usage):return (usage['prompt_tokens']*.15+usage['completion_tokens']*1.5)/1e6


def physical(root):
    usage=collections.Counter(prompt_tokens=0,completion_tokens=0);status=collections.Counter();reused=[];unknown_usage=[]
    responses=0
    for p in (root/'slots').glob('*.raw.json'):
        wrapped=json.loads(p.read_text())
        assert fingerprint(wrapped['raw'])==wrapped['integrity'],'raw receipt integrity'
        if wrapped.get('reused_from'):reused.append({'path':str(p),'reused_from':wrapped['reused_from']});continue
        responses+=1;raw=wrapped['raw'];status[str(raw['status_code'])]+=1;u,known=cost_usage(raw)
        if known:usage.update(u)
        else:unknown_usage.append({'path':str(p),'status':raw['status_code']})
    unknown=[str(p) for p in (root/'slots').glob('*.intent.json')]
    return {'new_returned_responses':responses,'new_unknown_intents':unknown,'new_physical_attempts':responses+len(unknown),
            'reused_responses':len(reused),'reused_provenance':reused,'http_status':dict(status),'known_usage_including_invalid':dict(usage),
            'configured_known_cny':price(usage),'responses_without_known_usage':unknown_usage,
            'known429_no_usage_are_known_failed_responses_not_unknown_completion':True,'actual_invoice_verified':False}


def resolve_receipt(root,physical_slot,request):
    seen=set();current=root
    while current is not None:
        if str(current) in seen:raise ValueError('reuse parent cycle')
        seen.add(str(current));p=current/'slots'/(physical_slot+'.raw.json')
        if p.exists():
            wrapped=json.loads(p.read_text())
            if wrapped['request']!=request:return {'status':'different_parent_request','physical_slot':physical_slot}
            if fingerprint(wrapped['raw'])!=wrapped['integrity']:raise ValueError('receipt tampered')
            return {'status':'returned','path':str(p),'wrapper':wrapped}
        intent=current/'slots'/(physical_slot+'.intent.json')
        if intent.exists():
            entry=json.loads(intent.read_text())
            if entry['request']==request:return {'status':'unknown_completion','path':str(intent),'physical_slot':physical_slot}
        plan=current/'plan.json';parent=json.loads(plan.read_text()).get('reuse_root') if plan.exists() else None
        current=Path(parent) if parent else None
    return {'status':'no_dispatch_receipt','physical_slot':physical_slot}


def logical_budgets(root,cases,methods):
    result={}
    for method in methods:
        usage=collections.Counter(prompt_tokens=0,completion_tokens=0);statuses=collections.Counter();unknown=[];slots=[];failed=0
        for case in cases:
            for call in case['methods'][method]['calls']:
                slot=call['slot'];failed+=not call.get('valid',True);entry=json.loads((root/'logical_calls'/(slot+'.json')).read_text())
                assert fingerprint(entry['request'])==entry['request_hash']
                assert entry['physical_slot']=='request_'+fingerprint(entry['request'])[:32]
                resolved=resolve_receipt(root,entry['physical_slot'],entry['request']);slots.append(entry['physical_slot'])
                if resolved['status']=='returned':
                    raw=resolved['wrapper']['raw'];statuses[str(raw['status_code'])]+=1;u,known=cost_usage(raw)
                    if known:usage.update(u)
                    else:unknown.append({'logical_slot':slot,'kind':'response_usage_unavailable','http_status':raw['status_code']})
                else:unknown.append({'logical_slot':slot,**resolved})
        result[method]={'logical_calls':len(slots),'distinct_physical_requests_used':len(set(slots)),'failed_logical_calls':int(failed),
            'response_status':dict(statuses),'known_logical_usage_including_invalid':dict(usage),'configured_known_logical_cny':price(usage),
            'unknown_usage_or_dispatch':unknown,'scope':'shared/reused requests counted per method; these logical costs must not be summed as provider invoice'}
    return result


def load_sealed(root):
    marker=root/'generation_complete.json'
    if not marker.exists():raise RuntimeError('generation incomplete; labels forbidden')
    plan=json.loads((root/'plan.json').read_text());inputs=json.loads((root/'inputs.json').read_text());seal=json.loads(marker.read_text())
    cases=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text()) for i in range(plan['count'])]
    if seal['hashes']!=[fingerprint(r) for r in cases]:raise ValueError('generation seal mismatch')
    if [r['id'] for r in cases]!=plan['ids'] or [r['id'] for r in inputs]!=plan['ids']:raise ValueError('ID order mismatch')
    if fingerprint(inputs)!=plan['inputs_fingerprint']:raise ValueError('input fingerprint mismatch')
    for name,digest in plan['source_sha256'].items():
        if sha(root/'source_snapshot'/name)!=digest:raise ValueError('frozen code mismatch '+name)
    if any(r.get('generation_labels_available') is not False for r in cases):raise ValueError('generation label isolation declaration missing')
    return plan,inputs,cases


def generation_audit(root,plan,inputs,cases):
    spec=importlib.util.spec_from_file_location('frozen_challenge_'+fingerprint(str(root))[:12],root/'source_snapshot/native_candidate_challenge.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    result={'valid_events_replayed':0,'invalid_events':0,'offered_source_identity_checks':0,'update_locality_checks':0,
            'focused_pair_requests':0,'fullpool_requests':0,'rank_requests':0,'generation_label_flag_checks':0,
            'fallback_checks':0,'semantic_entailment_verified':False,'source_identity_is_not_entailment':True,
            'initial_material_partition':'initial0 KG plus stats; initial1/2 disjoint training neighbors plus stats; single and added independent use union'}
    for row,case in zip(inputs,cases):
        pool=[m['movie_id'] for m in row['candidates']]
        base=case['methods']['independent3']['ranking']
        assert base==frozen.consensus_ranking([c['ranking'] for c in case['initial']],pool)
        for method,arm in case['methods'].items():
            assert len(arm['ranking'])==len(pool) and set(arm['ranking'])==set(pool)
            for call in arm['calls']:
                logical=json.loads((root/'logical_calls'/(call['slot']+'.json')).read_text());req=logical['request']
                assert req['labels_available'] is False;result['generation_label_flag_checks']+=1
            if method not in ('central_challenge','game_challenge'):continue
            mode=method.split('_')[0];commit=json.loads((root/'commitments'/('case'+str(cases.index(case))+'_'+mode+'.json')).read_text())
            assert fingerprint(base)==commit['initial_hash']
            expected=frozen.allocate(commit['targets'],commit['costs'],mode=mode)
            assert fingerprint(expected)==fingerprint(commit['allocation'])
            state=list(base)
            assert len(arm['events'])==len(expected['allocation'])
            for event,(reviewer,q) in zip(arm['events'],expected['allocation']):
                logical=json.loads((root/'logical_calls'/(event['slot']+'.json')).read_text());req=logical['request'];payload=json.loads(req['messages'][1]['content'])
                target=commit['targets'][q];assert target==event['target'];offered=payload['sources']
                assert payload['pair']==[target['a'],target['b']]
                size=len(payload['candidate_pool']);result['focused_pair_requests']+=size==2;result['fullpool_requests']+=size==len(pool)
                for sid,source in offered.items():
                    assert sid in row['sources'] and source==row['sources'][sid]
                    if source.get('allowed_use')=='current_expression':assert source['origin_case_id']==row['id'] and source.get('role')=='seeker'
                receipt=resolve_receipt(root,logical['physical_slot'],req)
                if event['valid']:
                    assert receipt['status']=='returned';record=parse_raw(receipt['wrapper']['raw'],req)
                    replay=frozen.apply_response(state,target,record['report'],offered,row['id'],expected_hash=fingerprint(state))
                    assert replay['valid']
                    for field in ('ranking','changed','bound_sources','entailment_verified','current_preference_certified'):
                        assert replay[field]==event[field]
                    assert all(s['source_id'] in offered for s in replay['bound_sources']);result['offered_source_identity_checks']+=1
                    result['valid_events_replayed']+=1
                else:
                    assert event['ranking']==state;result['invalid_events']+=1;result['fallback_checks']+=1
                before=list(state);state=list(event['ranking']);allowed={before.index(target['a']),before.index(target['b'])}
                assert all(state[i]==before[i] for i in range(len(state)) if i not in allowed);result['update_locality_checks']+=1
            assert state==arm['ranking']
    # Deduplicated request audit, separate from logical per-method duplication.
    for file in (root/'logical_calls').glob('*.json'):
        request=json.loads(file.read_text())['request']
        if request['kind']=='ranking_v6':result['rank_requests']+=1
    result['focused_mode']=result['focused_pair_requests']>0 and result['fullpool_requests']==0
    result['information_comparison_limit']='focused version removes nonpair pool and full initial rankings from each review; compare as revised readout, not unchanged prompt effect'
    return result


def compare_parent(root,plan,inputs,cases):
    parent=Path(plan['reuse_root']) if plan.get('reuse_root') else None
    if parent is None:return {'has_parent':False}
    pp,pi,pc=load_sealed(parent)
    if [r['id'] for r in pc]!=[r['id'] for r in cases]:return {'has_parent':True,'same_case_ids':False}
    controls=('retrieval','single1','single6','independent3','independent6')
    differences={m:sum(a['methods'][m]['ranking']!=b['methods'][m]['ranking'] for a,b in zip(cases,pc)) for m in controls}
    initial_changes=sum([x['ranking'] for x in a['initial']]!=[x['ranking'] for x in b['initial']] for a,b in zip(cases,pc))
    known429_repairs=[];new_parent_unknown=[]
    for path in (root/'slots').glob('*.raw.json'):
        wrapped=json.loads(path.read_text())
        if wrapped.get('reused_from'):continue
        old=parent/'slots'/path.name
        if old.exists():
            prev=json.loads(old.read_text())
            if prev['request']==wrapped['request'] and prev['raw']['status_code']==429:
                known429_repairs.append({'slot':path.name,'new_http_status':wrapped['raw']['status_code']})
        if (parent/'slots'/path.name.replace('.raw.json','.intent.json')).exists():new_parent_unknown.append(path.name)
    return {'has_parent':True,'parent':str(parent),'same_case_ids':True,'same_input_fingerprint':fingerprint(pi)==fingerprint(inputs),
        'control_ranking_changed_contexts':differences,'initial_ensemble_changed_contexts':initial_changes,
        'same_request_parent429_reexecutions':known429_repairs,'unknown_parent_intent_reexecution_candidates':new_parent_unknown,
        'interpretation':'focused removes fullpool/other rankings; known429 repair can additionally alter upstream controls, so cross-version difference is not a pure pair-prompt ablation'}


def lineage_cost(root):
    seen=set();items=[]
    while root is not None:
        name=str(root.resolve())
        if name in seen:raise ValueError('cost lineage cycle')
        seen.add(name);a=physical(root)
        items.append({'root':name,'known_cny':a['configured_known_cny'],'new_attempts':a['new_physical_attempts'],
                      'new_429':a['http_status'].get('429',0),'new_unknown_intents':a['new_unknown_intents'],'new_replayed_responses':a['reused_responses']})
        plan=json.loads((root/'plan.json').read_text());parent=plan.get('reuse_root');root=Path(parent) if parent else None
    return {'roots':items,'known_cny_sum':sum(a['known_cny'] for a in items),
        'unknown_completion_count':sum(len(a['new_unknown_intents']) for a in items),
        'known429_response_count':sum(a['new_429'] for a in items),'cost_is_configured_price_not_invoice':True}


def analyze(root):
    root=Path(root);plan,inputs,cases=load_sealed(root)
    # Audit public generations and actual offered source maps BEFORE loading labels.
    generation=generation_audit(root,plan,inputs,cases)
    all_labels={r['id']:r for r in json.loads(Path(plan['label_source']).read_text())};labels=[all_labels[r['id']] for r in cases]
    methods=plan['methods'];ranks={m:[r['methods'][m]['ranking'] for r in cases] for m in methods}
    comparisons={}
    for a,b in [('game_challenge','independent3'),('game_challenge','independent6'),('game_challenge','single6'),('game_challenge','central_challenge'),
                ('central_challenge','independent3'),('single6','single1'),('independent3','retrieval')]:
        comparisons[a+'_minus_'+b]=paired(labels,ranks[a],ranks[b])
    metrics={}
    for method in methods:
        totals=collections.Counter();n=0
        for label,ranking in zip(labels,ranks[method]):
            for target in label['targets']:totals.update(native_item_metrics(list(map(int,ranking)),target));n+=1
        metrics[method]={'contexts':len(cases),'targets':n,'hits@10':int(round(totals['recall@10'])),'metrics':{k:v/n for k,v in totals.items()},
                         'all_calls_valid_contexts':sum(c['methods'][method]['all_calls_valid'] for c in cases)}
    omissions={}
    for method in ('central_challenge','game_challenge'):
        selected=promoted=promoted_target_events=0;recovered=lost=0;sourceuses=collections.Counter();valid=changed=0;case_ids=[]
        for case,label in zip(cases,labels):
            initial_union={m for call in case['initial'] for m in call['ranking'][:10]};base=case['methods']['independent3']['ranking']
            for event in case['methods'][method]['events']:
                novel=event['target']['novel_to_initial_union'];selected+=novel
                valid+=event['valid'];changed+=event['changed']
                if event['valid']:
                    for source in event['bound_sources']:sourceuses[source.get('allowed_use',source.get('kind','unknown'))]+=1
                if novel and event['valid'] and event['changed'] and event['target']['b'] in event['ranking'][:10]:
                    promoted+=1;promoted_target_events+=sum(str(t)==event['target']['b'] for t in label['targets']);case_ids.append(case['id'])
            final=case['methods'][method]['ranking']
            for target in label['targets']:
                t=str(target);recovered+=t not in initial_union and t in final[:10];lost+=t in base[:10] and t not in final[:10]
        omissions[method]={'reviews':sum(len(c['methods'][method]['events']) for c in cases),'valid_reviews':int(valid),'changed_reviews':int(changed),
            'novel_to_initial_union_selected':int(selected),'novel_to_initial_union_actually_promoted':int(promoted),
            'promoted_true_target_events':int(promoted_target_events),'final_hits_on_initial_shared_omissions':int(recovered),
            'initial3_hits_lost':int(lost),'promotion_case_ids':case_ids,'bound_source_uses':dict(sourceuses),
            'promotion_is_not_necessarily_recommendation_gain':True}
    result={'analysis_source_sha256':sha(Path(__file__)),'generation_sealed_before_labels':True,'root':str(root.resolve()),
        'scope':plan['scope'],'contexts':len(cases),'item_targets':sum(len(r['targets']) for r in labels),'metrics':metrics,
        'comparisons':comparisons,'shared_omission_process':omissions,'generation_audit':generation,
        'physical_cost':physical(root),'reuse_lineage_cost':lineage_cost(root),'cross_version_parent':compare_parent(root,plan,inputs,cases),'logical_budgets':logical_budgets(root,cases,methods),
        'limitations':['same six-call caps do not equal identical tokens or information per individual call',
            'retrieval/prior-state fallback is included for every method; differs from earlier strict-zero protocol',
            'focused reviews see only pair candidate pool and pair initial positions; original mode sees full candidate pool and initial rankings',
            'same-model exact request reuse reduces repeated billing but is shared realization, not independent draw',
            'train statistics are prediction analogies, not current-user facts; source identity does not certify semantic entailment',
            'development optimization, no positive-result-only reporting or confirmatory significance claim']}
    snapshot=root/'source_snapshot/native_open_evidence_analysis.executed.py'
    if snapshot.exists() and sha(snapshot)!=sha(Path(__file__)):raise ValueError('analysis snapshot changed')
    snapshot.write_bytes(Path(__file__).read_bytes())
    create(root/'analysis.json',result)
    return result


def brief(result):
    return {'root':result['root'],'targets':result['item_targets'],'hits':{m:v['hits@10'] for m,v in result['metrics'].items()},
        'novel_promotions':{m:{k:v for k,v in row.items() if k in ('novel_to_initial_union_selected','novel_to_initial_union_actually_promoted','final_hits_on_initial_shared_omissions')} for m,row in result['shared_omission_process'].items()},
        'new_known_cny':result['physical_cost']['configured_known_cny']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',action='append');p.add_argument('--watch',action='store_true');a=p.parse_args()
    roots=[Path(r) for r in a.root] if a.root else [ROOT/'experiments/results'/name for name in DEFAULTS]
    remaining=list(roots)
    while remaining:
        for root in list(remaining):
            if not (root/'generation_complete.json').exists():continue
            if (root/'analysis.json').exists():result=json.loads((root/'analysis.json').read_text())
            else:result=analyze(root)
            print(json.dumps(brief(result),ensure_ascii=False),flush=True);remaining.remove(root)
        if not a.watch:break
        if remaining:time.sleep(10)

if __name__=='__main__':main()
