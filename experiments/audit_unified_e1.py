"""Independent sealed E1 integrity, budget and local-update audit; no API calls."""
import argparse,json
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.evidence_consensus.state import check_integrity


def audit(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());seal=json.loads((root/'generation_complete.json').read_text())
    inputs=json.loads((root/'inputs.json').read_text())
    cases=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text())for i in range(plan['count'])]
    assert [c['task_id']for c in cases]==plan['task_ids']==[t['id']for t in inputs]
    assert seal['hashes']==[fingerprint(c)for c in cases]
    assert seal.get('task_ids')==plan['task_ids']
    assert fingerprint(inputs)==plan['bounded_tasks_hash']
    for name,h in plan['source_sha256'].items():assert sha(root/'source_snapshot'/name)==h,name
    counts={'tasks':len(cases),'arms':0,'logical_calls':0,'valid_model_calls':0,'invalid_model_calls':0,'accepted_patches':0,'rejected_or_noop_patches':0,'new_fetches':0,'reuses':0,'shared_logical_calls':0}
    per_method={}
    for c in cases:
        assert c['labels_available'] is False
        assert set(c['methods'])==set(plan['methods'])
        for method,arm in c['methods'].items():
            counts['arms']+=1;check_integrity(arm['state']);events=arm['events'];budget=arm['budget']
            assert arm['semantic_truth_certified'] is False
            assert arm['logical_calls']==len(events)==budget['used_or_reserved']['calls']
            for key,limit in budget['limits'].items():assert budget['used_or_reserved'][key]<=limit+1e-9,(c['task_id'],method,key)
            assert len(events)<=plan['max_calls_per_task_arm']
            counts['logical_calls']+=len(events)
            version=0;accepted=0
            for event in events:
                key='valid_model_calls'if event['valid']else 'invalid_model_calls';counts[key]+=1
                physical='request_'+event['request_hash'][:32]
                raw=root/'slots'/(physical+'.raw.json');intent=root/'slots'/(physical+'.intent.json')
                parent=Path(plan['reuse_root'])/'slots' if plan.get('reuse_root') else None
                parent_unknown=parent is not None and (parent/(physical+'.intent.json')).exists() and not (parent/(physical+'.raw.json')).exists()
                assert raw.exists()or intent.exists()or parent_unknown,physical
                if raw.exists():
                    receipt=json.loads(raw.read_text());assert receipt['integrity']==fingerprint(receipt['raw'])
                    assert event['request_hash']==fingerprint(receipt['request'])
                if event.get('source_action'):
                    assert event['expected_state_version']==version
                    action=event['source_action'];new=action['new_sources'];counts['new_fetches']+=new;counts['reuses']+=new==0
                    assert set(action['before_visible_ids'])<=set(action['after_visible_ids'])
                if event.get('patch_accepted'):
                    accepted+=1;counts['accepted_patches']+=1;version+=1
                    patch=event['report']['C']['proposed_patch'];before=event['before_decision'];after=event['after_decision']
                    assert before!=after,'accepted E1 patch must change the public decision'
                    if c['kind']=='rec':
                        assert patch['operation']=='swap_candidate'
                        expected=list(before);i,j=expected.index(patch['a']),expected.index(patch['b']);expected[i],expected[j]=expected[j],expected[i]
                        assert after==expected,'non-target recommendation position changed'
                    else:assert patch['operation']=='set_answer'and after==patch['value']
                if event.get('patch_rejected'):counts['rejected_or_noop_patches']+=1
            assert arm['state']['version']==accepted
            assert arm['decision']==arm['state']['decision']
            row=per_method.setdefault(method,{'logical_calls':0,'valid_model_calls':0,'accepted_patches':0})
            row['logical_calls']+=len(events);row['valid_model_calls']+=sum(e['valid']for e in events);row['accepted_patches']+=accepted
    unknown=[];statuses={};known_cny=0.;receipts=list((root/'slots').glob('*.raw.json'))
    for p in receipts:
        raw=json.loads(p.read_text());assert raw['integrity']==fingerprint(raw['raw']);status=str(raw['raw']['status_code']);statuses[status]=statuses.get(status,0)+1
    for p in (root/'slots').glob('*.intent.json'):
        if not (root/'slots'/(p.name.removesuffix('.intent.json')+'.raw.json')).exists():unknown.append(p.name)
    result={'scope':'sealed engineering integration audit, not semantic correctness or confirmatory effect proof','plan_sha256':sha(root/'plan.json'),'seal_sha256':sha(root/'generation_complete.json'),'counts':counts,'methods':per_method,'physical_receipts':len(receipts),'http_statuses':statuses,'unknown_requests_do_not_resend':sorted(unknown),'checks':['input and case identity','original generation seal','frozen source snapshots','per-arm global budget','raw request/response integrity','sequential state versions','local decision updates','source visibility monotonicity','non-target ranking preservation']}
    create(root/'independent_engineering_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.output)
