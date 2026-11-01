"""Bind each E1 version to its own snapshot; compare initial receipts and Rec execution.

No evaluation labels or model/network calls. Run after both generation seals.
"""
import argparse,json,collections
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha


def load(path):return json.loads(Path(path).read_text())


def sealed(root):
    plan=load(root/'plan.json');seal=load(root/'generation_complete.json');inputs=load(root/'inputs.json')
    if len(inputs)!=plan['count']or fingerprint(inputs)!=plan['bounded_tasks_hash']:raise ValueError('input seal mismatch')
    cases=[load(root/'cases'/('case_'+str(i)+'.json'))for i in range(len(inputs))]
    if seal['hashes']!=[fingerprint(c)for c in cases]or seal['task_ids']!=[t['id']for t in inputs]:raise ValueError('case seal mismatch')
    for name,h in plan['source_sha256'].items():
        if sha(root/'source_snapshot'/name)!=h:raise ValueError('snapshot does not belong to version '+root.name+': '+name)
    return plan,inputs,cases


def summarize(cases):
    out={};unknown=[]
    for case in cases:
        kind=case['kind'];by_method=out.setdefault(kind,{})
        for method,arm in case['methods'].items():
            counts=by_method.setdefault(method,{'tasks':0,'valid_reviews':0,'accepted_patches':0,'accepted_swaps':0,'non_target_rank_violations':0,'version_or_source_rejections':0})
            counts['tasks']+=1
            for event in arm['events']:
                if 'source_action'not in event:continue
                counts['valid_reviews']+=bool(event['valid'])
                if event.get('patch_accepted'):
                    counts['accepted_patches']+=1;patch=event['report']['C']['proposed_patch']
                    if patch['operation']=='swap_candidate':
                        counts['accepted_swaps']+=1
                        before,after=event['before_decision'],event['after_decision'];a,b=patch['a'],patch['b']
                        changed=[i for i,(x,y)in enumerate(zip(before,after))if x!=y]
                        if len(before)!=len(after)or sorted(changed)!=sorted([before.index(a),before.index(b)]):counts['non_target_rank_violations']+=1
                reason=event.get('patch_reason','').lower()
                counts['version_or_source_rejections']+=bool(event.get('patch_rejected')and any(w in reason for w in ('version','source','authority','fetched','evidence')))
            for key in arm['budget']['unknown_requests']:
                receipt=arm['budget']['receipts'][key]
                unknown.append({'task_id':case['task_id'],'method':method,'key':key,'reserved_cny':receipt['reserved']['cny'],'charged_cny':receipt['charged']['cny']})
    return out,unknown


def initial_receipts(root,baseline):
    counts=collections.Counter();missing=[];changed=[]
    for path in sorted((root/'logical_calls').glob('*.json')):
        current=load(path)
        if not current['request']['kind'].endswith('initial'):continue
        counts['logical_initial_calls']+=1
        old_path=baseline/'logical_calls'/path.name
        if not old_path.exists():changed.append(path.name+':missing_prior_logical');continue
        prior=load(old_path)
        if current['request']!=prior['request']or current['request_hash']!=prior['request_hash']or current['physical_slot']!=prior['physical_slot']:
            changed.append(path.name+':changed_request');continue
        counts['exact_initial_requests']+=1
        slot=current['physical_slot'];raw=root/'slots'/(slot+'.raw.json');old_raw=baseline/'slots'/(slot+'.raw.json')
        if not raw.exists()or not old_raw.exists():missing.append({'logical_slot':path.name,'physical_slot':slot,'current_raw':raw.exists(),'prior_raw':old_raw.exists()});continue
        r,o=load(raw),load(old_raw)
        if r['integrity']!=fingerprint(r['raw'])or o['integrity']!=fingerprint(o['raw']):raise ValueError('raw integrity failed')
        if r['raw']!=o['raw']:changed.append(path.name+':changed_raw_body');continue
        counts['byte_equivalent_raw_objects']+=1
    return {'counts':dict(counts),'changed':changed,'missing_raw_receipts_preserved_without_retry':missing,'note':'logical slots can share the same physical request; count is not physical billing count'}


def audit(root,baseline):
    root,baseline=Path(root),Path(baseline);plan,inputs,cases=sealed(root);old_plan,old_inputs,old_cases=sealed(baseline)
    if inputs!=old_inputs:raise ValueError('version comparison uses different tasks')
    now,unknown=summarize(cases);old,old_unknown=summarize(old_cases);initial=initial_receipts(root,baseline)
    if initial['changed']:raise ValueError('initial generation changed between versions')
    if any(x['charged_cny']<=0 for x in unknown):raise ValueError('nonfree-provider unknown cost treated as zero')
    out={'scope':'no labels/API; each version bound to its own executed snapshot; initial generation same, review contract changed',
         'versions':{root.name:{'plan_sha256':sha(root/'plan.json'),'generation_seal_sha256':sha(root/'generation_complete.json'),'source_sha256':plan['source_sha256']},baseline.name:{'plan_sha256':sha(baseline/'plan.json'),'generation_seal_sha256':sha(baseline/'generation_complete.json'),'source_sha256':old_plan['source_sha256']}},
         'current':now,'baseline':old,'initial_generation_comparison':initial,'current_unknown_logical_receipts':unknown,'baseline_unknown_logical_receipts':old_unknown,
         'source_files_changed':[n for n in plan['source_sha256']if plan['source_sha256'][n]!=old_plan['source_sha256'].get(n)],
         'semantic_truth_certified':False,'warning':'accepted/executed swaps do not establish recommendation gain; no task labels used'}
    create(root/'independent_version_and_rec_execution_audit.json',out);print(json.dumps(out,ensure_ascii=False,indent=2));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/results/unified_e1_v4');p.add_argument('--baseline',type=Path,default=ROOT/'experiments/results/unified_e1_v3');a=p.parse_args();audit(a.output,a.baseline)
