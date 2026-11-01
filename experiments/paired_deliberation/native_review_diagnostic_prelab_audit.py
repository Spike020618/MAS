"""Independent label-free audit of frozen policy and real request provenance.

Safe during live collection: reads only fixed packets, policy, code, logical
requests and per-provider plan. Never opens response labels or future dialogue.
"""
import argparse,collections,json,datetime
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_reviewer_calibration import request

DEFAULT=ROOT/'experiments/results/native_review_diagnostic_live_v1'


def read(path):return json.loads(Path(path).read_text())


def run(root=DEFAULT):
    root=Path(root);plan=read(root/'plan.json');packets=Path(plan['packets'])
    inputs=read(packets/'pair64.inputs.json');states=read(packets/'pair_states.json');policy=read(root/'policy_commitment.json')
    rows={r['task_id']:r for r in inputs};issues=[]
    checks={'inputs_hash':sha(packets/'pair64.inputs.json')==plan['inputs_sha256'],'states_hash':sha(packets/'pair_states.json')==plan['states_sha256'],'frozen_rates_hash':sha(packets/'frozen_fit_rates.json')==plan['frozen_rates_sha256'],'policy_hash':fingerprint(policy)==plan['policies_hash'],'tasks_match':list(rows)==plan['tasks'],'distinct_conversations':len({r['case_id'].rsplit(':',1)[0]for r in inputs})==64,'states_match_targets':all(fingerprint(states[r['task_id']])==r['target']['base_ranking_hash']for r in inputs),'safe_valid_empty':read(packets/'prefix-only-serving-data/valid_data.json')==[],'snapshot_hashes':all(sha(root/'source_snapshot'/name)==value for name,value in plan['source_sha256'].items()),'current_request_code_matches_snapshot':sha(ROOT/'experiments/paired_deliberation/native_reviewer_calibration.py')==plan['source_sha256']['native_reviewer_calibration.py']}
    if not checks['current_request_code_matches_snapshot']:raise ValueError('cannot audit expected request with changed constructor')
    expected={fingerprint(request(r)):r for r in inputs}
    provider_receipts={};seen={};total=0
    for model in plan['models']:
        logical=sorted((root/model/'logical_calls').glob('*.json'));bad=[];hashes=[]
        for file in logical:
            value=read(file);req=value['request'];h=fingerprint(req);hashes.append(h);total+=1
            if value['request_hash']!=h or h not in expected:bad.append(file.name);continue
            row=expected[h];body=json.loads(req['messages'][1]['content'])
            if not all(sid in row['sources'] and card==row['sources'][sid]for sid,card in body['sources'].items()):bad.append(file.name)
            if body['dialogue']!=row['text']:bad.append(file.name)
            if value['physical_slot']!='request_'+h[:32]:bad.append(file.name)
        physical=list((root/model/'slots').glob('*.raw.json'));intents=list((root/model/'slots').glob('*.intent.json'))
        provider_receipts[model]={'logical_requests_checked':len(logical),'distinct_request_hashes':len(set(hashes)),'invalid_request_files':sorted(set(bad)),'raw_responses_observed':len(physical),'intent_files_observed':len(intents),'provider_cap':plan['physical_cap_per_provider'],'at_most_64_logical_requests':len(logical)<=64}
        if bad:issues.append(model+': request mismatch')
        if len(logical)>64:issues.append(model+': logical cap exceeded')
        seen[model]=set(hashes)
    caps=[];action_counts=collections.Counter();selected=collections.Counter()
    for group in policy['groups']:
        group_ids=group['task_ids'];valid=len(group_ids)==4 and len(set(group_ids))==4
        groupcheck={'group':group['group'],'four_distinct_tasks':valid,'policies':{}}
        for name,actions in group['policies'].items():
            within=len(actions)<=policy['max_calls_per_group'] and all(a['task_id']in group_ids and group_ids[a['query']]==a['task_id']for a in actions)
            unique=len({a['provider']for a in actions})==len(actions) if name in ('game','central')else True
            groupcheck['policies'][name]={'calls':len(actions),'within_group_total_cap':within,'game_each_player_at_most_one':unique if name in ('game','central')else 'not_applicable_single_model_control_can_make_three_calls'}
            action_counts[name]+=len(actions)
            for a in actions:selected[name+':'+a['provider']]+=1
            if not within or not unique:issues.append('group '+str(group['group'])+' '+name+' cap')
        caps.append(groupcheck)
    intersections={left+'__'+right:len(seen[left]&seen[right])for i,left in enumerate(plan['models'])for right in plan['models'][i+1:]}
    report={'audit_version':'prelabel-request-provenance-v1','recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'label-free checks of committed policy and requests; no diagnostic future dialogue or evaluation labels read','checks':checks,'live_progress_snapshot_not_final':provider_receipts,'common_identical_request_hashes_across_providers':intersections,'all_checked_request_sources_exact_subsets_of_prefix_packets':not any(p['invalid_request_files']for p in provider_receipts.values()),'policy_calls':dict(action_counts),'policy_provider_assignments':dict(selected),'group_caps':caps,'issues':issues+[k for k,v in checks.items()if not v],'limitations':['Fit strata were balanced 24/24/16 whereas diagnostic pair selection is label-blind and natural in winner mix; rates are hypotheses, not deployment-certified action probabilities.','Fit packets had original per-role movie annotation mapping; diagnostic packets reconstruct role movie links only from fold80 known refs and provide aggregate current context IDs without role attribution. Changes in rates cannot be attributed solely to data generalization.','Three-provider192 potential outcomes are paid for evaluation. Policies are frozen/replayed counterfactual allocations; this is not adaptive online negotiation.','Game/central use each reviewer at most once per four-query group. Fixed single-provider controls may use that same model three times, sharing total API cap but not an identical provider-composition constraint.','Cost-aware main policy uses count gain per CNY weight1, likely excludes paid models for tiny gain estimates; zero participation or free-GLM allocation is a real policy result, not proof that paid reviewers have no information.'],'plan_sha256':sha(root/'plan.json'),'packets_plan_sha256':sha(packets/'plan.json'),'policy_sha256':sha(root/'policy_commitment.json'),'audit_source_sha256':sha(Path(__file__))}
    create(root/'prelabel_independent_audit.json',report)
    if report['issues']:raise ValueError(str(report['issues']))
    print(json.dumps({k:report[k]for k in ('checks','live_progress_snapshot_not_final','policy_calls','policy_provider_assignments','issues')},indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args();run(a.output)
