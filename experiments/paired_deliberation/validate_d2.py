"""Execute exactly one frozen D2 purpose calibration, then score separately."""
import argparse
import copy
import json
import math
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT,create,fingerprint
from experiments.paired_deliberation.preflight_d2 import verify
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.state_d2 import StateD2
from experiments.paired_deliberation.verification_d2 import VerificationD2,utterance_request
from experiments.paired_deliberation.output_adapter_d2 import D2Client
from experiments.paired_deliberation.budget_d2 import JointBudget,ChannelClient
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.transport_d2 import TransportD2,parse_d2_raw
from experiments.paired_deliberation.validate_atomic_evidence import faithful
from experiments.paired_deliberation.atomic_evidence import parser_request
from experiments.paired_deliberation.atomic_relations_v4 import relation_request
from experiments.paired_deliberation.json_contract_d2 import JSONContractClient,json_request
from experiments.paired_deliberation.source_alias_d2 import SourceAliasClient


def infer(row,client,service_id):
    reg=EvidenceRegistry(row['evidence'],row['training_view']);claim=row['claim']
    try:reg.validate_claim(claim)
    except (ValueError,KeyError,TypeError) as exc:
        return {'authority_rejected':True,'error':str(exc),'hard_exclusion_active':False,'sentinel_preserved':True}
    last=reg.evidence['candidates_in_shared_retrieval_order'][-1];mid=last['movie_id']
    sentinel={'claim_id':'sentinel','text':'Movie '+mid+' has catalog year '+str(last['year'])+'.','claim_kind':'catalog_fact',
              'evidence_refs':[{'source_id':reg.resolve(last['source_id']),'use':'supports'}],'candidate_ids':[mid],
              'depends_on':[],'scope':None,'constraint_mode':'soft','stance':'support'}
    report={'ranking':[m['movie_id'] for m in reg.evidence['candidates_in_shared_retrieval_order'][:5]],'claims':[claim,sentinel],'updates':[]}
    state=StateD2({'current_request':report},reg);before=copy.deepcopy(state.claims['current_request:sentinel'])
    router=VerificationD2(reg,client,service_id=service_id);target='current_request:c1'
    state.observe('central',target,fingerprint(state.claims[target]),router,row['id'])
    return {'authority_rejected':False,'verification':state.claims[target]['verification'],
            'target_status':state.claims[target]['status'],'hard_exclusion_active':state.claims[target]['hard_constraint_active'],
            'sentinel_preserved':before==state.claims['current_request:sentinel']}


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,center-width),min(1,center+width)]


def provenance(root,plan,details):
    inputs={r['id']:r for r in json.loads((root/'inputs.json').read_text())};expected={};missing=[]
    for detail in details:
        row=inputs[detail['id']];claim=row['claim']
        if detail['expected_rejected'] or claim['scope'] is None:continue
        reg=EvidenceRegistry(row['evidence'],row['training_view'])
        legacy={'claim_id':claim['claim_id'],'text':claim['text'],'source_ids':[r['source_id'] for r in claim['evidence_refs']],
                'candidate_ids':claim['candidate_ids'],'claim_type':'explicit_preference','depends_on':claim['depends_on'],'stance':claim['stance']}
        expected[row['id']+'_parse']=parser_request(legacy,{})
        check=detail.get('verification',{});atom=check.get('atom')
        if isinstance(atom,dict) and atom.get('scope')!='unsupported' and not check.get('scope_disagreement'):
            if claim['claim_kind']=='utterance':expected[row['id']+'_witness']=utterance_request(atom,reg.dialogue())
            else:expected[row['id']+'_relations']=relation_request(atom,reg.dialogue())
    for slot,request in expected.items():
        request=json_request(request)
        path=root/'slots'/(slot+'.raw.json')
        if not path.exists():missing.append(slot);continue
        saved=json.loads(path.read_text())
        if (saved['request']!=request or saved['integrity']!=fingerprint(saved['raw']) or
            saved['dependency_hash']!=fingerprint({'manifest':plan,'request':request,'slot':slot})):
            raise ValueError('paid request/response provenance mismatch')
    extras=[p.name for p in (root/'slots').glob('*.raw.json') if p.name.removesuffix('.raw.json') not in expected]
    return {'missing':missing,'unexpected_slots':extras,'expected_receipts':len(expected)}


def score(root,write=True):
    root=Path(root);plan=verify(root);labels=json.loads((root/'labels.json').read_text());details=[]
    for label in labels:
        path=root/'predictions'/(label['id']+'.json')
        if not path.exists():raise RuntimeError('incomplete D2 calibration; no release score')
        saved=json.loads(path.read_text())
        if saved['integrity']!=fingerprint(saved['result']) or saved['plan_hash']!=plan['plan_hash']:raise ValueError('prediction altered')
        pred=saved['result'];check=pred.get('verification',{})
        valid=not pred['authority_rejected'] and not check.get('verification_failed',False)
        atom_ok=label['expected_scope'] is None or faithful(check.get('atom'),label['expected_scope'])
        verdict_ok=check.get('verdict')==label['expected_verdict']
        correct=(pred['authority_rejected'] if label['expected_rejected'] else
                 valid and atom_ok and verdict_ok and pred['hard_exclusion_active']==label['expected_hard']) and pred['sentinel_preserved']
        details.append({**label,**pred,'valid':valid,'atom_correct':atom_ok,'verdict_correct':verdict_ok,'joint_correct':correct})
    new=[r for r in details if r['group']=='validation' and r['purpose'] in {'utterance','current_preference'}]
    threshold=plan['thresholds'];gates={};summaries={}
    for purpose in sorted({r['purpose'] for r in details}):
        rows=[r for r in details if r['purpose']==purpose and r['group']=='validation'];k=sum(r['joint_correct'] for r in rows);n=len(rows)
        summaries[purpose]={'correct':k,'n':n,'joint_accuracy':k/n,'wilson95':wilson(k,n),
            'false_support':sum(r.get('verification',{}).get('verdict')=='supported' and r['expected_verdict'] not in {'supported','rejected'} for r in rows),
            'false_refutation':sum(r.get('verification',{}).get('verdict')=='refuted' and r['expected_verdict']!='refuted' for r in rows),
            'invalid':sum(not r['valid'] for r in rows if not r['expected_rejected'])}
    gates['new_semantic_joint_accuracy']=sum(r['joint_correct'] for r in new)/len(new)>=threshold['new_semantic_joint_accuracy_min']
    gates['each_soft_purpose']=all(summaries[p]['joint_accuracy']>=threshold['each_soft_purpose_joint_accuracy_min'] for p in ('utterance','current_preference'))
    gates['new_invalid']=sum(not r['valid'] for r in new)<=threshold['new_semantic_invalid_max']
    gates['catalog_exact']=summaries['catalog_fact']['correct']==summaries['catalog_fact']['n']
    gates['prediction_authority']=all(r['joint_correct'] for r in details if r['purpose'] in {'prediction_basis','authority'})
    gates['hard_guards']=all(r['joint_correct'] for r in details if r['purpose']=='hard_guard')
    gates['critical_conflict']=not any(r.get('verification',{}).get('verdict') in {'supported','refuted'} for r in new if r['critical_conflict'])
    regression=[r for r in details if r['group']=='regression'];gates['known_regressions']=all(r['joint_correct'] for r in regression)
    gates['sentinel']=all(r['sentinel_preserved'] for r in details)
    trace=provenance(root,plan,details)
    gates['receipt_provenance']=not trace['missing'] and not trace['unexpected_slots']
    usage_path=root/'usage.json'
    usage=json.loads(usage_path.read_text())['usage'] if usage_path.exists() else None
    gates['known_joint_usage']=bool(usage and usage['usage_complete'])
    raw=[json.loads(p.read_text()) for p in (root/'slots').glob('*.raw.json')]
    bodies=[json.loads(r['raw']['body']) for r in raw if r['raw']['status_code']==200]
    models=sorted({b.get('model') for b in bodies});pending=len(list((root/'slots').glob('*.intent.json')))
    gates['one_frozen_model_no_pending']=models==plan['verifier']['allowed_returned_models'] and pending==0 and all(r['raw']['status_code']==200 for r in raw)
    prompt=sum(b.get('usage',{}).get('prompt_tokens',0) for b in bodies);output=sum(b.get('usage',{}).get('completion_tokens',0) for b in bodies)
    profile=plan['verifier'];estimated=(prompt*profile['input_cny_per_million']+output*profile['output_cny_per_million'])/1_000_000
    reused=sum('reused_from' in r for r in raw);fresh_responses=len(raw)-reused
    cumulative=plan.get('prior_physical_attempts',0)+fresh_responses
    gates['physical_caps']=(len(raw)<=plan['maximum_physical_attempts'] and estimated<=plan['maximum_estimated_cny'] and
                            cumulative<=plan.get('cumulative_physical_cap',plan['maximum_physical_attempts']))
    result={'version':'d2_purpose_release_v1','plan_hash':plan['plan_hash'],'pass':all(gates.values()),'gates':gates,
            'purpose_summary':summaries,'new_semantic_correct':sum(r['joint_correct'] for r in new),'new_semantic_n':len(new),
            'regression_correct':sum(r['joint_correct'] for r in regression),'regression_n':len(regression),'details':details,'receipt_provenance':trace,
            'physical':{'responses':len(raw),'new_responses':fresh_responses,'reused_responses':reused,'cumulative_stage_attempts':cumulative,'pending_intents':pending,'models':models,'prompt_tokens':prompt,'completion_tokens':output,
                        'estimated_peak_list_price_cny':estimated,'actual_bill_unknown':True},
            'authorization_scope':'shared-initial engineering only' if all(gates.values()) else 'none',
            'pilot_allowed':False,'scope':plan['scope']}
    if write:create(root/'release_score.json',result)
    return result


def execute(root,offline_replay=False):
    root=Path(root);plan=verify(root)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False);load_dotenv(ROOT/'.env.local',override=False)
    provider=DurableProvider(root/'slots',plan,TransportD2('deepseek',plan['verifier'],root/'slots',
        plan['maximum_physical_attempts'],plan['maximum_estimated_cny'],12000,4096),parse_d2_raw,not offline_replay,
        reuse_initial=plan.get('reuse_root'),reuse_all=bool(plan.get('reuse_root')))
    ledger=JointBudget(plan['limits'],plan['profiles']);client=SourceAliasClient(JSONContractClient(D2Client(ChannelClient(ledger,'verifier',provider))))
    try:
        for row in json.loads((root/'inputs.json').read_text()):
            result=infer(row,client,plan['verifier']['service_id'])
            create(root/'predictions'/(row['id']+'.json'),{'result':result,'integrity':fingerprint(result),'plan_hash':plan['plan_hash']})
            print('collected '+row['id'],flush=True)
    except (RuntimeError,ValueError,KeyError,TypeError) as exc:
        create(root/'interruption.json',{'error':str(exc),'usage':ledger.summary(),'online_release':False})
        raise
    create(root/'usage.json',{'usage':ledger.summary(),'adaptations':client.adaptations})
    return score(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--execute-authorized',action='store_true');p.add_argument('--score',action='store_true');p.add_argument('--offline-replay',action='store_true')
    args=p.parse_args();result=execute(args.output,offline_replay=args.offline_replay) if args.execute_authorized or args.offline_replay else score(args.output) if args.score else verify(args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False,indent=2))
