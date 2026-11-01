"""One bounded, release-gated D2 shared-initial engineering batch; no task labels."""
import argparse
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT,OUTPUT,create,fingerprint,read_rows,sha
from experiments.paired_deliberation.preflight_d2 import require_engineering_release,ACTORS,source_names
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.training_evidence_view import load
from experiments.paired_deliberation.runner_d2_live import clients,shared_initial,run_method
from experiments.paired_deliberation.budget_d2 import JointBudget
from experiments.paired_deliberation.contracts_v6 import METHODS
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.transport_d2 import TransportD2,parse_d2_raw

PRIOR=ROOT/'experiments/results/paired_deliberation_d1_atomic_engineering_r3'
EVIDENCE=ROOT/'experiments/results/paired_deliberation_training_evidence_v1'


def prepare(root,validation):
    root,validation=Path(root),Path(validation).resolve();vp,score=require_engineering_release(validation)
    previous=json.loads((PRIOR/'plan.json').read_text());serving={r['case_id']:r for r in read_rows(OUTPUT/'data/dev.inputs.jsonl')}
    items=[]
    for cid in previous['case_ids']:
        e=json.loads((PRIOR/'inputs'/(cid.replace(':','_')+'.json')).read_text())
        if fingerprint(e)!=previous['case_hashes'][cid]:raise ValueError('prior engineering input changed')
        view=load(EVIDENCE,serving[cid],[m['movie_id'] for m in e['candidates_in_shared_retrieval_order']],'independent')
        reg=EvidenceRegistry(e,view);items.append({'id':cid,'evidence':e,'training_view':view,'registry_hash':reg.integrity})
    max_sources=max(len(item['evidence']['observations']) for item in items)
    source_cap=min(4096,max(768,192*max_sources))
    # 3 cases * 2 actors * (peer old pair + 3 scoped old/new pairs) = 84 requests.
    verifier_upper=(84*12000*vp['verifier']['input_cny_per_million']+42*(512+source_cap)*vp['verifier']['output_cny_per_million'])/1_000_000
    if verifier_upper>3.0:raise ValueError('complete verifier batch bound exceeds 3 yuan; new smaller protocol required before collection')
    plan={'version':'d2_shared_initial_engineering_v1','validation_root':str(validation),
          'validation_score_sha256':sha(validation/'release_score.json'),'semantic_plan_hash':vp['plan_hash'],
          'actors':ACTORS,'verifier':vp['verifier'],'case_ids':[r['id'] for r in items],'inputs_hash':fingerprint(items),
          'rounds':1,'expected_paths':30,'max_actor_physical_attempts_each':39,'max_verifier_physical_attempts':84,
          'max_actor_cny_each':1.0,'max_verifier_cny':3.0,'verifier_full_batch_upper_cny':verifier_upper,
          'limits':{'max_calls':12,'max_evidence_checks':2,'per_call_prompt_tokens':64000,'per_call_completion_tokens':4096,
                    'total_prompt_tokens':12*64000,'total_completion_tokens':12*4096,'max_estimated_cny':1.0},
          'code_sha256':{n:sha(ROOT/n) for n in source_names()},
          'gates':{'all_paths_completed':True,'contract_rejections_max':0,'invalid_verifier_outputs_max':0,
                   'unknown_usage_paths_max':0,'shared_evidence_mismatch_max':0,'pending_intents_max':0},
          'runtime_labels_available':False,'pilot_allowed':False,'phase':'engineering_only',
          'stop':'per actor stop after first case with contract/verifier/initial failure; transport/pending halts entire batch; no retry or prompt repair',
          'scope':'three already-used development inputs, common reports and predictive evidence; no semantic accuracy or recommendation score'}
    plan['plan_hash']=fingerprint(plan);create(root/'inputs.json',items);create(root/'plan.json',plan)
    for n in plan['code_sha256']:
        dst=root/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    return plan


def verify_engineering(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    if plan['version']!='d2_shared_initial_engineering_v1' or plan['plan_hash']!=fingerprint({k:v for k,v in plan.items() if k!='plan_hash'}):raise ValueError('engineering plan changed')
    require_engineering_release(plan['validation_root'])
    if sha(Path(plan['validation_root'])/'release_score.json')!=plan['validation_score_sha256']:raise ValueError('semantic release changed')
    if fingerprint(json.loads((root/'inputs.json').read_text()))!=plan['inputs_hash']:raise ValueError('engineering inputs changed')
    for n,h in plan['code_sha256'].items():
        if sha(ROOT/n)!=h or sha(root/'source_snapshot'/n)!=h:raise ValueError('engineering source changed')
    return plan


def store(path,result,plan_hash):create(path,{'result':result,'integrity':fingerprint(result),'plan_hash':plan_hash})
def read_saved(path,plan_hash):
    saved=json.loads(path.read_text())
    if saved['integrity']!=fingerprint(saved['result']) or saved.get('plan_hash')!=plan_hash:raise ValueError('saved engineering record changed')
    return saved['result']


def execute(root):
    root=Path(root);plan=verify_engineering(root)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False);load_dotenv(ROOT/'.env.local',override=False)
    judge_slots=root/'slots/verifier'
    judge_provider=DurableProvider(judge_slots,plan,TransportD2('deepseek',plan['verifier'],judge_slots,
        plan['max_verifier_physical_attempts'],plan['max_verifier_cny'],12000,4096),parse_d2_raw,True)
    all_cases=[];global_halt=False
    for actor_name,actor_profile in plan['actors'].items():
        actor_slots=root/'slots'/actor_name
        actor_provider=DurableProvider(actor_slots,plan,TransportD2(actor_name,actor_profile,actor_slots,
            plan['max_actor_physical_attempts_each'],plan['max_actor_cny_each'],64000,4096),parse_d2_raw,True)
        profiles={'actor':actor_profile,'verifier':plan['verifier']};providers={'actor':actor_provider,'verifier':judge_provider}
        for item in json.loads((root/'inputs.json').read_text()):
            cid=item['id'];prefix=actor_name+'_'+cid.replace(':','_');reg=EvidenceRegistry(item['evidence'],item['training_view'])
            if reg.integrity!=item['registry_hash']:raise ValueError('registry changed')
            path=root/'cases'/(prefix+'.json')
            if path.exists():result=read_saved(path,plan['plan_hash'])
            else:
                result={'actor':actor_name,'case_id':cid,'plan_hash':plan['plan_hash'],'methods':{},'stop_actor':False,'global_halt':False}
                ledger=JointBudget(plan['limits'],profiles);cc=clients(ledger,providers)
                try:
                    common_path=root/'shared'/(prefix+'.json')
                    if common_path.exists():common=read_saved(common_path,plan['plan_hash'])
                    else:
                        common=shared_initial(reg,cc['actor'],prefix+'_common');store(common_path,common,plan['plan_hash'])
                    for method in METHODS:
                        method_path=root/'methods'/actor_name/(cid.replace(':','_')+'_'+method+'.json')
                        if method_path.exists():record=read_saved(method_path,plan['plan_hash'])
                        else:
                            try:
                                row=run_method(reg,method,providers,plan['limits'],profiles,prefix+'_'+method,common,max_rounds=1)
                                record={'status':'completed','result':row,'engineering_authorization':plan['semantic_plan_hash']}
                            except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                                record={'status':'failed','error':str(exc),'usage':getattr(exc,'d2_usage',None),
                                        'partial':getattr(exc,'d2_partial',None),'global_halt':isinstance(exc,RuntimeError)}
                            store(method_path,record,plan['plan_hash'])
                        result['methods'][method]=record
                        if record.get('global_halt'):result['global_halt']=True;break
                    result['shared_adaptations']=cc['actor'].adaptations
                except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                    result['failure']={'error':str(exc),'usage':ledger.summary()};result['global_halt']=isinstance(exc,RuntimeError)
                failed=bool(result.get('failure')) or len(result['methods'])!=5
                for rec in result['methods'].values():
                    failed|=rec['status']!='completed'
                    if rec['status']=='completed':
                        failed|=any(r['kind']=='contract' for r in rec['result']['rejected'])
                        failed|=any(r.get('verification_failed',False) for r in rec['result']['verification_cache'].values())
                result['stop_actor']=failed;store(path,result,plan['plan_hash'])
            all_cases.append(result);print('collected '+prefix,flush=True)
            if result.get('global_halt'):global_halt=True;break
            if result.get('stop_actor'):break
        if global_halt:break
    paths=[r for case in all_cases for r in case['methods'].values()]
    completed=[r['result'] for r in paths if r['status']=='completed']
    contract=sum(r['kind']=='contract' for row in completed for r in row['rejected'])
    invalid=sum(r.get('verification_failed',False) for row in completed for r in row['verification_cache'].values())
    pending=len(list((root/'slots').rglob('*.intent.json')))
    gates={'all_paths_completed':len(completed)==30,'zero_contract_rejections':contract==0,'no_invalid_verifier_outputs':invalid==0,
           'known_usage':all(r['usage']['usage_complete'] for r in completed),
           'budgets':all(r['usage']['attempts']<=12 and r['usage']['reserved_cost_cny']<=1.0+1e-12 for r in completed),
           'no_pending':pending==0,'no_initial_failure':not any(r.get('failure') for r in all_cases)}
    physical={}
    for name in ['verifier']+list(plan['actors']):
        raws=[json.loads(p.read_text()) for p in (root/'slots'/name).glob('*.raw.json')]
        bodies=[json.loads(r['raw']['body']) for r in raws if r['raw']['status_code']==200]
        profile=plan['verifier'] if name=='verifier' else plan['actors'][name]
        prompt=sum(b.get('usage',{}).get('prompt_tokens',0) for b in bodies);out=sum(b.get('usage',{}).get('completion_tokens',0) for b in bodies)
        physical[name]={'responses':len(raws),'pending':len(list((root/'slots'/name).glob('*.intent.json'))),
            'prompt_tokens':prompt,'completion_tokens':out,'models':sorted({b.get('model') for b in bodies}),
            'estimated_list_price_ceiling_cny':(prompt*profile['input_cny_per_million']+out*profile['output_cny_per_million'])/1_000_000}
    score={'version':plan['version'],'plan_hash':plan['plan_hash'],'pass':all(gates.values()),'gates':gates,
           'completed_paths':len(completed),'contract_rejections':contract,'invalid_verifier_outputs':invalid,'physical':physical,
           'initial_failures':[r for r in all_cases if r.get('failure')],'pilot_allowed':False,
           'scope':plan['scope'],'semantic_accuracy_validated':False,'recommendation_gain_validated':False}
    create(root/'engineering_acceptance.json',score)
    return score


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--validation');p.add_argument('--execute-authorized',action='store_true')
    args=p.parse_args();print(json.dumps(execute(args.output) if args.execute_authorized else prepare(args.output,args.validation),ensure_ascii=False,indent=2))
