"""Explicit commit-v4 engineering collector gated by its own joint semantic batch."""
import argparse
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, OUTPUT, create, fingerprint, read_rows, sha
from experiments.paired_deliberation.validate_commit_v4 import verify_plan
from experiments.paired_deliberation.commit_engine import run_method, shared_initial
from experiments.paired_deliberation.contracts_v6 import METHODS
from experiments.paired_deliberation.training_evidence_view import load
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.schema_echo_client import SchemaEchoClient
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw
from experiments.paired_deliberation.run_v6 import acceptance

PRIOR=ROOT/'experiments/results/paired_deliberation_d1_atomic_engineering_r3'


def check_release(root):
    root=Path(root)
    plan=verify_plan(root)
    path=root/'release_score.json'
    if not path.exists():raise RuntimeError('commit-v4 joint validation incomplete')
    score=json.loads(path.read_text())
    if (score.get('contract_version')!='commit_v4' or score.get('plan_hash')!=plan['plan_hash']
            or not score.get('pass') or not score.get('gates') or not all(score['gates'].values())):
        raise RuntimeError('commit-v4 semantic gate failed; no engineering dispatch')
    return plan,score


def prepare(root,validation,evidence_root):
    root,validation,evidence_root=Path(root),Path(validation).resolve(),Path(evidence_root).resolve()
    vplan,score=check_release(validation)
    old=json.loads((PRIOR/'plan.json').read_text())
    serving={r['case_id']:r for r in read_rows(OUTPUT/'data/dev.inputs.jsonl')}
    items=[]
    for cid in old['case_ids']:
        evidence=json.loads((PRIOR/'inputs'/(cid.replace(':','_')+'.json')).read_text())
        if fingerprint(evidence)!=old['case_hashes'][cid]:raise ValueError('prior input changed')
        view=load(evidence_root,serving[cid],[r['movie_id'] for r in evidence['candidates_in_shared_retrieval_order']],'independent')
        items.append({'id':cid,'evidence':evidence,'training_view':view})
    files=set(vplan['code_sha256'])|{str(Path(__file__).relative_to(ROOT))}
    files|={'experiments/paired_deliberation/'+n for n in ('training_evidence_view.py','coordination_v6.py','run_v6.py')}
    limits={'max_calls':20,'max_evidence_checks':6,'per_call_prompt_tokens':64000,'per_call_completion_tokens':4096,
            'total_prompt_tokens':20*64000,'total_completion_tokens':20*4096}
    plan={'version':'commit_v4_shared_initial_engineering_r1','model':vplan['model'],'validation_root':str(validation),
          'validation_score_sha256':sha(validation/'release_score.json'),'validation_models':score['physical']['models'],
          'evidence_manifest_sha256':sha(evidence_root/'manifest.json'),'evidence_root':str(evidence_root),
          'case_ids':[r['id'] for r in items],'inputs_hash':fingerprint(items),'limits':limits,'rounds':3,
          'maximum_physical_attempts':180,'code_sha256':{n:sha(ROOT/n) for n in sorted(files)},
          'pilot_allowed':False,'runtime_labels_available':False,
          'scope':'three known development inputs, common initial reports, equal explicit training evidence; engineering only, not recommendation gain'}
    plan['plan_hash']=fingerprint(plan)
    create(root/'inputs.json',items);create(root/'plan.json',plan)
    for name in files:
        target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=(ROOT/name).read_bytes():raise ValueError('snapshot changed')
        target.write_bytes((ROOT/name).read_bytes())
    return plan


def execute(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    if plan['plan_hash']!=fingerprint({k:v for k,v in plan.items() if k!='plan_hash'}):raise ValueError('plan changed')
    check_release(plan['validation_root'])
    if sha(Path(plan['validation_root'])/'release_score.json')!=plan['validation_score_sha256']:raise ValueError('semantic score changed')
    if sha(Path(plan['evidence_root'])/'manifest.json')!=plan['evidence_manifest_sha256']:raise ValueError('evidence manifest changed')
    for name,wanted in plan['code_sha256'].items():
        if sha(ROOT/name)!=wanted or sha(root/'source_snapshot'/name)!=wanted:raise ValueError('source changed')
    items=json.loads((root/'inputs.json').read_text())
    if fingerprint(items)!=plan['inputs_hash']:raise ValueError('inputs changed')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False)
    raw=RawDeepSeek(model=plan['model']);slots=root/'slots'
    class Capped:
        def count(self):return len(list(slots.glob('*.raw.json')))+len(list(slots.glob('*.intent.json')))
        def preflight(self,request):
            raw.preflight(request)
            if self.count()>=plan['maximum_physical_attempts']:raise RuntimeError('batch cap reached')
        def __call__(self,request):
            if self.count()>plan['maximum_physical_attempts']:raise RuntimeError('batch cap reached')
            try:return raw(request)
            except OSError as exc:raise RuntimeError('transport pending; no automatic repeat: '+str(exc)) from exc
    provider=DurableProvider(slots,plan,Capped(),parse_raw,True);results=[]
    for item in items:
        cid=item['id'];prefix=cid.replace(':','_');evidence=item['evidence'];view=item['training_view']
        path=root/'cases'/(prefix+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['integrity']!=fingerprint(saved['result']) or saved['result']['plan_hash']!=plan['plan_hash']:raise ValueError('case changed')
            result=saved['result']
        else:
            result={'case_id':cid,'plan_hash':plan['plan_hash'],'methods':{}}
            client=SchemaEchoClient(AttemptClient(provider,plan['limits']))
            try:
                shared=shared_initial(evidence,client,prefix+'_shared',training_view=view)
                if {c['record']['model'] for c in shared['calls']}!=set(plan['validation_models']):raise ValueError('model differs from validation')
                create(root/'shared'/(prefix+'.json'),shared)
                for method in METHODS:
                    c=SchemaEchoClient(AttemptClient(provider,plan['limits']))
                    try:
                        row=run_method(evidence,method,c,prefix+'_'+method,shared,plan['rounds'],view)
                        result['methods'][method]={'status':'completed','result':row,'schema_echoes':c.ignored_echoes}
                    except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                        result['methods'][method]={'status':'failed','error':str(exc),'usage':c.summary()}
                        if isinstance(exc,RuntimeError):break
                result['shared_schema_echoes']=client.ignored_echoes
            except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                result['failure']={'error':str(exc),'usage':client.summary()}
            create(path,{'result':result,'integrity':fingerprint(result)})
        results.append(result);print('collected '+cid,flush=True)
        failed=bool(result.get('failure')) or len(result['methods'])!=5
        for entry in result['methods'].values():
            failed|=entry['status']!='completed'
            if entry['status']=='completed':
                failed|=any(r['kind']=='contract' for r in entry['result']['rejected'])
                failed|=any(r.get('verification_failed',False) for r in entry['result']['verification_cache'].values())
        if failed:break
    score=acceptance(results,plan)
    score.update(plan_hash=plan['plan_hash'],contract_version='commit_v4',pilot_allowed=False)
    raws=[json.loads(p.read_text()) for p in slots.glob('*.raw.json')]
    bodies=[json.loads(r['raw']['body']) for r in raws]
    score['physical']={'responses':len(raws),'pending_intents':len(list(slots.glob('*.intent.json'))),
                       'models':sorted({b.get('model') for b in bodies}),
                       'prompt_tokens':sum(b['usage']['prompt_tokens'] for b in bodies),
                       'completion_tokens':sum(b['usage']['completion_tokens'] for b in bodies)}
    score['gates']['no_pending']=not score['physical']['pending_intents'];score['pass']=all(score['gates'].values())
    create(root/'engineering_acceptance.json',score)
    return score


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--validation');parser.add_argument('--evidence-root');parser.add_argument('--execute-authorized',action='store_true')
    args=parser.parse_args();print(json.dumps(execute(args.root) if args.execute_authorized else prepare(args.root,args.validation,args.evidence_root),indent=2))
