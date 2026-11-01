"""Strict inspection and alternative recovery for frozen E1 runs.

Native E1 v3 already supports stable first-start metadata and run locking.
This optional entry verifies the recorded source list without regenerating plans.
Read-only by default. --resume requires an exclusive run lock and exact source
hashes, preserves first-start metadata and unknown intents, and uses the same
cumulative provider caps. A completed seal causes score-only replay, no API.
"""
import argparse,fcntl,json,os,datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint


def inspect(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());rows=json.loads((root/'inputs.json').read_text())
    if fingerprint(rows)!=plan['bounded_tasks_hash'] or [r['id']for r in rows]!=plan['task_ids']:raise ValueError('frozen input identity mismatch')
    for name,h in plan['source_sha256'].items():
        if sha(root/'source_snapshot'/name)!=h:raise ValueError('snapshot mismatch: '+name)
        if not (ROOT/name).exists()or sha(ROOT/name)!=h:raise ValueError('live source changed: '+name+'; do not overwrite old run')
    return rows,plan


def recover(root,resume=False):
    root=Path(root);rows,plan=inspect(root)
    with (root/'run.lock').open('a+')as lock:
        busy=False
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:busy=True
        status={'root':str(root.resolve()),'tasks':len(rows),'completed_cases':len(list((root/'cases').glob('case_*.json'))),'generation_sealed':(root/'generation_complete.json').exists(),'collector_lock_busy':busy,'source_hashes_match':True,'mode':'inspect','unknown_or_inflight_intents':[p.name for p in sorted((root/'slots').glob('*.intent.json'))],'network_calls_in_inspect':0}
        if not resume:return status
        if busy:raise RuntimeError('original collector still owns run.lock; no duplicate collector started')
        from experiments.evidence_consensus.runner import task_case,score
        if status['generation_sealed']:
            score(root);return {**status,'mode':'sealed_score_only','new_api_calls':0}
        from dotenv import load_dotenv
        from experiments.paired_deliberation.native_optimization_runtime import Runtime
        load_dotenv(ROOT/'.env.local',override=False)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        create(root/'recovery_sessions'/(stamp+'.json'),{'actual_pid':os.getpid(),'plan_hash':fingerprint(plan),'recovery_source_sha256':sha(Path(__file__)),'rule':'first started.json untouched; exact successful requests reused; unknown not resent'})
        api=Runtime(root,plan,cap=plan['physical_cap'],max_cny=plan['configured_max_cny']);results={}
        with ThreadPoolExecutor(max_workers=2)as pool:
            jobs={pool.submit(task_case,i,t,api,root):i for i,t in enumerate(rows)}
            for f in as_completed(jobs):results[jobs[f]]=f.result()
        create(root/'generation_complete.json',{'hashes':[fingerprint(results[i])for i in range(len(rows))],'task_ids':plan['task_ids'],'labels_available':False})
        score(root);return {**status,'mode':'resumed_then_sealed','new_api_calls':'see raw receipt ledger; not inferred from logical calls'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();print(json.dumps(recover(a.output,a.resume),ensure_ascii=False,indent=2))
