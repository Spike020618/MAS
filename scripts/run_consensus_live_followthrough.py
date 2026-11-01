"""Persist the remaining real stages after an already running training process.

Does not restart a failed training job or retry unknown API intents. Logs and
status survive chat interruption; the ordinary shell entry remains canonical.
"""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args();root=a.root.resolve();repo=Path(__file__).resolve().parent.parent;status=root/'live_status.json';log=root/'live_followthrough.log'
 def record(**fields):
  tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps({'updated_at_unix':time.time(),**fields},ensure_ascii=False,indent=2));tmp.replace(status)
 record(state='waiting_for_training',training_pid=a.wait_pid,supervisor_pid=os.getpid())
 try:
  deadline=time.monotonic()+4*3600
  while True:
   try:os.kill(a.wait_pid,0)
   except ProcessLookupError:break
   if time.monotonic()>deadline:raise RuntimeError('training wait timeout; no subsequent requests dispatched')
   time.sleep(5)
  if not(root/'model/model.joblib').exists():raise RuntimeError('training ended without a fitted model; inspect probe/train, no subsequent requests dispatched')
  stages=('main','value','ablations','budget','stress','transfer','semantic','report')
  with log.open('a',buffering=1)as stream:
   for stage in stages:
    record(state='running',stage=stage,supervisor_pid=os.getpid());stream.write('\nSTAGE '+stage+'\n')
    command=[str(repo/'.venv-native-benchmark/bin/python'),'-m','experiments.consensus_mechanism.cli',stage,'--root',str(root)]
    if stage!='report':command+=['--run']
    result=subprocess.run(command,cwd=repo,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('stage failed: '+stage+'; exit='+str(result.returncode))
  record(state='completed',report=str(root/'report.md'))
 except Exception as exc:
  record(state='failed',error=str(exc),log=str(log));raise
if __name__=='__main__':main()
