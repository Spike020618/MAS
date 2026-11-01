"""Independent request projection validation for the new reader diagnostic."""
import json
from pathlib import Path
from experiments.coordination_game_v1.schema import digest
ROOT=Path(__file__).resolve().parents[2]
def check(root=None):
 root=Path(root or ROOT/'experiments/results/evidence_use_repair_v1')
 requests=json.loads((root/'requests.json').read_text());plan=json.loads((root/'collect/plan.json').read_text())
 tasks={t['id']:t for t in json.loads((ROOT/'experiments/results/coordination_game_s3_prepare_v1/tasks.json').read_text())};errors=[];groups={}
 if digest([tuple(x)for x in requests])!=plan['requests_hash']:errors.append('request_manifest_hash')
 for slot,request in requests:
  tid,method=slot.split('/');task=tasks[tid];p=json.loads(request['messages'][-1]['content'])
  if set(p)-{'question','sources','operation'}:errors.append('unexpected_public_field:'+slot)
  if p['question']!=task['question']:errors.append('question:'+slot)
  catalog={s['id']:s['title']+': '+' '.join(s['sentences'])for s in task['sources']}
  if any(k not in catalog or v!=catalog[k]for k,v in p['sources'].items()):errors.append('source_provenance:'+slot)
  groups.setdefault(tid,{})[method]=p['sources']
 for tid,arms in groups.items():
  if arms['raw_only']!=arms['chain_same5']:errors.append('same_sources:'+tid)
  if not set(arms['raw_only'])<=set(arms['chain_full10']):errors.append('full_sources:'+tid)
 return {'request_count':len(requests),'questions':len(groups),'errors':errors,'source_projection_valid':not errors,
  'cautions':['old baseline uses historical independent responses; revised requests canonicalize key/source order, so old-vs-new is a reader-package comparison, not pure removal-of-observations causality',
  'new raw-only and chain arms share identical source texts but system instructions/output schema and operation metadata differ; report the complete intervention',
  '12 exposed development questions selected by hash; not an independent confirmation set']}
if __name__=='__main__':
 result=check();p=ROOT/'experiments/results/evidence_use_repair_v1/preflight.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
