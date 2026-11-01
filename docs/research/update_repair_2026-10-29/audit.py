"""Read-only audit; never reads labels or sends requests."""
import json,collections
from pathlib import Path
root=Path('experiments/results/consensus_mechanism_paper_v2/main');out=Path('docs/research/update_repair_2026-10-29');counts=collections.Counter();errors=collections.Counter();cases=list((root/'cases').glob('case_*.json'))
for f in cases:
 case=json.loads(f.read_text())
 for event in case['methods']['game']['events']:
  if event.get('phase')!='review':continue
  counts['reviews']+=1;report=event.get('report')or{};counts['valid']+=bool(event['valid']);counts['accepted']+=bool(event.get('accepted'));counts['null_patch']+=report.get('patch')is None
  if event['valid']:counts['valid_'+str(report.get('verdict'))]+=1
for f in (root/'slots').glob('*.raw.json'):
 row=json.loads(f.read_text())
 if row.get('raw',{}).get('status_code')==400:
  try:errors[json.loads(row['raw']['body']).get('error',{}).get('message','unknown')]+=1
  except ValueError:errors['unparsed']+=1
result={'completed_cases':len(cases),'missing_indices':sorted(set(range(400))-{int(f.stem.split('_')[1])for f in cases}),'sealed':(root/'generation_complete.json').exists(),'game_review':dict(counts),'http400':dict(errors),'labels_read':False,'requests_sent':0}
(out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
