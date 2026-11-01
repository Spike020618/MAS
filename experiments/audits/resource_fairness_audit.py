"""Zero-API receipt-based resource audit. Never changes frozen artifacts."""
import json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def audit(name):
 root=ROOT/'experiments/results'/name
 groups=collections.defaultdict(lambda:dict(slots=0,complete=0,rejected=0,prompt_tokens=0,completion_tokens=0,estimated_cny=0.,source_occurrences=0,source_counts=[]))
 for path in (root/'slots').glob('*.json'):
  d=json.loads(path.read_text());slot=d.get('slot','');parts=slot.split('/')
  group='full_context'if'full_context'in slot else(parts[1]if len(parts)>2 else'shared_initial')
  if group.startswith('initial'):group='shared_initial'
  g=groups[group];g['slots']+=1;g['complete']+=d.get('status')=='complete';g['rejected']+=d.get('status')=='rejected'
  u=(d.get('response')or{}).get('usage',{})
  g['prompt_tokens']+=u.get('prompt_tokens',0);g['completion_tokens']+=u.get('completion_tokens',0)
  g['estimated_cny']+=(u.get('prompt_tokens',0)*.15+u.get('completion_tokens',0)*1.5)/1e6
  request=d.get('request',{});messages=request.get('messages',[])
  try:payload=json.loads(messages[-1]['content']);n=len(payload.get('sources',{}))
  except (ValueError,KeyError,IndexError):n=0
  g['source_occurrences']+=n
  if slot.endswith('readout')or slot.endswith('full_context'):g['source_counts'].append(n)
 for g in groups.values():
  counts=g.pop('source_counts');g['readout_source_count_mean']=sum(counts)/len(counts)if counts else None
 archived=[]
 for directory in root.iterdir():
  if directory.is_dir()and directory.name.startswith('slots_'):
   items=list(directory.glob('*.json'));tokens=0;complete=0
   for p in items:
    try:d=json.loads(p.read_text());u=(d.get('response')or{}).get('usage',{});tokens+=u.get('total_tokens',0);complete+=d.get('status')=='complete'
    except ValueError:pass
   archived.append({'directory':directory.name,'receipts':len(items),'complete':complete,'reported_tokens':tokens})
 return {'root':str(root.relative_to(ROOT)),'groups':dict(groups),'archived_superseded_receipts':archived,'accounting':'actual reported usage in final slot view; missing usage is unknown not free; shared initial cost charged once physically, once per method for standalone deployment; not invoice; copied/reused receipts need global dedup before total spend claim'}

def main():
 out=ROOT/'experiments/results/consolidation_2026_10_31/resources';out.mkdir(parents=True,exist_ok=True)
 results=[audit(n)for n in ['coordination_game_s3_v1/collect','coordination_game_s3_power_v1/collect','coordination_game_s3_anchor_v1']]
 anchor=results[2]['groups']['full_context'];shared=results[0]['groups'].get('shared_initial',{})
 comparison={}
 for m,g in results[0]['groups'].items():
  if m=='shared_initial':continue
  comparison[m]={'standalone_prompt_tokens':g['prompt_tokens']+shared.get('prompt_tokens',0),'standalone_output_tokens':g['completion_tokens']+shared.get('completion_tokens',0),'standalone_calls':g['slots']+shared.get('slots',0),'standalone_cny':g['estimated_cny']+shared.get('estimated_cny',0)}
 result={'runs':results,'same60_full_context':anchor,'same60_methods_with_initial_charged':comparison,'interpretation':['Full-context is an observed strong baseline, not a mathematical upper bound on model quality.','Budget-2 limits additional reads, not total evidence seen: initial sources plus selected sources are passed raw to common readout.','Calls and actual tokens must both be compared; full-context single call can cost less than multiple limited reads.']}
 (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({'anchor':anchor,'same60':comparison},ensure_ascii=False))
if __name__=='__main__':main()
