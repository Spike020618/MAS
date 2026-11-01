"""Zero-API post-hoc coverage/selection audit; labels never deployed as features."""
import json,collections,statistics
from pathlib import Path
from experiments.audits.final_integrity_audit import score as native_score
from experiments.coordination_game_v1.stage3_run import question_features
ROOT=Path(__file__).resolve().parents[2]

def audit(data,collect):
 data=ROOT/data;collect=ROOT/collect
 tasks={t['id']:t for t in json.loads((data/'tasks.json').read_text())};vault=json.loads((data/'private/scoring.json').read_text());panel=json.loads((collect/'panel.json').read_text())
 slots={}
 for p in (collect/'slots').glob('*.json'):
  r=json.loads(p.read_text());slots[r['slot']]=r
 rows=[];features=collections.defaultdict(list)
 for t in tasks.values():
  for m in range(3):
   for c in t['candidates']:
    for k,v in question_features(t,m,c).items():features[k].append(v)
 for r in panel['records']:
  t=tasks[r['task_id']];gold=vault[r['task_id']];support=set(gold['supporting_facts']['title']);initial={s for xs in t['member_initial_sources'].values()for s in xs};chosen={a[1]for a in r['assignments']};allids=initial|chosen;titles={s['id']:s['title']for s in t['sources']}
  coverage=lambda ids:len(support&{titles[s]for s in ids})/max(1,len(support))
  slot=slots.get(r['task_id']+'/'+r['method']+'/readout');actual=set();parse_error=False
  if slot:
   try:actual=set(json.loads(slot['request']['messages'][-1]['content'])['sources'])
   except Exception:parse_error=True
  row={**r,'native_f1':native_score(r['answer'],gold['answer']),'support_title_coverage':coverage(allids),'initial_support_title_coverage':coverage(initial),'new_support_title_coverage':coverage(allids)-coverage(initial),'expected_readout_sources':sorted(allids),'actual_readout_sources':sorted(actual),'readout_source_mismatch':actual!=allids,'readout_parse_error':parse_error}
  rows.append(row)
 methods={}
 for m in sorted({r['method']for r in rows}):
  group=[r for r in rows if r['method']==m];full=[r for r in group if r['support_title_coverage']==1]
  methods[m]={'n':len(group),'native_f1':statistics.mean(r['native_f1']for r in group),'mean_support_title_coverage':statistics.mean(r['support_title_coverage']for r in group),'mean_added_support_title_coverage':statistics.mean(r['new_support_title_coverage']for r in group),'full_support_titles':len(full),'full_support_titles_but_f1_below_1':sum(r['native_f1']<1 for r in full),'readout_source_mismatches':sum(r['readout_source_mismatch']for r in group)}
 by={(r['task_id'],r['method']):r for r in rows};pairs=[]
 for tid in tasks:
  a=by.get((tid,'random'));b=by.get((tid,'central'))
  if a and b:pairs.append({'task_id':tid,'random_minus_central_f1':a['native_f1']-b['native_f1'],'random_minus_central_support_coverage':a['support_title_coverage']-b['support_title_coverage']})
 return {'methods':methods,'random_vs_central':{'wins':sum(p['random_minus_central_f1']>0 for p in pairs),'losses':sum(p['random_minus_central_f1']<0 for p in pairs),'wins_with_more_support_coverage':sum(p['random_minus_central_f1']>0 and p['random_minus_central_support_coverage']>0 for p in pairs),'wins_without_more_support_coverage':sum(p['random_minus_central_f1']>0 and p['random_minus_central_support_coverage']<=0 for p in pairs)},'features':{k:{'min':min(v),'max':max(v),'unique':len(set(v))}for k,v in features.items()},'scope':'Post-hoc native supporting-title coverage is diagnostic, not sufficient semantic evidence, not causal attribution. Scores independently normalized native token F1. No fitting or calls.','records':rows,'pairs':pairs}

def main():
 out=ROOT/'experiments/results/consolidation_2026_10_31/selection';out.mkdir(parents=True,exist_ok=True)
 for name,data,collect in [('power200','experiments/results/coordination_game_s3_power_prepare_v1','experiments/results/coordination_game_s3_power_v1/collect'),('stage3_60','experiments/results/coordination_game_s3_prepare_v1','experiments/results/coordination_game_s3_v1/collect')]:
  d=audit(data,collect);(out/(name+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2));print(name,json.dumps({k:v for k,v in d.items()if k in ('methods','random_vs_central','features')},ensure_ascii=False))
if __name__=='__main__':main()
