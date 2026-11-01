"""Post-hoc source-sentence audit, never a deployment feature or prompt constructor."""
import json,re,string,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def norm(x):return ' '.join(str(x).lower().translate(str.maketrans('','',string.punctuation)).split())
def main():
 base=ROOT/'experiments/results';selection=json.loads((base/'consolidation_2026_10_31/selection/power200.json').read_text())
 tasks={t['id']:t for t in json.loads((base/'coordination_game_s3_power_prepare_v1/tasks.json').read_text())};vault=json.loads((base/'coordination_game_s3_power_prepare_v1/private/scoring.json').read_text())
 slots={}
 for p in(base/'coordination_game_s3_power_v1/collect/slots').glob('*.json'):
  d=json.loads(p.read_text());slots[d['slot']]=d
 rows=[]
 for r in selection['records']:
  if r['method']!='central'or r['support_title_coverage']!=1 or r['native_f1']>=1:continue
  tid=r['task_id'];t=tasks[tid];g=vault[tid];receipt=slots[tid+'/central/readout'];payload=json.loads(receipt['request']['messages'][-1]['content']);text='\n'.join(payload['sources'].values());answer=g['answer'];pred=r['answer']
  facts=[]
  for title,index in zip(g['supporting_facts']['title'],g['supporting_facts']['sent_id']):
   matches=[s for s in t['sources']if s['title']==title];sentences=[s['sentences'][index]for s in matches if index<len(s['sentences'])];facts.append({'title':title,'sentence_index':index,'sentences':sentences,'present_verbatim':bool(sentences)and any(s in text for s in sentences)})
  present=norm(answer)in norm(text);pred_present=norm(pred)in norm(text);observations=json.dumps(payload.get('observations',[]),ensure_ascii=False)
  category='answer_literal_absent'if not present else('answer_present_abstained'if pred.lower()in('unknown','noanswer','not enough information')else('answer_present_wrong_entity_candidate'if pred_present else'answer_present_output_not_verbatim'))
  rows.append({'task_id':tid,'question':t['question'],'gold':answer,'prediction':pred,'native_f1':r['native_f1'],'category':category,'answer_literal_present_normalized':present,'prediction_literal_present_normalized':pred_present,'all_support_sentences_present':all(f['present_verbatim']for f in facts),'support_sentences':facts,'gold_mentioned_in_observations':norm(answer)in norm(observations),'prediction_mentioned_in_observations':norm(pred)in norm(observations),'raw_sources':payload['sources'],'observations':payload.get('observations',[]),'notes':'Entity confusion/observation influence are candidate explanations, not causal proof. Literal absence includes possible aliases.'})
 out=base/'evidence_use_repair_v1/diagnosis';out.mkdir(parents=True,exist_ok=True)
 summary={'cases':len(rows),'categories':dict(collections.Counter(r['category']for r in rows)),'all_support_sentences_present':sum(r['all_support_sentences_present']for r in rows),'answer_literal_present':sum(r['answer_literal_present_normalized']for r in rows),'gold_in_observations':sum(r['gold_mentioned_in_observations']for r in rows),'scope':'All 24 central errors with full supporting-title coverage; post-hoc labels confined to diagnosis, no prompt output or fitting. No alias truth adjudication inferred mechanically.'}
 (out/'cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
 for r in rows[:6]:print('\n',r['task_id'],r['question'],'GOLD',r['gold'],'PRED',r['prediction'],'FACTS',r['support_sentences'])
if __name__=='__main__':main()
