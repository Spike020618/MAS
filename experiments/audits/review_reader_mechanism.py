"""Post-hoc mechanism audit, never imported by inference or task selection."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 root=ROOT/'experiments/results/reader_validation_new24_v1';summary=json.loads((root/'summary.json').read_text());tasks={t['id']:t for t in json.loads((root/'data/tasks.json').read_text())};by={(r['task_id'],r['arm']):r for r in summary['records']};records=[]
 for path in(root/'collect/slots').glob('*.json'):
  receipt=json.loads(path.read_text());tid,arm=receipt['slot'].split('/')
  if arm!='bound_chain5':continue
  parsed=receipt['response'].get('parsed')or{};catalog={s['id']:s for s in tasks[tid]['sources']};steps=[]
  for step in parsed.get('steps',[]):
   source=catalog.get(step.get('source_id'));indices=step.get('sentence_indices',[])
   steps.append({'relation':step.get('relation'),'source_id':step.get('source_id'),'title':source['title']if source else None,'original_sentences':[source['sentences'][i]for i in indices]if source else[]})
  delta=by[tid,arm]['native_f1']-by[tid,'plain5']['native_f1']
  records.append({'task_id':tid,'question':tasks[tid]['question'],'f1_delta':delta,'changed_score':delta!=0,'original_answer':by[tid,'plain5']['answer'],'chain_answer':parsed.get('answer'),'reported_steps':steps,'semantic_entailment':'not_certified','reference_binding_valid':by[tid,arm]['binding_valid']})
 # Manual reviews are diagnostic data only. Never used as prompt examples or rules.
 reviews=[
 {'task_id':'5a81624d5542995ce29dcbf1','finding':'F1 gain without established question-entity link. Cited constituency is represented by James Gray; reported chain does not identify the 2015 Universities and Science minister or establish its constituency. Correct chamber alone is insufficient proof.'},
 {'task_id':'5a79311755429970f5fffe67','finding':'F1 gain with a broken author bridge. First step cites Silver Spoon/Hiromu Arakawa, second cites Masakazu Katsura. The reported steps do not connect the requested manga to the birth year, despite a correct final year.'},
 {'task_id':'5a84c01e5542991dd0999dd6','finding':'Partial F1 gain with disconnected entities. Harald sons Eric/Haakon in first citation are not linked to Olaf Tryggvason in the second citation. Date overlap is not relational correctness.'},
 {'task_id':'5add4dd35542992ae4cec507','finding':'Small F1 gain does not establish comparison. One citation describes Squarepusher album contrast, the other only names a Ceephax album; cited steps do not establish the shared musical property.'},
 {'task_id':'5ae4cda05542990ba0bbb148','finding':'Harm includes answer length/granularity and unsupported comparison. Serena-only ranking/greatest-player citations do not substantiate a Grand Slam count comparison with Mariaan de Swardt.'},
 {'task_id':'5a790e69554299148911f9d7','finding':'Entity/kinship error. Father/son citation about Alonso de Cordova y Figueroa does not support older-brother relation, nor connect to Archduke Charles.'}]
 out={'scope':'post-hoc developer semantic review of all six score-changing cases; not independent human annotation, not deployed evidence gate','all_cases':len(records),'binding_passed':sum(r['reference_binding_valid']for r in records),'score_improved':sum(r['f1_delta']>0 for r in records),'score_harmed':sum(r['f1_delta']<0 for r in records),'all_case_records':records,'score_changing_case_reviews':reviews,'decision':'Do not promote the reader as reliable evidence reasoning on F1 alone. No prompt changes, ID rules, answer aliases or label edits in this audit. Freeze this cohort as development. Next experiments must separately measure relation correctness and decision quality using task-independent controls.'}
 (root/'mechanism_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in out.items()if k not in('all_case_records','score_changing_case_reviews')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
