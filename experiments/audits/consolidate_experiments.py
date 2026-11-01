"""Recompute native metrics, paired uncertainty and public-request checks. Zero API."""
import json,random
from pathlib import Path
from collections import defaultdict
from experiments.audits.final_integrity_audit import score,digest
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/consolidation_2026_10_31'

def paired(differences,draws=4000):
    rng=random.Random(20261031);n=len(differences)
    boots=sorted(sum(rng.choices(differences,k=n))/n for _ in range(draws))
    return {'n':n,'mean_difference':sum(differences)/n,'ci95':[boots[int(.025*draws)],boots[int(.975*draws)]],'scope':'exploratory question-paired bootstrap; not equivalence proof'}

def public_requests(collect,data):
    tasks={t['id']:t for t in json.loads((data/'tasks.json').read_text())};count=0;errors=[]
    for path in (collect/'slots').glob('*.json'):
        r=json.loads(path.read_text());tid=r['slot'].split('/')[0];task=tasks[tid]
        payload=json.loads(r['request']['messages'][-1]['content']);count+=1
        catalog={s['id']:f"{s['title']}: {' '.join(s['sentences'])}"for s in task['sources']}
        if set(payload)-{'question','sources','operation','observations'}:errors.append({'slot':r['slot'],'issue':'unexpected_top_level_field'})
        if payload.get('question')!=task['question']:errors.append({'slot':r['slot'],'issue':'question_mismatch'})
        sources=payload.get('sources',{})
        if any(sid not in catalog or text!=catalog[sid]for sid,text in sources.items()):errors.append({'slot':r['slot'],'issue':'source_text_not_from_public_context'})
        if payload.get('operation')!='readout'and'observations'in payload:errors.append({'slot':r['slot'],'issue':'peer_observations_in_independent_review'})
    return {'requests_checked':count,'errors':errors,'scope':'request projection and source-text provenance only; public development exposure and output contamination are separate questions'}

def main():
    out={'schema':'consolidation-summary.v1','api_calls':0,'cohorts':{}}
    for name in ('stage3_60','power200'):
        data=json.loads((OUT/'selection'/f'{name}.json').read_text());records=data['records'];by={(r['task_id'],r['method']):r for r in records};ids=sorted({r['task_id']for r in records});methods=sorted({r['method']for r in records})
        comparisons={m:paired([by[(id,m)]['native_f1']-by[(id,'random')]['native_f1']for id in ids])for m in methods if m!='random'}
        prefix='coordination_game_s3_v1'if name=='stage3_60'else'coordination_game_s3_power_v1'
        prepare='coordination_game_s3_prepare_v1'if name=='stage3_60'else'coordination_game_s3_power_prepare_v1'
        projection=public_requests(ROOT/'experiments/results'/prefix/'collect',ROOT/'experiments/results'/prepare)
        out['cohorts'][name]={'methods':data['methods'],'vs_random_native_f1':comparisons,'public_request_audit':projection,'random_vs_central':data['random_vs_central']}
    integrity=json.loads((OUT/'integrity/summary.json').read_text());out['integrity']=integrity
    out['method_identity']=json.loads((OUT/'method_identity/summary.json').read_text())
    resources=json.loads((OUT/'resources/audit.json').read_text());out['resources']=resources
    anchor=integrity['coordination_game_s3_anchor_v1']['native_f1']['full_context']
    game=out['cohorts']['stage3_60']['methods']['game']['native_f1']
    out['full_context_vs_game_same60']={'native_f1_difference':anchor-game,'anchor_native_f1':anchor,'game_native_f1':game,
      'cost_ratio_anchor_over_game':resources['same60_full_context']['estimated_cny']/resources['same60_methods_with_initial_charged']['game']['standalone_cny'],
      'scope':'observed context/inference workflow comparison, not isolated coordination effect; missing billed usage remains unknown'}
    out['decisions']={
      'rerun_completed_generation':False,
      'solver_effect_claim_supported':False,
      'all_game_mechanisms_disproven':False,
      'deployable_evidence':'existing full-context baseline outperforms multi-call budgeted pipeline in this 60-question cohort at lower reported usage cost',
      'next_optimization':'evidence-use audit then same-task prediction calibration; do not tune old static aliases or train on these cohorts and call them fresh tests',
      'engineering_status':'receipt hashes intact; power plan metadata reconstructable; reject classification and native scoring corrected in derived artifacts; unknown completion costs not certified zero'}
    (OUT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps({'cohorts':{k:{'vs_random_native_f1':v['vs_random_native_f1'],'request_errors':len(v['public_request_audit']['errors'])}for k,v in out['cohorts'].items()},'full_context':out['full_context_vs_game_same60']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
