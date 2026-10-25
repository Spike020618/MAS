"""Post-hoc split-sensitivity, not 20 independent replications or model selection."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.mechanism import audit_simple_readout_crossfit as base
from experiments.mechanism.attribute_legacy_gain import ROOT,trace_kind
from experiments.mechanism.audit_legacy_source_groups import normalize,split_question
from experiments.mechanism.audit_legacy_conclusion_parsing import conclusion
from experiments.mechanism.run_upgrade_pilot import freeze

OUT=base.OUT/'fold_stability'


def main():
    path=ROOT/'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl'
    source=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    source=[r for r in source if '__static_aeic@' in r['condition']]
    splits=json.loads((ROOT/'experiments/mechanism/splits.json').read_text())['datasets']
    qs={d:{str(r['question_id']):r['question'] for r in json.loads((ROOT/sp['source']).read_text())['questions']} for d,sp in splits.items()}
    groups=json.loads((ROOT/'experiments/results/audit/legacy_gain_attribution_2026-10-22/source_groups.json').read_text())
    excluded={(d,str(r['id'])) for d,v in groups['datasets'].items() for r in v['grouping']['excluded_records']}
    rows=[]
    for r in source:
        if (r['dataset'],str(r['question_id'])) in excluded:continue
        turns=sorted([t for t in r['rounds'] if t['round_id']==3],key=lambda t:t['agent_id'])
        stem=normalize(split_question(qs[r['dataset']][str(r['question_id'])])[0])
        rows.append({'id':[r['dataset'],str(r['question_id'])],'group':r['dataset']+'\n'+stem,'gold':r['ground_truth'],
                     'turns':[{'agent':t['agent_id'],'route':trace_kind(t),'answer':conclusion(t['report']['C'],t['stance'],'literal_yes_no_only')} for t in turns]})
    assert len(rows)==370
    results=[]
    for split_index in range(20):
        fold=lambda r:int(sha256(('split-stability-'+str(split_index)+'\n'+r['group']).encode()).hexdigest()[:8],16)%5
        predictions=[]
        for f in range(5):
            training=[r for r in rows if fold(r)!=f];testing=[r for r in rows if fold(r)==f]
            assert not {r['group'] for r in training}&{r['group'] for r in testing}
            model=base.fit(training)
            predictions += [(r,base.predict(r['turns'],model)) for r in testing]
        results.append({'split_index':split_index,'n':370,
                        'correct':{policy:sum(p[policy]==r['gold'] for r,p in predictions) for policy in predictions[0][1]}})
    OUT.mkdir(parents=True,exist_ok=True)
    summary={policy:{'min':min(r['correct'][policy] for r in results),'max':max(r['correct'][policy] for r in results),
                     'mean':sum(r['correct'][policy] for r in results)/len(results)} for policy in results[0]['correct']}
    freeze(OUT/'result.json',{'scope':'post-hoc partition stability on same370 old records, NOT independent replications or confidence intervals',
        'selected_scope':'Static final-round source-clean literal parser, chosen after seeing earlier cross-fit; exploratory only',
        'new_model_calls':0,'all20_partitions':results,'summary':summary,
        'input_sha256':{str(path.relative_to(ROOT)):base.fingerprint(path)},'code_sha256':base.fingerprint(Path(__file__)),
        'limits':['do not choose the best partition','mean/range across overlapping folds is not a sampling CI',
                  'does not establish route weighting is novel or resolves semantic truth','historical feedback trajectories not rerun']})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
