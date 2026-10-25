"""Grouped cross-fit simple readout controls on old reports; exploratory replay.

No new model outputs, deployed savings, independence or novel weighted-voting
theory claimed. Baseline routines never inspect an evaluated row's label.
"""
from collections import Counter,defaultdict
import json
from math import log
from pathlib import Path

from experiments.mechanism.attribute_legacy_gain import ROOT,vote,trace_kind
from experiments.mechanism.audit_legacy_source_groups import normalize,split_question
from experiments.mechanism.audit_legacy_conclusion_parsing import conclusion
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.study_information_complementarity import fingerprint

OUT=ROOT/'experiments/results/audit/simple_readout_crossfit_2026-10-24'


def fit(training):
    routes=defaultdict(lambda:[0,0]);agents=defaultdict(lambda:[0,0])
    for row in training:
        for t in row['turns']:
            ok=t['answer']==row['gold']
            for cell in (routes[t['route']],agents[t['agent']]):cell[0]+=ok;cell[1]+=1
    weights={k:log((correct+1)/(n-correct+1)) for k,(correct,n) in routes.items()}
    best=min(agents,key=lambda a:(-agents[a][0]/agents[a][1],a))
    return {'route_weights':weights,'best_agent':best,
            'training_route_counts':dict(routes),'training_agent_counts':dict(agents)}


def predict(turns,model):
    valid=[t for t in turns if t['answer'] in ('yes','no')]
    margin=sum(1 if t['answer']=='yes' else -1 for t in valid)
    weighted=sum(model['route_weights'].get(t['route'],0)*(1 if t['answer']=='yes' else -1) for t in valid)
    decision=lambda m:'yes' if m>1e-12 else 'no' if m< -1e-12 else 'neutral'
    return {'historical_plurality':vote([t['answer'] for t in turns]),
            'binary_unweighted':decision(margin),'route_reliability_weighted':decision(weighted),
            'calibration_best_agent':next(t['answer'] for t in turns if t['agent']==model['best_agent'])}


def main():
    source=ROOT/'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl'
    data=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
    splitpath=ROOT/'experiments/mechanism/splits.json'
    split=json.loads(splitpath.read_text())['datasets'];questions={};hashes={}
    for d,sp in split.items():
        path=ROOT/sp['source'];questions[d]={str(q['question_id']):q['question'] for q in json.loads(path.read_text())['questions']}
        hashes[str(path.relative_to(ROOT))]=fingerprint(path)
    groupingpath=ROOT/'experiments/results/audit/legacy_gain_attribution_2026-10-22/source_groups.json'
    groups=json.loads(groupingpath.read_text())
    excluded={(d,str(r['id'])) for d,v in groups['datasets'].items() for r in v['grouping']['excluded_records']}
    grouped=defaultdict(list)
    for row in data:
        stem=normalize(split_question(questions[row['dataset']][str(row['question_id'])])[0])
        groupkey=row['dataset']+'\n'+stem
        from hashlib import sha256
        fold=int(sha256(('readout-crossfit-v1\n'+groupkey).encode()).hexdigest()[:8],16)%5
        for mode in ('original','literal_yes_no_only'):
            for rid in (1,3):
                turns=sorted([t for t in row['rounds'] if t['round_id']==rid],key=lambda t:t['agent_id'])
                item={'id':[row['dataset'],str(row['question_id'])],'group':groupkey,'fold':fold,'gold':row['ground_truth'],
                    'turns':[{'agent':t['agent_id'],'route':trace_kind(t),
                              'answer':t['stance'] if mode=='original' else conclusion(t['report']['C'],t['stance'],mode)} for t in turns]}
                grouped[(mode,'all',row['condition'],rid)].append(item)
                if tuple(item['id']) not in excluded:
                    grouped[(mode,'source_clean',row['condition'],rid)].append(item)
    results=[]
    for (mode,scope,condition,rid),rows in sorted(grouped.items()):
        predictions=[];models={}
        for fold in range(5):
            training=[r for r in rows if r['fold']!=fold];testing=[r for r in rows if r['fold']==fold]
            assert not {r['group'] for r in training}&{r['group'] for r in testing}
            model=fit(training);models[str(fold)]=model
            for r in testing:
                predictions.append({'id':r['id'],'fold':fold,'gold':r['gold'],'predictions':predict(r['turns'],model)})
        policies=list(predictions[0]['predictions'])
        results.append({'mode':mode,'scope':scope,'condition':condition,'round':rid,'n':len(rows),
            'correct':{a:sum(r['predictions'][a]==r['gold'] for r in predictions) for a in policies},
            'weighted_vs_binary':{'corrections':sum(r['predictions']['binary_unweighted']!=r['gold'] and r['predictions']['route_reliability_weighted']==r['gold'] for r in predictions),
                'harms':sum(r['predictions']['binary_unweighted']==r['gold'] and r['predictions']['route_reliability_weighted']!=r['gold'] for r in predictions)},
            'models':models,'predictions':predictions})
    OUT.mkdir(parents=True,exist_ok=True)
    hashes.update({str(source.relative_to(ROOT)):fingerprint(source),str(splitpath.relative_to(ROOT)):fingerprint(splitpath),
                   str(groupingpath.relative_to(ROOT)):fingerprint(groupingpath)})
    freeze(OUT/'result.json',{'scope':'post-hoc grouped cross-fit sanity baselines; not a new independent experiment or novel algorithm',
           'new_model_calls':0,'fold_rule':'five folds by normalized original source question hash; global not per-domain weights',
           'weight_rule':'log((correct+1)/(incorrect+1)) per actual execution route; unknown route weight0; no independence claim',
           'input_sha256':hashes,'code_sha256':fingerprint(Path(__file__)),'results':results,
           'limits':['historical reports/reputation generated under historical protocol, not rerun under new readout',
              'cross-fit labels are used in other folds, not a prospective sealed test; overlapping training folds create dependence',
              'bundle reliability combines backend, prompt, budget, question selection; not backend-only causality',
              'new readout cost and control-plane implementation not executed; no savings claimed',
              'all policies/scopes/modes reported; no post-hoc winner promoted as independent result']})
    for r in results:
        if r['scope']=='source_clean' and '__static_aeic@' in r['condition']:
            print(r['mode'],'round',r['round'],'n',r['n'],r['correct'])


if __name__=='__main__':main()
