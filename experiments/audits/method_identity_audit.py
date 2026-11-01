"""Offline method identity check. Never changes frozen artifacts or issues API calls."""
import itertools,json,random
from pathlib import Path
from experiments.coordination_game_v1.stage3_run import assignments_for
from experiments.coordination_game_v1.schema import digest

ROOT=Path(__file__).resolve().parents[2]
METHODS=('greedy','central','game','cbba')

def audit():
    rng=random.Random(20261031);synthetic=[]
    task={'id':'synthetic','candidates':['A','B','C']}
    for n in range(100):
        values={(m,c):rng.randint(-5,5)for m in range(3)for c in task['candidates']}
        selections={method:assignments_for(task,method,0,lambda t,m,c:values[m,c])for method in METHODS}
        objectives={method:sum(values[m,c]for m,c in chosen)for method,chosen in selections.items()}
        synthetic.append({'same_objective':len(set(objectives.values()))==1,'same_assignments':len({digest(sorted(s))for s in selections.values()})==1})
    collect=ROOT/'experiments/results/coordination_game_s3_v1/collect'
    panel=json.loads((collect/'panel.json').read_text())['records'];records={(r['task_id'],r['method']):r for r in panel}
    slots={}
    for path in (collect/'slots').glob('*.json'):
        data=json.loads(path.read_text());slots[data['slot']]=data
    pairs={}
    for left,right in itertools.combinations(METHODS,2):
        count=0;same_config=0;same_sources=0;same_reads=0;different_answers=0
        for tid in sorted({r['task_id']for r in panel}):
            a=records.get((tid,left));b=records.get((tid,right))
            if not a or not b:continue
            count+=1
            config_a=sorted(a['assignments']);config_b=sorted(b['assignments'])
            same_config+=config_a==config_b
            same_sources+=sorted(x[1]for x in config_a)==sorted(x[1]for x in config_b)
            request_a=[slots.get(f'{tid}/{left}/read-{i}',{}).get('request_hash')for i in range(2)]
            request_b=[slots.get(f'{tid}/{right}/read-{i}',{}).get('request_hash')for i in range(2)]
            same_reads+=None not in request_a+request_b and sorted(request_a)==sorted(request_b)
            if config_a==config_b:different_answers+=a['answer']!=b['answer']
        pairs[left+'__'+right]={'n':count,'same_assignment_set':same_config,'same_source_set':same_sources,'same_read_request_multiset':same_reads,'different_answers_despite_same_assignment':different_answers}
    return {'schema':'method-identity-audit.v1','api_calls':0,'synthetic_cases':100,'same_objective_cases':sum(x['same_objective']for x in synthetic),'same_assignment_cases':sum(x['same_assignments']for x in synthetic),'observed_60_pairs':pairs,
            'implementation_findings':[
            'stage3 game uses the same score-sort and source-dedup loop as greedy; no marginal-utility best-response solver is called',
            'stage3 cbba is a centralized highest-bid loop; no distributed bundles, message exchange, conflict timestamps or consensus convergence are measured',
            'central maximizes an additive value over distinct candidate sources with no per-member action limit; hence top-two per-source maxima is already optimal',
            'all methods force two reads even when every estimated value is nonpositive; stop is absent',
            'response stochasticity can create answer differences under equal assignments; these differences are not evidence of solver contribution'],
            'conclusion':'These arms test sampling and a shared additive selection rule, not distinct implemented game/CBBA coordination mechanisms. Preserve historical method names as recorded labels; qualify their implementation scope.'}

def main():
    output=ROOT/'experiments/results/consolidation_2026_10_31/method_identity'
    output.mkdir(parents=True,exist_ok=True);path=output/'summary.json'
    result=audit()
    if path.exists()and json.loads(path.read_text())!=result:raise ValueError('audit output changed; use a versioned destination')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
