"""Known Shapley attribution of the observed 3-query input configurations only."""
from itertools import combinations
from math import factorial
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism.run_upgrade_pilot import freeze

OUT=p.ROOT/'experiments/results/redesign_v1/observed_shapley_v1'


def shapley(values,n=3):
    players=range(n); coalitions={frozenset(s) for k in range(n+1) for s in combinations(players,k)}
    if set(values)!=coalitions:raise ValueError('missing observed coalition; no imputation')
    out=[]
    for i in players:
        value=0.
        for s in coalitions:
            if i in s:continue
            weight=factorial(len(s))*factorial(n-len(s)-1)/factorial(n)
            value+=weight*(values[s|{i}]-values[s])
        out.append(value)
    if abs(sum(out)-(values[frozenset(players)]-values[frozenset()]))>1e-9:raise AssertionError('efficiency violated')
    return out


def main():
    p.verify();outputs=p.base.read(p.OUT/'sealed_outputs.json');labels=p.base.read(p.OUT/'private_labels.json');cases=[]
    for cid in labels:
        values={frozenset():p.answer_score(outputs[cid]['direct']['answer'],labels[cid])['em']}
        for i in range(3):values[frozenset([i])]=p.answer_score(outputs[cid]['s'+str(i)]['answer'],labels[cid])['em']
        for i,j in combinations(range(3),2):values[frozenset([i,j])]=p.answer_score(outputs[cid][f'p{i}{j}']['answer'],labels[cid])['em']
        values[frozenset(range(3))]=p.answer_score(outputs[cid]['all']['answer'],labels[cid])['em']
        credits=shapley(values)
        cases.append({'id':cid,'query_credits':credits,'full_minus_empty':values[frozenset(range(3))]-values[frozenset()],
            'observed_values':{','.join(map(str,sorted(s))):v for s,v in values.items()}})
    mean=[sum(c['query_credits'][i] for c in cases)/len(cases) for i in range(3)]
    result={'scope':'post-seal attribution, not a new algorithm, payment or expected causal credit',
        'n':len(cases),'mean_query_credits':mean,'mean_full_minus_empty':sum(c['full_minus_empty'] for c in cases)/len(cases),
        'efficiency_residual':sum(mean)-sum(c['full_minus_empty'] for c in cases)/len(cases),
        'cases':cases,'model_calls':0,
        'limits':['input order differs across some identical evidence sets in the original protocol',
            'one frozen stochastic answer per configuration is not its expected payoff',
            'outcome labels used offline only, credit not available to a test-time agent',
            'positive credit to one query can coexist with zero total improvement']}
    OUT.mkdir(parents=True,exist_ok=True);freeze(OUT/'result.json',result)
    print(__import__('json').dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
