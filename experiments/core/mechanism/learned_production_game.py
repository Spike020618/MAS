"""Finite empirical cooperative production game, NOT ECON/BNE reproduction.

Two software controllers choose retain(0)/thinking-upgrade(1). A fixed third
report anchors majority. Utility_i = delayed correctness - own upgrade cost.
Potential = correctness - total upgrade cost. Same observed resource menu is
also offered to a centralized empirical optimizer, an essential control.
"""
from collections import Counter
from fractions import Fraction
from itertools import product

PROFILES = tuple(product((0,1),repeat=2))
COST_PER_TOKEN = Fraction(1,50000)  # .02 reward units per 1000 upgrade tokens


def majority(answers):
    counts = Counter(a for a in answers if a not in (None,'abstain'))
    winner = counts.most_common(1)
    return winner[0][0] if winner and winner[0][1]>=2 else 'abstain'


def context(initial):
    if set(initial) != {'fast0','fast1','fast2'}:
        raise ValueError('exactly three initial reports required')
    answers=[initial['fast'+str(i)]['answer'] for i in range(3)]
    return 'unanimous' if len(set(answers))==1 and answers[0] not in (None,'abstain') else 'disagreement'


def outcome(slots, profile):
    if tuple(profile) not in PROFILES: raise ValueError('invalid action profile')
    selected=[slots['fast0']]+[slots[('think' if a else 'fast')+str(i+1)] for i,a in enumerate(profile)]
    return majority([r['answer'] for r in selected])


def table(records):
    """TRAIN records only; no question text, domain features, or test input."""
    if not records: raise ValueError('nonempty training data required')
    result={}
    for profile in PROFILES:
        correct=Fraction(sum(outcome(r['slots'],profile)==r['label'] for r in records),len(records))
        costs=tuple(COST_PER_TOKEN*Fraction(sum(r['slots']['think'+str(i+1)]['tokens'] for r in records),len(records))*a
                    for i,a in enumerate(profile))
        result[''.join(map(str,profile))]={'correct':str(correct),'costs':list(map(str,costs)),
            'utilities':list(map(str,(correct-costs[0],correct-costs[1]))),
            'potential':str(correct-sum(costs))}
    return result


def field(t, profile, name, player=None):
    value=t[''.join(map(str,profile))][name]
    return Fraction(value[player] if player is not None else value)


def best_response(t):
    current=(0,0); trace=[{'profile':list(current),'potential':str(field(t,current,'potential'))}]
    for _ in range(8):
        changed=False
        for i in range(2):
            other=list(current);other[i]=1-other[i];other=tuple(other)
            if field(t,other,'utilities',i)>field(t,current,'utilities',i):
                assert field(t,other,'potential')>field(t,current,'potential')
                current=other;changed=True
                trace.append({'profile':list(current),'potential':str(field(t,current,'potential')),'player':i})
        if not changed: break
    regrets=[]
    for i in range(2):
        other=list(current);other[i]=1-other[i]
        regrets.append(max(Fraction(0),field(t,other,'utilities',i)-field(t,current,'utilities',i)))
    assert max(regrets)==0
    return {'profile':list(current),'trace':trace,'empirical_regrets':list(map(str,regrets))}


def fit(records):
    groups={'global':records}
    for state in ('unanimous','disagreement'):
        subset=[r for r in records if r['context']==state]
        if len(subset)>=3: groups[state]=subset
    result={'cost_per_token':str(COST_PER_TOKEN),'contexts':{},'scope':'empirical finite potential game; no population equilibrium or natural LLM rationality claim'}
    for name, rows in groups.items():
        t=table(rows)
        # Fixed lexicographic tie order shared across methods.
        erm=max(PROFILES,key=lambda p:field(t,p,'potential'))
        result['contexts'][name]={'n':len(rows),'table':t,'game':best_response(t),
                                  'centralized_erm':list(erm)}
    return result


def decide(policy, initial, method):
    """Only frozen policy + label-free initial reports are accepted."""
    state=context(initial)
    selected=policy['contexts'].get(state,policy['contexts']['global'])
    if method=='game': return selected['game']['profile']
    if method=='centralized_erm': return selected['centralized_erm']
    raise ValueError('unknown learned controller')
