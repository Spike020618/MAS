"""Fixed multi-choice token adapter for a developmental same-model comparison.

All observed distributions are required. No gold/domain input, no score repair.
This is restricted token coding, not full-sequence likelihood replication.
"""
from math import exp, isfinite
from experiments.core.mechanism.signaling_game import SignalingGame, distribution


def decode_score(response, codes):
    result={'usable':False,'reason':'invalid_http_or_response','tokens':None}
    try:
        import json
        body=json.loads(response['body'])
        result['tokens']=body.get('usage',{}).get('total_tokens')
        result['model']=body.get('model');result['fingerprint']=body.get('system_fingerprint')
        if response['http_status']!=200:return result
        choice=body['choices'][0];text=choice['message']['content']
        positions=(choice.get('logprobs') or {}).get('content') or []
        if choice['finish_reason']!='stop' or text not in codes or len(positions)!=1 or positions[0]['token']!=text:
            result['reason']='not_complete_exact_single_code_with_probability';return result
        masses={}
        for token in positions[0].get('top_logprobs',[]):
            name=token.get('token');value=token.get('logprob')
            if name not in codes:continue
            if name in masses or type(value) not in (int,float) or not isfinite(value) or value>0:
                result['reason']='invalid_candidate_probability';return result
            masses[name]=exp(value)
        if set(masses)!=set(codes) or min(masses.values())<=0 or sum(masses.values())>1+1e-8:
            result['reason']='missing_or_unrepresentable_candidate_mass';return result
        total=sum(masses.values())
        result.update(usable=True,reason='conditional_exact_code_distribution',sampled_code=text,
                      mass=masses,mass_total=total,probabilities={k:masses[k]/total for k in codes})
        return result
    except (KeyError,TypeError,ValueError,IndexError,OverflowError):return result


def rank(choices, scores):
    order=sorted(range(len(scores)),key=lambda i:-scores[i])
    return choices[order[0]] if scores[order[0]]-scores[order[1]]>1e-10 else 'abstain'


def compare(choices, generator_positive, generator_negative, verifiers, strength=.5):
    """Return ALL controls; parameters fixed, never select using correctness."""
    choices=tuple(choices);n=len(choices)
    if len(set(choices))!=n or n<2:raise ValueError('invalid candidates')
    g=(distribution(generator_positive,n,'G+'),distribution(generator_negative,n,'G-'))
    if len(verifiers)!=n:raise ValueError('missing candidate verification')
    d=tuple(distribution(row,2,'D') for row in verifiers)
    gt=[[g[v][y]/sum(g[u][y] for u in range(2)) for y in range(n)] for v in range(2)]
    cross_g=tuple(tuple(p/sum(row) for p in row) for row in gt)
    dt=[[d[y][v]/sum(d[z][v] for z in range(n)) for v in range(2)] for y in range(n)]
    cross_d=tuple(tuple(p/sum(row) for p in row) for row in dt)
    game=SignalingGame(choices,cross_g,cross_d,strength,strength)
    solved=game.solve()
    scores={'generation':list(g[0]),'contrastive_generation':list(cross_g[0]),
            'verification':[r[0] for r in d],
            'raw_product':[g[0][i]*d[i][0] for i in range(n)],
            'contrastive_product':[cross_g[0][i]*cross_d[i][0] for i in range(n)],
            'game':list(solved['generator'][0])}
    answers={name:rank(choices,values) for name,values in scores.items()}
    if not solved['converged_to_tolerance']:answers['game']='abstain'
    return {'answers':answers,'scores':{name:dict(zip(choices,values)) for name,values in scores.items()},
            'game_reference':{'generator':cross_g,'discriminator':cross_d},
            'game_result':solved}
