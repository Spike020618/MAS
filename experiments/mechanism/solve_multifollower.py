"""Finite pure-commitment Stackelberg-Nash enumeration from complete payoffs.

Each follower must have no profitable unilateral deviation. Missing pure Nash
equilibria are reported; no simultaneous rationality or mixed solution invented.
"""
from itertools import product
import math


def solve_multifollower(game,tolerance=1e-10):
    agents=game['agents']
    if not agents or len(set(agents))!=len(agents): raise ValueError('unique agents required')
    profiles=list(product(*(game['actions'][a] for a in agents)))
    if not profiles: raise ValueError('nonempty action sets required')
    contracts=game['contracts']
    if len({c['id'] for c in contracts})!=len(contracts): raise ValueError('duplicate contracts')
    results=[]
    for contract in contracts:
        table={tuple(r['profile']):r for r in contract['payoffs']}
        if len(table)!=len(contract['payoffs']) or set(table)!=set(profiles):
            raise ValueError('complete unique joint-action payoff table required')
        for row in table.values():
            if len(row['followers'])!=len(agents): raise ValueError('payoff dimension mismatch')
            if not all(math.isfinite(float(v)) for v in [row['leader']]+row['followers']):
                raise ValueError('finite payoffs required')
        equilibria=[]
        for profile,row in table.items():
            regrets=[]
            for i,agent in enumerate(agents):
                values=[]
                for action in game['actions'][agent]:
                    deviation=list(profile); deviation[i]=action
                    values.append(table[tuple(deviation)]['followers'][i])
                regrets.append(max(values)-row['followers'][i])
            if max(regrets)<=tolerance:
                equilibria.append({'profile':list(profile),'leader':row['leader'],
                                   'followers':row['followers'],'unilateral_regrets':regrets})
        results.append({'contract_id':contract['id'],'pure_nash':equilibria})
    solutions={}
    for convention in ['optimistic','pessimistic']:
        candidates=[]
        for result in results:
            if not result['pure_nash']: continue
            ordered=sorted(result['pure_nash'],key=lambda r:(r['leader'],str(r['profile'])))
            response=ordered[-1] if convention=='optimistic' else ordered[0]
            candidates.append({'contract_id':result['contract_id'],**response})
        solutions[convention]=max(candidates,key=lambda r:r['leader']) if candidates else None
    return {'scope':'finite pure leader commitments and pure follower Nash; supplied payoffs only',
            'contracts':results,'solutions':solutions,
            'contracts_without_pure_equilibrium':[r['contract_id'] for r in results if not r['pure_nash']],
            'mixed_equilibria_solved':False}
