"""Offline structural audit for the explicit game, not model-effect evidence."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random

from experiments.core.mechanism.signaling_game import SignalingGame
from experiments.mechanism.run_upgrade_pilot import freeze

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/audit/signaling_game_v1'


def fingerprint(path):return sha256(path.read_bytes()).hexdigest()


def main():
    rng=random.Random(20261024)
    def row(n):
        v=[rng.random()+.1 for _ in range(n)];s=sum(v)
        return tuple(x/s for x in v)
    records=[]
    for n in (2,3,4,5):
        for index in range(12):
            g=tuple(row(n) for _ in range(2));d=tuple(row(2) for _ in range(n))
            for strength in (.15,.5,1.,3.):
                game=SignalingGame(tuple(map(str,range(n))),g,d,strength,strength)
                baseline=game.readout(g,d)
                for first in ('generator','discriminator'):
                    solved=game.solve(first=first)
                    assert all(b['potential']>=a['potential']-1e-12 for a,b in zip(solved['trace'],solved['trace'][1:]))
                    records.append({'candidates':n,'fixture':index,'strength':strength,'first':first,
                                    'reference':asdict(game),'baseline':baseline,
                                    'result':solved})
    same_as_generator=sum(r['baseline']['generator']['answer']==r['result']['readout']['generator']['answer'] for r in records)
    wrong=SignalingGame(('wrong','right'),((.99,.01),(.01,.99)),((.99,.01),(.01,.99))).solve()
    assert wrong['readout']['generator']['answer']=='wrong' and wrong['converged_to_tolerance']
    summary={'constructed_solver_runs':len(records),'model_questions':0,'new_api_calls':0,
        'converged_to_numeric_tolerance':sum(r['result']['converged_to_tolerance'] for r in records),
        'max_residual':max(r['result']['residual']['max_gap'] for r in records),
        'same_readout_as_initial_generator':same_as_generator,
        'different_readout_from_initial_generator':len(records)-same_as_generator,
        'same_readout_as_initial_product':sum(r['baseline']['product']['answer']==r['result']['readout']['generator']['answer'] for r in records),
        'stable_wrong_counterexample':wrong,
        'scope':'constructed inputs only; differences not correctness gains; no universal convergence/global-optimum guarantee',
        'code_sha256':{str(p.relative_to(ROOT)):fingerprint(p) for p in
                       [Path(__file__),ROOT/'experiments/core/mechanism/signaling_game.py']}}
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'fixtures.json',records);freeze(OUT/'result.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('stable_wrong_counterexample','code_sha256')},indent=2))


if __name__=='__main__':main()
