"""Post-hoc piKL update diagnostic on OLD token-code scores; not replication."""
from hashlib import sha256
import json
from math import log
from pathlib import Path
from experiments.core.mechanism.equilibrium_ranking_reference import solve
from experiments.core.mechanism.scored_signaling import rank
from experiments.mechanism.run_upgrade_pilot import freeze, lock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'experiments/results/resource_allocation/signaling_pilot_v1'
OUT = ROOT/'experiments/results/audit/equilibrium_reference_replay_2026-10-25'


def main():
    with lock(OUT):
        cases = json.loads((SOURCE/'public_cases.json').read_text())
        sealed = json.loads((SOURCE/'sealed_outputs.json').read_text())
        readouts = {}
        for case in cases:
            ids = case['choices']; codes = 'ABCDEFGHIJKLMNOPQRST'[:len(ids)]
            slots = sealed[case['id']]
            required = ['g_positive','g_negative']+['d/'+x for x in ids]
            if not all(slots[key]['usable'] for key in required):
                readouts[case['id']] = {'ER-G':None, 'ER-D':None, 'reason':'incomplete_token_scores'}
                continue
            # Raw returned code masses, not conditional normalized rows.
            g = tuple(tuple(log(slots[key]['mass'][code]) for code in codes)
                      for key in ('g_positive','g_negative'))
            d = tuple(tuple(log(slots['d/'+cid]['mass'][code]) for code in ('A','B')) for cid in ids)
            result = solve(g,d)
            readouts[case['id']] = {'ER-G':rank(ids, result['generator'][0]),
                'ER-D':rank(ids, [row[0] for row in result['discriminator']]), 'result':result}
        freeze(OUT/'sealed_readouts.json',readouts)
        # Already-open historical labels are read only AFTER current readouts seal.
        previous = json.loads((SOURCE/'result.json').read_text())
        labels = {r['id']:r['gold'] for r in previous['records']}
        old = {r['id']:r['answers'] for r in previous['records']}
        result = {'scope':'post-hoc OLD12 code-probability update diagnostic, NOT full-text reproduction or independent test',
            'new_api_calls':0, 'n':len(cases),
            'complete_inputs':sum('result' in r for r in readouts.values()),
            'metrics':{m:{'correct':sum(r[m]==labels[k] for k,r in readouts.items()),
                           'abstain':sum(r[m] in (None,'abstain') for r in readouts.values())} for m in ('ER-G','ER-D')},
            'previous_metrics':previous['metrics'],
            'changed_from_previous_game':{m:sum(r[m] not in (None,'abstain') and r[m]!=old[k]['game']
                                              for k,r in readouts.items()) for m in ('ER-G','ER-D')},
            'limits':['same old opened questions; no independent evidence',
                      'single-code masses are not full-answer sequence likelihoods',
                      'updates and reference normalization changed together; not isolated causal attribution',
                      'no parameter tuning or automatic expansion'],
            'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in
                (SOURCE/'public_cases.json',SOURCE/'sealed_outputs.json',SOURCE/'result.json')},
            'readouts_sha256':sha256((OUT/'sealed_readouts.json').read_bytes()).hexdigest()}
        freeze(OUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('previous_metrics','source_sha256')},indent=2))


if __name__ == '__main__': main()
