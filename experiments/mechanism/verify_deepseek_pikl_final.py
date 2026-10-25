"""Preserve v1 snapshot, correct prose with a new separately hashed record."""
import json
from experiments.mechanism import run_deepseek_pikl_pilot as p
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    previous=json.loads((p.OUT/'acceptance.json').read_text())
    for rel,value in previous['sha256'].items(): assert p.h(p.ROOT/rel)==value,rel
    readouts=json.loads((p.OUT/'sealed_readouts.json').read_text())
    assert sum(len(set(r['answers'].values()))==1 for r in readouts.values())==5
    assert all(r['answers']['ER-G']==r['answers']['G'] for r in readouts.values())
    assert all(r['answers']['ER-D']==r['answers']['D'] for r in readouts.values())
    paths=[p.ROOT/'docs/research/DeepSeek_piKL逐题核查更正_2026-10-25.md',
           p.ROOT/'experiments/mechanism/verify_deepseek_pikl_final.py',p.OUT/'acceptance.json']
    result={'status':'final_verified_no_increment_in_corresponding_readouts',
            'all_six_methods_identical_cases':5,'total_cases':6,
            'ER_G_equals_G_cases':6,'ER_D_equals_D_cases':6,
            'original_snapshot_preserved':True,'new_api_calls':0,
            'sha256':{str(path.relative_to(p.ROOT)):p.h(path) for path in paths}}
    freeze(p.OUT/'final_acceptance.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
