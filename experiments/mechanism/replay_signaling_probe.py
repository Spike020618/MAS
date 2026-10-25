"""Replay a single synthetic API fixture into explicit games; zero new calls."""
from dataclasses import asdict
import json
from pathlib import Path

from experiments.core.mechanism.token_signaling_adapter import game_from_probe
from experiments.mechanism.probe_signaling_interface import extract
from experiments.mechanism.run_upgrade_pilot import freeze,read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/audit/signaling_probe_replay_v1'
SOURCE=ROOT/'experiments/results/resource_allocation/signaling_interface_probe_v3'


def main():
    raw=read_rows(SOURCE/'responses.jsonl')
    slots={r['key'].split('/')[1]:extract(r) for r in raw}
    assert len(slots)==len(raw)==4
    models={r['returned_model'] for r in slots.values()};assert models=={'deepseek-flash'}
    records={}
    for mode in ('conditional','contrastive'):
        game=game_from_probe(slots,normalization=mode)
        records[mode]={'reference':asdict(game),'initial_readouts':game.readout(game.generator_reference,game.discriminator_reference),
                       'solved':game.solve()}
    result={'scope':'ONE synthetic fixture; interface-to-solver execution only, no accuracy experiment',
            'new_api_calls':0,'source':str(SOURCE.relative_to(ROOT)),'models':sorted(models),
            'negative_prompt_top_code':max(slots['generate_negative']['conditional_probabilities'],
                                          key=slots['generate_negative']['conditional_probabilities'].get),
            'warning':'negative-generation prompt still favors code A; abstract type is not verified factual truth',
            'variants':records}
    OUT.mkdir(parents=True,exist_ok=True);freeze(OUT/'result.json',result)
    print(json.dumps({'scope':result['scope'],'negative_prompt_top_code':result['negative_prompt_top_code'],
                     'variants':{k:{'answer':v['solved']['readout']['generator']['answer'],
                                    'residual':v['solved']['residual']['max_gap']} for k,v in records.items()}},indent=2))


if __name__=='__main__':main()
