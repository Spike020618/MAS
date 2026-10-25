"""Validate new case labels and rendered options against pinned original pools."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

from experiments.mechanism import study_information_complementarity as design
from experiments.mechanism import study_two_backend_complementarity as main
from experiments.mechanism import confirm_thinking_control as confirm
from experiments.mechanism.prepare_routing_holdout import PINS
from experiments.mechanism.prepare_upgrade_p2 import pool,native_mc
from experiments.mechanism.prepare_upgrade_pilot import canonical
from experiments.mechanism.run_upgrade_pilot import freeze


def verify():
    dep=Path('/private/tmp/mas_p2_dependencies')
    if dep.is_dir():sys.path.insert(0,str(dep))
    args=SimpleNamespace(
        strategyqa=design.p.ROOT/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json',
        truthfulqa=design.p.ROOT/'experiments/data/raw/upgrade_p2_v1/truthfulqa_mc_authors.json',
        mmlu=design.p.ROOT/'experiments/data/raw/upgrade_p2_v1/mmlu_test.parquet')
    for d,h in PINS.items():assert design.fingerprint(getattr(args,d))==h
    mmlu,args.mmlu=args.mmlu,None
    source,_=pool(args)
    import pyarrow.parquet as pq
    for row in pq.read_table(mmlu).to_pylist():
        if len(set(row['choices']))==len(row['choices']):
            source.append(native_mc('mmlu',design.p.digest([row['subject'],row['question']])[:24],
                                    row['question'],list(row['choices']),int(row['answer'])))
    native={r['domain']+':'+r['source_id']:r for r in source}
    checked=[];seen_ids=set();seen_stems=set()
    for directory in (main.OUT,confirm.DATA):
        cases=json.loads((directory/'public_cases.json').read_text())
        labels={r['id']:r['label'] for r in json.loads((directory/'private_labels.json').read_text())}
        assert len(cases)==len(labels)==36
        for case in cases:
            expected=native[case['id']]
            assert case['question']==expected['question'] and case['choices']==expected['choices']
            assert labels[case['id']]==expected['label']
            stem=canonical(expected['base_question'])
            assert case['id'] not in seen_ids and stem not in seen_stems
            seen_ids.add(case['id']);seen_stems.add(stem)
            checked.append({'id':case['id'],'source_match':True,'label_match':True,'rendered_options_match':True})
        provenance=json.loads((directory/'provenance.json').read_text())
        for name,h in provenance['files'].items():assert design.fingerprint(directory/name)==h
    result={'status':'72 unique source groups and original labels/options verified',
            'source_pins':PINS,'checked':checked,'new_model_calls':0,
            'scope':'source integrity, not independent adjudication of benchmark factual truth or semantics'}
    target=main.OUT/'source_verification.json';freeze(target,result)
    print(result['status'])
    return result


if __name__=='__main__':verify()
