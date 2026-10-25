"""Offline journal replay and integrity acceptance of DeepSeek piKL pilot."""
import io
import json
from pathlib import Path
from hashlib import sha256
from unittest.mock import patch
from experiments.mechanism import run_deepseek_pikl_pilot as pilot
from experiments.mechanism.run_upgrade_pilot import freeze,read_rows


def main():
    out=pilot.OUT; root=pilot.ROOT
    protocol=json.loads((out/'protocol.json').read_text())
    for rel,value in protocol['code_sha256'].items(): assert pilot.h(root/rel)==value,rel
    assert pilot.h(out/'public_cases.json')==protocol['public_sha256']
    rows=read_rows(out/'responses.jsonl'); attempts=read_rows(out/'attempts.jsonl')
    assert len(rows)==len(attempts)==34==protocol['planned_calls']
    before={p.name:pilot.h(p) for p in out.iterdir() if p.is_file() and p.name!='run.lock'}
    with (patch('socket.socket.connect',side_effect=AssertionError('offline replay')),
          patch('socket.getaddrinfo',side_effect=AssertionError('offline replay')),
          patch('builtins.print')):
        pilot.run()
    assert before=={name:pilot.h(out/name) for name in before}
    readouts=json.loads((out/'sealed_readouts.json').read_text())
    result=json.loads((out/'result.json').read_text())
    assert result['complete_cases']==6 and result['unknown_usage']==0
    identical=sum(len(set(r['answers'].values()))==1 for r in readouts.values())
    diagnosis={'n':6,'identical_all_six_readouts':identical,
        'interpretation':'no selected answer changes under fixed piKL on these six inputs; not evidence that all games fail',
        'new_api_calls_during_verification':0,
        'sampling_uncertainty':'six previously unexecuted questions, one scoring realization each; developmental only',
        'separate_old_replay':'12 old opened questions remain separate; do not pool as independent sample'}
    freeze(out/'diagnosis.json',diagnosis)
    reference=root/'experiments/results/audit/equilibrium_reference_replay_2026-10-25/acceptance.json'
    old=json.loads(reference.read_text())
    for rel,value in old['sha256'].items(): assert pilot.h(root/rel)==value,rel
    paths=[p for p in out.iterdir() if p.is_file() and p.name not in ('run.lock','acceptance.json')]
    paths += [Path(__file__),Path(pilot.__file__),root/'docs/research/DeepSeek_piKL小批量结果_2026-10-25.md']
    accepted={'status':'api_adaptation_complete_no_increment_on_six_cases',
        'physical_calls':34,'verification_calls':0,'known_tokens':4258,
        'reference_acceptance_sha256':pilot.h(reference),
        'sha256':{str(p.relative_to(root)):pilot.h(p) for p in paths}}
    freeze(out/'acceptance.json',accepted)
    print(json.dumps(diagnosis,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
