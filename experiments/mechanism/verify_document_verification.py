"""Offline full-source/requests/replay verification, not an efficacy proof."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from experiments.mechanism import run_document_verification as run
from experiments.core.mechanism.document_verification import retrieve
from experiments.mechanism.run_upgrade_pilot import read_rows,freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def main():
    old=frozen_hashes();out=run.OUT
    protocol=json.loads((out/'protocol.json').read_text())
    assert run.h(run.ROOT/'docs/research/文档核验行动_预定方案_2026-10-25.md')==protocol['design_sha256']
    for rel,value in protocol['code_sha256'].items():assert run.h(run.ROOT/rel)==value,rel
    selection=json.loads((out/'selection.json').read_text())
    assert run.h(run.ARCHIVE)==selection['archive_sha256']
    for name,value in selection['files'].items():assert run.h(out/name)==value,name
    with zipfile.ZipFile(run.ARCHIVE) as z:
        originals=[json.loads(line) for line in z.read(selection['member']).decode().splitlines() if line.strip()]
    cases=json.loads((out/'public_cases.json').read_text());labels=json.loads((out/'private_labels.json').read_text())
    sealed=json.loads((out/'sealed_outputs.json').read_text());assert len(cases)==len(sealed)==len(labels)==20
    assert len({c['article_id'] for c in cases})==len({c['article_sha256'] for c in cases})==20
    for c in cases:
        assert set(c)=={'id','article_id','question','choices','article','article_sha256'}
        source=[(row,q) for row in originals if str(row['article_id'])==c['article_id'] for q in row['questions'] if q['question']==c['question']]
        assert source
        assert any(row['article']==c['article'] and dict(zip('ABCD',q['options']))==c['choices']
                   and 'ABCD'[int(q['gold_label'])-1]==labels[c['id']] for row,q in source)
        directory=out/c['id'].replace(':','_');rows=read_rows(directory/'responses.jsonl')
        attempts=read_rows(directory/'attempts.jsonl')
        assert len(rows)==len(attempts)==6
        assert [r['key'] for r in rows]==['full','fixed','review1','review2','plan','local']
        retrieval=json.loads((directory/'retrieval.json').read_text())
        assert retrieval['windows']==retrieve(c['article'],retrieval['query'])
        for w in retrieval['windows']:assert w['text']==c['article'][w['start']:w['end']]
        local=json.loads(rows[-1]['request']['messages'][1]['content'])
        assert [w['text'] for w in local['source_windows']]==[w['text'] for w in retrieval['windows']]
        assert all('gold_label' not in json.loads(r['request']['messages'][1]['content']) for r in rows)
    def forbidden(*args,**kwargs):raise AssertionError('offline replay cannot use network')
    with (patch('socket.socket.connect',side_effect=forbidden),patch('socket.getaddrinfo',side_effect=forbidden),
          patch.object(run,'transport',side_effect=forbidden)):
        with patch('builtins.print'):
            run.run()
            from experiments.mechanism.audit_document_verification import main as audit
            audit()
        stream=io.StringIO()
        tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.test_document_verification','experiments.tests.test_equilibrium_ranking_reference',
             'experiments.tests.test_equilibrium_reference_pipeline','experiments.tests.test_learned_production_game',
             'experiments.tests.test_learned_production_runner']))
        assert tests.wasSuccessful(),stream.getvalue()
    assert frozen_hashes()==old
    previous=run.ROOT/'experiments/results/resource_allocation/learned_production_v1/acceptance.json'
    for rel,value in json.loads(previous.read_text())['sha256'].items():assert run.h(run.ROOT/rel)==value,rel
    result=json.loads((out/'result.json').read_text())
    assert result['physical_calls']==120
    paths=[p for p in out.rglob('*') if p.is_file() and p.name not in ('run.lock','acceptance.json')]
    paths += [Path(__file__),Path(run.__file__),run.ROOT/'experiments/mechanism/audit_document_verification.py',
              run.ROOT/'experiments/core/mechanism/document_verification.py',
              run.ROOT/'experiments/tests/test_document_verification.py',
              run.ROOT/'docs/research/文档核验行动_预定方案_2026-10-25.md',
              run.ROOT/'docs/research/文档核验行动_结果_2026-10-25.md',run.ARCHIVE]
    accepted={'status':'document_grounded_developmental_screen_complete', 'tests':tests.testsRun,
              'original_frozen_files':len(old),'new_calls_in_verification':0,
              'physical_calls':result['physical_calls'],'physical_tokens':result['physical_tokens'],
              'claims':'not game advantage, not independent confirmation or verified entailment',
              'sha256':{str(p.relative_to(run.ROOT)):run.h(p) for p in paths}}
    freeze(out/'acceptance.json',accepted)
    print(json.dumps({k:v for k,v in accepted.items() if k!='sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
