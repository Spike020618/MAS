"""Offline source, temporal feedback and resume audit of interaction pilot."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile
from experiments.mechanism import run_targeted_challenge as run
from experiments.mechanism.run_upgrade_pilot import read_rows,freeze
from experiments.core.mechanism import targeted_challenge as method
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def main():
    original=frozen_hashes();out=run.OUT
    protocol=json.loads((out/'protocol.json').read_text())
    for rel,value in protocol['code_sha256'].items():assert run.h(run.ROOT/rel)==value,rel
    assert run.h(run.ROOT/'docs/research/针对性竞争核验_预定方案_2026-10-25.md')==protocol['design_sha256']
    selection=json.loads((out/'selection.json').read_text())
    for name,value in selection['files'].items():assert run.h(out/name)==value,name
    assert run.h(run.previous.ARCHIVE)==selection['archive_sha256']
    cases=json.loads((out/'public_cases.json').read_text());sealed=json.loads((out/'sealed_outputs.json').read_text())
    labels=json.loads((out/'private_labels.json').read_text())
    with zipfile.ZipFile(run.previous.ARCHIVE) as archive:
        originals=[json.loads(line) for line in archive.read('QuALITY.v1.0.1.htmlstripped.dev').decode().splitlines() if line.strip()]
    prior=json.loads((run.previous.OUT/'public_cases.json').read_text())
    assert not {c['article_id'] for c in prior}&{c['article_id'] for c in cases}
    assert not {c['article_sha256'] for c in prior}&{c['article_sha256'] for c in cases}
    for c in cases:
        assert set(c)=={'id','article_id','article','article_sha256','question','choices'}
        matched=[(row,q) for row in originals if str(row['article_id'])==c['article_id']
                 for q in row['questions'] if q['question']==c['question']]
        assert any(row['article']==c['article'] and dict(zip('ABCD',q['options']))==c['choices']
                   and 'ABCD'[int(q['gold_label'])-1]==labels[c['id']] for row,q in matched)
        directory=out/c['id'].replace(':','_');rows=read_rows(directory/'responses.jsonl')
        assert len(rows)==len(read_rows(directory/'attempts.jsonl'))==37
        index={r['key']:r for r in rows};assert len(index)==37
        for candidate in 'ABCD':
            for name in method.STAGES:
                row=index[candidate+'/'+name]
                data=json.loads(row['request']['messages'][1]['content'])
                assert data['registered_candidate']==candidate
                assert set(data)=={'question','choices','registered_candidate','source_windows','history'}
            generic=index[candidate+'/restatement'];challenge=index[candidate+'/challenge'];reply=index[candidate+'/reply']
            assert rows.index(generic)<rows.index(challenge)<rows.index(reply)
            assert 'challenge' not in json.loads(generic['request']['messages'][1]['content'])['history']
            assert 'challenge' in json.loads(reply['request']['messages'][1]['content'])['history']
            assert set(json.loads(challenge['request']['messages'][1]['content'])['history'])=={'author'}
            a=json.loads(index[candidate+'/nonadaptive']['request']['messages'][1]['content'])
            b=json.loads(index[candidate+'/interactive']['request']['messages'][1]['content'])
            assert a['source_windows']==b['source_windows']
            assert a['history']['author']==b['history']['author'] and a['history']['challenge']==b['history']['challenge']
            assert a['history']['reply']==method.message(sealed[c['id']]['claims'][candidate]['restatement'],'restatement')
            assert b['history']['reply']==method.message(sealed[c['id']]['claims'][candidate]['reply'],'reply')
            retrieval=json.loads((directory/(candidate+'_retrieval.json')).read_text())
            for w in retrieval['windows']:assert w['text']==c['article'][w['start']:w['end']]
    names=('test_targeted_challenge','test_document_verification','test_equilibrium_ranking_reference',
           'test_equilibrium_reference_pipeline','test_learned_production_game','test_learned_production_runner')
    with (patch('socket.socket.connect',side_effect=AssertionError('offline')),
          patch('socket.getaddrinfo',side_effect=AssertionError('offline')),
          patch.object(run,'transport',side_effect=AssertionError('no API on replay'))):
        with patch('builtins.print'):
            run.run()
            from experiments.mechanism.diagnose_targeted_challenge import main as diagnose
            diagnose()
        stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.'+name for name in names]))
        assert tests.wasSuccessful(),stream.getvalue()
    assert frozen_hashes()==original
    previous=run.previous.OUT/'acceptance.json'
    for rel,value in json.loads(previous.read_text())['sha256'].items():assert run.h(run.ROOT/rel)==value,rel
    result=json.loads((out/'result.json').read_text());assert result['physical_calls']==222
    paths=[p for p in out.rglob('*') if p.is_file() and p.name not in ('run.lock','acceptance.json')]
    paths += [Path(__file__),Path(run.__file__),Path(method.__file__),
        run.ROOT/'experiments/tests/test_targeted_challenge.py',
        run.ROOT/'experiments/mechanism/diagnose_targeted_challenge.py',
        run.ROOT/'docs/research/针对性竞争核验_预定方案_2026-10-25.md',
        run.ROOT/'docs/research/针对性竞争核验_结果_2026-10-25.md']
    accepted={'status':'targeted_challenge_pilot_completed_not_equilibrium_proof','tests':tests.testsRun,
        'original_frozen_files':len(original),'physical_calls':222,'physical_tokens':result['physical_tokens'],
        'verification_new_calls':0,'scope':'six documents; all24 candidate claims retained; not formal independent efficacy',
        'sha256':{str(p.relative_to(run.ROOT)):run.h(p) for p in paths}}
    freeze(out/'acceptance.json',accepted)
    print(json.dumps({k:v for k,v in accepted.items() if k!='sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
