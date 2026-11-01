"""Date-independent offline replay/verification without mutable invocation logs."""
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from experiments.mechanism import run_sequential_allocation_screen as r
from experiments.mechanism import correct_sequential_allocation_costs as correction
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, read_rows, digest
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def main():
    before=frozen_hashes();r.verify();correction.verify()
    cases=r.p.base.read(r.OUT/'public_cases.json');outputs=r.p.base.read(r.OUT/'sealed_outputs.json')
    assert len(cases)==len(outputs)==40
    old=r.ROOT/'experiments/results/redesign_v1/hour_acceptance_v1/acceptance.json'
    old_manifest=r.p.base.read(old)
    assert all(r.p.base.h(r.ROOT/name)==value for name,value in old_manifest['sha256'].items())
    source={x['id']:x for x in r.p.data.parquet_module().read_table(r.p.data.RAW/'dev_fullwiki.parquet',columns=['id','question','context','answer']).to_pylist()}
    labels=r.p.base.read(r.OUT/'private_labels.json');all_raw=[];wanted={};row_count=0
    def blocked(req):raise AssertionError('verification attempted new API')
    with patch('socket.socket.connect',side_effect=AssertionError('offline')),patch('socket.getaddrinfo',side_effect=AssertionError('offline')):
        stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames([
            'experiments.tests.test_sequential_allocation_screen','experiments.tests.test_evidence_acquisition',
            'experiments.tests.test_counterfactual_credit_audit','experiments.tests.test_acquisition_replication',
            'experiments.tests.test_answer_order_control','experiments.tests.test_evidence_thinking_control',
            'experiments.tests.test_sequential_cost_correction']))
        assert tests.wasSuccessful(),stream.getvalue()
        for case in cases:
            cid=case['id'];folder=r.OUT/cid;states=outputs[cid];assert len(states)==3
            assert r.p.base.read(folder/'sealed_outputs.json')==states
            s=source[cid];assert case['question']==s['question'] and labels[cid]==s['answer']
            assert [d['title'] for d in case['native_documents']]==s['context']['title']
            assert [d['text'] for d in case['native_documents']]==[''.join(v) for v in s['context']['sentences']]
            rows=read_rows(folder/'responses.jsonl');attempts=read_rows(folder/'attempts.jsonl')
            assert len(rows)==len(attempts)<=8
            assert {x['key'] for x in rows}=={x['key'] for x in attempts}
            by_key={x['key']:x for x in rows};journal=CallJournal(folder,blocked,8)
            previous,_=r.canonical_source.canonical(case['native_documents'])
            for stage,state in enumerate(states):
                assert state['stage']==stage
                if stage:
                    plan=state['query_plan'];request=r.p.request(case,'seq_plan',previous)
                    row=journal.call(plan['source_key'],request)
                    assert r.p.parse(row,'seq_plan',previous)==plan['parsed']
                    env=r.p.base.read(folder/('search'+str(stage)+'.json'))
                    assert env==state['retrieval'] and env['query']==plan['parsed']['query']
                    assert not env['event']['skipped_budget_ids']
                    for d in env['documents']:wanted[d['docid']]=d
                    material,_=r.canonical_source.canonical(previous+env['documents'])
                    assert state['new_documents']==len({d['docid'] for d in material}-{d['docid'] for d in previous})
                else:material=previous
                assert state['documents']==material
                for mode in r.MODES:
                    answer=state['answers'][mode];request=r.final_request(case,material,mode)
                    row=journal.call(answer['source_key'],request)
                    assert r.p.parse(row,'native',material)==answer['parsed']
                previous=material
            for row in rows:
                assert digest(row['request'])==row['request_hash'] and digest(row['response'])==row['response_hash']
                body=json.loads(row['response']['body']);assert row['response']['http_status']==200
                assert (body['model'],body['system_fingerprint'])==('deepseek-flash',r.p.base.FP)
                payload=json.loads(row['request']['messages'][1]['content'])
                assert set(payload)=={'question','documents'} and payload['question']==case['question']
            all_raw+=rows;row_count+=len(rows)
        # Validate every returned original source, not merely citation syntax.
        found=set()
        for batch in r.p.data.parquet_module().ParquetFile(r.p.data.RAW/'corpus.parquet').iter_batches(batch_size=20000):
            for doc in batch.to_pylist():
                key=str(doc['_id'])
                if key in wanted:
                    assert (doc['title'],doc['text'])==(wanted[key]['title'],wanted[key]['text']);found.add(key)
        assert found==set(wanted)
        with patch('builtins.print'):r.analyze();correction.analyze()
    assert before==frozen_hashes();assert row_count<=320
    subprocess.run(['git','diff','--check'],cwd=r.ROOT,check=True,capture_output=True)
    for f in (Path(r.__file__),Path(__file__),r.DESIGN,r.ROOT/'experiments/tests/test_sequential_allocation_screen.py'):
        checked=subprocess.run(['git','diff','--no-index','--check','--','/dev/null',str(f)],cwd=r.ROOT,capture_output=True,text=True)
        assert checked.returncode in (0,1) and not checked.stdout and not checked.stderr
    result=r.p.base.read(r.OUT/'corrected_analysis.json')
    splits=r.p.base.read(r.OUT/'splits.json')
    for mode in r.MODES:
        for split,cell in result['cells'][mode]['splits'].items():
            ids=[c for c in outputs if split=='all' or splits[c]==split]
            for name,policy in cell['simple_policies'].items():
                costs=[]
                for cid,stage in zip(ids,policy['stages']):
                    rows={x['key']:x for x in read_rows(r.OUT/cid/'responses.jsonl')}
                    costs.append(r.path_account(outputs[cid],rows,stage,mode,name=='abstain_only'))
                assert policy['tokens']==sum(c['tokens'] for c in costs)
                assert policy['calls']==sum(c['calls'] for c in costs)
    accepted={'status':'completed_development_screen_NOT_game_training','n':40,'tests':tests.testsRun,
        'new_verification_calls':0,'physical_calls':row_count,'physical_tokens':result['physical_tokens'],
        'usd_proxy':sum(r.usd(x) for x in all_raw),'legacy_frozen_files':len(before),
        'previous_acceptance_artifacts_verified':len(old_manifest['sha256']),
        'source_documents_verified':len(wanted),'gate':result['gate'],
        'sha256':{str(f.relative_to(r.ROOT)):r.p.base.h(f) for f in sorted(r.OUT.rglob('*'))
            if f.is_file() and f.suffix in ('.json','.jsonl') and f.name not in ('acceptance.json','invocations.jsonl')}}
    accepted['sha256'][str(Path(__file__).relative_to(r.ROOT))]=r.p.base.h(Path(__file__))
    freeze(r.OUT/'acceptance.json',accepted)
    print(json.dumps({k:v for k,v in accepted.items() if k!='sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
