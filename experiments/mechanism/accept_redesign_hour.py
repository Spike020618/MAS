"""Offline acceptance of this research hour; never makes a model request."""
from collections import Counter
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism import repeat_open_evidence_solver as repeat
from experiments.mechanism import probe_acquired_evidence_pool as pool
from experiments.mechanism import replicate_evidence_acquisition as r
from experiments.mechanism import attribute_acquisition_paths as paths
from experiments.mechanism import control_answer_order as order
from experiments.mechanism import control_evidence_thinking as thinking
from experiments.mechanism import audit_open_retrieval_coverage as coverage
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.run_upgrade_pilot import freeze,read_rows,digest

ROOT=p.ROOT;OUT=ROOT/'experiments/results/redesign_v1/hour_acceptance_v1'


def main():
    before=frozen_hashes();roots=[p.OUT,repeat.OUT,pool.OUT,r.OUT,paths.OUT,order.OUT,thinking.OUT]
    modules=[p,repeat,pool,r,paths,order,thinking,coverage]
    for module in modules:module.verify()
    names=('test_open_evidence_screen_v2','test_evidence_acquisition','test_counterfactual_credit_audit',
        'test_sampling_coordination_screen','test_bandit_policy_learning','test_action_contribution_frontier',
        'test_option_binding_check','test_full_context_challenge','test_targeted_challenge_preflight',
        'test_targeted_challenge','test_document_verification','test_equilibrium_ranking_reference',
        'test_equilibrium_reference_pipeline','test_learned_production_game','test_learned_production_runner',
        'test_open_evidence_screen','test_repeat_open_evidence_solver','test_title_anchor_retrieval',
        'test_evidence_payoff_landscape','test_acquired_evidence_pool','test_acquisition_replication',
        'test_observed_evidence_shapley','test_acquisition_path_attribution','test_answer_order_control',
        'test_evidence_thinking_control')
    def blocked(req):raise AssertionError('acceptance attempted new API')
    with patch('socket.socket.connect',side_effect=AssertionError('offline')),patch('socket.getaddrinfo',side_effect=AssertionError('offline')):
        stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(['experiments.tests.'+n for n in names]))
        assert tests.wasSuccessful(),stream.getvalue()
        with patch('builtins.print'):
            r.run(blocked);r.analyze();paths.collect(blocked);paths.analyze()
            order.collect(blocked);order.analyze();thinking.collect(blocked);thinking.analyze()
            repeat.collect(blocked);repeat.analyze();pool.collect(blocked);pool.analyze()
    raw=[];batches={};models=Counter();fps=Counter();wanted={}
    for root in roots:
        rows=[x for f in root.rglob('responses.jsonl') for x in read_rows(f)]
        for f in root.rglob('attempts.jsonl'):
            attempts=read_rows(f);responses=read_rows(f.with_name('responses.jsonl'))
            assert len(attempts)==len(responses)
            assert {x['key'] for x in attempts}=={x['key'] for x in responses}
        for row in rows:
            assert digest(row['request'])==row['request_hash'] and digest(row['response'])==row['response_hash']
            body=json.loads(row['response']['body']);assert row['response']['http_status']==200
            models[body['model']]+=1;fps[body['system_fingerprint']]+=1
            usage=body['usage'];assert usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens']
            value=json.loads(row['request']['messages'][1]['content'])
            assert set(value)<={'question','documents'} and isinstance(value['question'],str)
            for d in value.get('documents',[]):assert set(d)=={'docid','title','text','text_sha256'}
        batches[root.name]={'calls':len(rows),'tokens':sum(json.loads(x['response']['body'])['usage']['total_tokens'] for x in rows),
            'sum_http_seconds':sum(x['wall_seconds'] for x in rows),
            'usd_proxy':sum((json.loads(x['response']['body'])['usage']['prompt_tokens']*.3+json.loads(x['response']['body'])['usage']['completion_tokens']*1.2)/1e6 for x in rows)}
        raw+=rows
    assert set(models)=={'deepseek-flash'} and set(fps)=={p.base.FP}
    # Verify new inputs directly against raw dev and all returned corpus sources.
    source={x['id']:x for x in p.data.parquet_module().read_table(p.data.RAW/'dev_fullwiki.parquet',columns=['id','question','context','answer']).to_pylist()}
    old_ids={c['id'] for c in p.base.read(p.OUT/'public_cases.json')}
    coverage_ids={c['id'] for c in p.base.read(coverage.OUT/'public_cases.json')}
    cases=p.base.read(r.OUT/'public_cases.json');new_ids={c['id'] for c in cases}
    assert len(new_ids)==16 and not(new_ids&old_ids) and not(new_ids&coverage_ids)
    labels=p.base.read(r.OUT/'private_labels.json')
    for c in cases:
        s=source[c['id']];assert c['question']==s['question'] and labels[c['id']]==s['answer']
        assert [d['text'] for d in c['native_documents']]==[''.join(t) for t in s['context']['sentences']]
        assert [d['title'] for d in c['native_documents']]==s['context']['title']
        envs=p.base.read(r.OUT/c['id']/'query_environments.json')+[p.base.read(r.OUT/c['id']/'next_environment.json')]
        for env in envs:
            for d in env['documents']:wanted[d['docid']]=d
    found=set()
    for batch in p.data.parquet_module().ParquetFile(p.data.RAW/'corpus.parquet').iter_batches(batch_size=20000):
        for d in batch.to_pylist():
            key=str(d['_id'])
            if key in wanted:
                assert (d['title'],d['text'])==(wanted[key]['title'],wanted[key]['text']);found.add(key)
    assert found==set(wanted)
    assert before==frozen_hashes()
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True,capture_output=True)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    frozen_format_warnings=[]
    for name in untracked:
        file=ROOT/name
        if file.suffix not in ('.py','.md','.json'):continue
        result=subprocess.run(['git','diff','--no-index','--check','--','/dev/null',str(file)],cwd=ROOT,capture_output=True,text=True)
        expected_warning=str(file)+':96: new blank line at EOF.\n'
        if name=='experiments/tests/test_open_evidence_screen_v2.py' and result.stdout==expected_warning and not result.stderr:
            # This byte sequence was frozen before collection. Preserve it;
            # never "fix" a preregistered source to conceal a formatting note.
            assert p.base.h(file)==p.verify()['code_sha256'][name]
            frozen_format_warnings.append({'file':name,'warning':'new blank line at EOF, pre-collection frozen hash unchanged'})
        else:
            assert result.returncode in (0,1) and not result.stdout and not result.stderr,(name,result.stdout,result.stderr)
    outputs={name:p.base.read(ROOT/'experiments/results/redesign_v1'/name/'result.json') for name in
        ('open_evidence_screen_v2','evaluator_replication_v1','acquired_evidence_pool_v1','acquisition_replication_v1',
         'acquisition_path_attribution_v1','answer_order_control_v1','evidence_thinking_control_v1','retrieval_coverage_v1',
         'screen_validity_v1','index_parity_v1')}
    costs=sum(v['usd_proxy'] for v in batches.values());assert costs<5
    summary={'status':'research_hour_verified_NOT_new_game_contribution','tests':tests.testsRun,'verification_api_calls':0,
        'legacy_frozen_files':len(before),'models':dict(models),'fingerprints':dict(fps),'physical_calls':len(raw),
        'physical_tokens':sum(v['tokens'] for v in batches.values()),'usd_proxy_total':costs,
        'sum_http_seconds':sum(v['sum_http_seconds'] for v in batches.values()),'batches':batches,
        'n_solver_questions':24,'n_retrieval_only_questions':128,'replication_questions':16,
        'extra_repeats_are_not_extra_independent_questions':True,'unresolved_attempts':0,
        'tracked_git_diff_check':'passed','frozen_format_warnings':frozen_format_warnings,
        'corpus_rows_parity_verified':outputs['index_parity_v1']['rows'],
        'main_replication':outputs['acquisition_replication_v1']['metrics'],
        'format_control':outputs['answer_order_control_v1']['metrics'],
        'thinking_control':outputs['evidence_thinking_control_v1']['metrics'],
        'path_metrics':outputs['acquisition_path_attribution_v1']['metrics_by_bitmask_ABC'],
        'claim_limits':['development evidence, not a formal held-out study','not equal total cost',
            'no trained game, no equilibrium or new mechanism performance claim',
            'sole extra EM point over sequential path is left hand versus left; not a new factual correction'],
        'result_sha256':{str((ROOT/'experiments/results/redesign_v1'/name/'result.json').relative_to(ROOT)):p.base.h(ROOT/'experiments/results/redesign_v1'/name/'result.json') for name in outputs}}
    OUT.mkdir(parents=True,exist_ok=True);freeze(OUT/'summary.json',summary)
    freeze(ROOT/'docs/research/redesign_v1/一小时验收_2026-10-25.json',summary)
    files=[f for f in (ROOT/'experiments/results/redesign_v1').rglob('*') if f.is_file() and f.suffix in ('.json','.jsonl') and f.parent!=OUT]
    files+=[ROOT/n for n in untracked if (ROOT/n).is_file() and (ROOT/n).suffix in ('.py','.md')]
    files+=[ROOT/'docs/research/ACTIVE_RESEARCH_PLAN.md',ROOT/'docs/research/redesign_v1/一小时验收_2026-10-25.json',Path(__file__)]
    freeze(OUT/'acceptance.json',{'summary_sha256':p.base.h(OUT/'summary.json'),
        'sha256':{str(f.relative_to(ROOT)):p.base.h(f) for f in sorted(set(files))}})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('result_sha256','main_replication','format_control','thinking_control','path_metrics')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
