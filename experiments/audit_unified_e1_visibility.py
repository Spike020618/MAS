"""Independent read-only E1 input/source visibility audit. Never reads labels/API."""
import argparse,json,hashlib
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,sha,fingerprint,create


def load(path):return json.loads(Path(path).read_text())


def audit(root):
    root=Path(root);plan=load(root/'plan.json');tasks=load(root/'inputs.json');byid={t['id']:t for t in tasks}
    audit_path=root/'independent_input_visibility_audit.prelabel.json'
    prior=load(audit_path) if audit_path.exists() else None
    if prior is not None:
        assert prior['plan_sha256']==sha(root/'plan.json') and prior['bounded_inputs_sha256']==sha(root/'inputs.json')
    # Repeat audits verify the original captured partial inventory, not a moving stream.
    frozen_slots={r['logical_slot']for r in prior['requests']} if prior else None
    assert len(tasks)==24 and sum(t['kind']=='qa'for t in tasks)==12 and sum(t['kind']=='rec'for t in tasks)==12
    assert [t['id']for t in tasks]==plan['task_ids'] and fingerprint(tasks)==plan['bounded_tasks_hash']
    assert sha(plan['input_source'])==plan['input_sha256']
    hashes={}
    for name,expected in plan['source_sha256'].items():
        snap=root/'source_snapshot'/name
        assert sha(snap)==expected, name
        hashes[name]={'snapshot_matches_plan':True,'live_matches_plan':sha(ROOT/name)==expected}
    qa_banks=0;web_excerpts=0;source_count=0
    native_questions=load(ROOT/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/cases.json')
    questions={c['question_id']:c['question']for c in native_questions}
    forbidden={'answer','gold','label','correct_answer','supporting_facts','decomposition'}
    for t in tasks:
        assert t['metadata']['query_labels_available']is False
        assert not (forbidden&set(t))
        if t['kind']=='qa':
            assert t['options']==['yes','no'] and t['initial']=='abstain'
            assert t['metadata']['native_dataset']=='StrategyQA'
            assert t['prompt']==questions[t['metadata']['native_id']]
            path=Path(t['metadata']['source_bank']);assert sha(path)==t['metadata']['source_bank_sha256'];bank=load(path);qa_banks+=1
            assert not (forbidden&set(bank))
            for s in bank['sources']:
                assert not (forbidden&set(s))
                assert hashlib.sha256(s['text'].encode()).hexdigest()==s['excerpt_sha256']
                snapshot=path.parent.parent/'snapshots'/(s['source_hash']+'.html')
                assert sha(snapshot)==s['source_hash'];web_excerpts+=1
        for sid,s in t['source_catalog'].items():
            assert s['source_id']==sid and s['scope']['task_id']==t['id']
            assert s['semantic_truth_certified']is False
            assert len(s['text'])<=plan['max_source_chars'];source_count+=1
    checked=[]
    for p in sorted((root/'logical_calls').glob('*.json')):
        if frozen_slots is not None and p.stem not in frozen_slots:continue
        rec=load(p);req=rec['request'];payload=json.loads(req['messages'][1]['content']);t=byid[payload['task_id']]
        assert req['kind']in ('unified_e1_initial','unified_e1_review')
        assert payload['question']==t['prompt'] and payload['candidates']==t['candidates']
        assert payload['options']==t['options']
        offered=payload['available_source_catalog']
        assert {s['source_id']for s in offered}==set(t['source_catalog'])
        assert all(set(s)=={'source_id','kind','scope'}for s in offered)
        for sid,s in payload['fetched_sources'].items():
            assert sid in t['source_catalog']
            expected={k:v for k,v in t['source_catalog'][sid].items()if k in ('source_id','kind','scope','text','allowed_uses','semantic_truth_certified','truncated')}
            assert s==expected
        if payload['phase']=='initial':
            assert len(payload['fetched_sources'])==(0 if t['kind']=='qa'else 1)
        checked.append({'logical_slot':p.stem,'task_id':t['id'],'phase':payload['phase'],'visible_source_ids':list(payload['fetched_sources'])})
    method_sources={}
    for p in sorted((root/'cases').glob('case_*.json')):
        c=load(p)
        if prior is not None and c['task_id'] not in prior['completed_case_method_resource_inventory']:continue
        method_sources[c['task_id']]={m:{'visible_sources':len(a['state']['visible_sources']),
            'fetch_actions':sum('source_action'in e for e in a['events']), 'logical_calls':a['logical_calls']}
            for m,a in c['methods'].items()}
        assert all('source_action'not in e for e in c['methods']['independent']['events'])
    result={'scope':'read-only input and source-visibility audit during E1 generation; no labels or scores read',
        'plan_sha256':sha(root/'plan.json'),'bounded_inputs_sha256':sha(root/'inputs.json'),
        'tasks':len(tasks),'qa_native_dataset':'StrategyQA only','qa_tasks':12,'rec_tasks':12,
        'source_count':source_count,'qa_banks_verified':qa_banks,'external_snapshot_excerpts_verified':web_excerpts,
        'source_hashes':hashes,'logical_requests_checked':len(checked),'requests':checked,
        'completed_case_method_resource_inventory':method_sources,
        'labels_read':False,'generation_complete_at_audit':prior['generation_complete_at_audit'] if prior else (root/'generation_complete.json').exists(),
        'limits':['All tasks are reused development engineering cases, not independent confirmation.',
                  'QA is StrategyQA yes/no only, not three QA benchmarks or native MMLU validation.',
                  'All arms can access same catalog, but independent arm does not fetch additional sources or review; actual information processed and calls differ.',
                  'E1 compares complete paths; changes cannot be attributed separately to AEIC, source access, game allocation, or extra computation.',
                  'Game/central use homogeneous heuristic source scores, not calibrated task contribution or verified reviewer skill.',
                  'Cached external excerpts are historical retrieval snapshots, not freshly checked world facts or guaranteed gold supporting evidence.',
                  'Hash/field checks support absence of explicit gold-label plumbing; they do not prove all historical selection was uncontaminated or model pretraining unseen.',
                  'Source presence/authority checks are not semantic-truth certification.']}
    create(root/'independent_input_visibility_audit.prelabel.json',result)
    print(json.dumps({k:result[k]for k in ('tasks','source_count','qa_banks_verified','external_snapshot_excerpts_verified','logical_requests_checked','labels_read','generation_complete_at_audit')},ensure_ascii=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/results/unified_e1_v3');a=p.parse_args();audit(a.output)
