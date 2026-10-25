"""Source-stem historical exclusion and fixed input-only sampling; no API calls."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.build_upgrade_history import build
from experiments.mechanism.prepare_upgrade_pilot import canonical,strings
from experiments.mechanism.prepare_upgrade_p2 import pool,native_mc
from experiments.mechanism.run_upgrade_pilot import freeze

OUT=p.ROOT/'experiments/results/resource_allocation/routing_holdout_data_v1'
AUDIT=p.ROOT/'experiments/results/audit/routing_holdout_2026-10-23'
PINS={
    'strategyqa':'0f72560b8a6fad3db847d2f1a7524fb087297843bb45a8d3245ffd68cbe0de12',
    'truthfulqa':'c90d853454fd48c676accb33e16923ac438dbb6fb37ecc7c6a70e33c9e556e87',
    'mmlu':'74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7'}


def prepare():
    if (OUT/'provenance.json').exists():
        prov=json.loads((OUT/'provenance.json').read_text())
        for name,h in prov['output_sha256'].items():
            if hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=h:raise ValueError('existing data changed')
        print(json.dumps({'existing_frozen_data':True,'n':24}));return prov
    dependency=Path('/private/tmp/mas_p2_dependencies')
    if dependency.is_dir():sys.path.insert(0,str(dependency))
    args=SimpleNamespace(
        strategyqa=p.ROOT/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json',
        truthfulqa=p.ROOT/'experiments/data/raw/upgrade_p2_v1/truthfulqa_mc_authors.json',
        mmlu=p.ROOT/'experiments/data/raw/upgrade_p2_v1/mmlu_test.parquet')
    for domain in PINS:
        if hashlib.sha256(getattr(args,domain).read_bytes()).hexdigest()!=PINS[domain]:raise ValueError('source pin mismatch '+domain)
    # Read native Arrow rows directly; installed pandas/Arrow extension registries differ.
    # Do not patch either runtime or modify the pinned parquet snapshot.
    mmlu_file=args.mmlu;args.mmlu=None
    candidates,sources=pool(args)
    import pyarrow.parquet as parquet
    mmlu_rows=parquet.read_table(mmlu_file).to_pylist();excluded=[]
    if len(mmlu_rows)<1000:raise ValueError('incomplete MMLU source')
    for row in mmlu_rows:
        sid=p.digest([row['subject'],row['question']])[:24]
        if len(set(row['choices']))!=len(row['choices']):
            excluded.append(sid);continue
        candidates.append(native_mc('mmlu',sid,row['question'],list(row['choices']),int(row['answer'])))
    sources.append({'domain':'mmlu','path':str(mmlu_file),'sha256':PINS['mmlu'],'n':len(mmlu_rows),
        'duplicate_option_exclusions':excluded,'reader':'native pyarrow rows, no pandas conversion'})
    history=build(p.ROOT)
    if history['errors']:raise ValueError('unresolved historical parse errors: '+json.dumps(history['errors']))
    AUDIT.mkdir(parents=True,exist_ok=True)
    freeze(AUDIT/'history_snapshot.json',history)
    used={canonical(v['text']) for v in history['texts'].values()}
    ranked=[]
    for domain in ['strategyqa','mmlu','truthfulqa']:
        seen=set();shortlist=[]
        for row in sorted([r for r in candidates if r['domain']==domain],key=lambda r:p.digest(['routing-holdout-2026-10-23-v1',r['source_id']])):
            stem=canonical(row['base_question'])
            if stem in seen or stem in used:continue
            seen.add(stem);shortlist.append(row)
            if len(shortlist)==80:break
        ranked+=shortlist
    blocked={};coverage=[]
    for file in history['files']:
        if file.get('role')=='verified_official_raw_pool_not_model_execution' or file.get('duplicate_of'):continue
        path=p.ROOT/file['path'];raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=file['sha256']:raise ValueError('history changed during screening')
        data=[json.loads(s) for s in raw.decode().splitlines() if s.strip()] if path.suffix=='.jsonl' else json.loads(raw)
        leaves=list(strings(data));ids=set(leaves);content='\n'.join(canonical(s) for s in leaves)
        for row in ranked:
            ident=row['domain']+':'+row['source_id']
            if ident not in blocked and (row['source_id'] in ids or canonical(row['base_question']) in content):blocked[ident]=file['path']
        coverage.append({'path':file['path'],'sha256':file['sha256']})
    selected=[]
    for domain in ['strategyqa','mmlu','truthfulqa']:
        available=[r for r in ranked if r['domain']==domain and r['domain']+':'+r['source_id'] not in blocked]
        if len(available)<8:raise ValueError('not enough screened unused stems '+domain)
        selected+=available[:8]
    if len({canonical(r['base_question']) for r in selected})!=24:raise ValueError('cross-domain duplicate')
    public=[{'id':r['domain']+':'+r['source_id'],'source_id':r['source_id'],'question':r['question'],'choices':r['choices']} for r in selected]
    public.sort(key=lambda r:p.digest(['holdout-execution-order-v1',r['id']]))
    private=[{'id':r['domain']+':'+r['source_id'],'label':r['label']} for r in selected]
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'public_cases.json',public);freeze(OUT/'private_labels.json',private)
    provenance={'version':'routing-holdout-data-v1','n':24,'per_domain':8,'sources':sources,
        'sampling':'input-only SHA ordering, no outcome/risk selection',
        'source_groups':[{'id':r['domain']+':'+r['source_id'],'base_question':r['base_question']} for r in selected],
        'history_files':len(history['files']),'unique_history_texts':history['unique_texts'],
        'backstop_scanned_unique_files':len(coverage),'backstop_coverage_hash':p.digest(coverage),
        'history_snapshot_sha256':hashlib.sha256((AUDIT/'history_snapshot.json').read_bytes()).hexdigest(),
        'candidate_collisions':blocked,'code_hash':p.digest(Path(__file__).read_text()),
        'output_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['public_cases.json','private_labels.json']},
        'limits':['exact normalized source-stem/ID exclusion in indexed experiment files, no paraphrase or model pretraining novelty guarantee',
            'audit-derived files and verified official raw pools excluded as historical API evidence',
            'not a claim about logs outside this repository']}
    freeze(AUDIT/'backstop_coverage.json',coverage);freeze(OUT/'provenance.json',provenance)
    print(json.dumps({'n':24,'history_files':len(history['files']),'backstop_files':len(coverage),'collisions':len(blocked)}))
    return provenance


if __name__=='__main__':prepare()
