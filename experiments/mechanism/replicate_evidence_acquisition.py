"""16 new questions, two solver repeats: factual acquisition baseline only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from experiments.core.mechanism.evidence_acquisition import Corpus, Ledger
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism import repeat_open_evidence_solver as repeat
from experiments.mechanism import probe_acquired_evidence_pool as pool
from experiments.mechanism.build_upgrade_history import build, normalize
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows

ROOT = p.ROOT
OUT = ROOT / 'experiments/results/redesign_v1/acquisition_replication_v1'
DESIGN = ROOT / 'docs/research/redesign_v1/追加取证_独立开发复核_2026-10-25.md'


def canonical(documents):
    docs = {}; origins = {}
    for d in documents:
        if sha256(d['text'].encode()).hexdigest() != d['text_sha256']: raise ValueError('corrupt source')
        identity = sha256(json.dumps([d['title'], d['text']], ensure_ascii=False).encode()).hexdigest()
        key = 'source:' + identity
        docs[key] = {**{k:d[k] for k in ('title', 'text', 'text_sha256')}, 'docid': key}
        origins.setdefault(key, set()).add(d['docid'])
    ordered = [docs[k] for k in sorted(docs)]
    if sum(len(d['text'].encode()) for d in ordered) > 40000: raise ValueError('whole source pool over cap')
    return ordered, {k: sorted(v) for k,v in origins.items()}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists(): verify(); return
    p.verify(); pool.verify()
    history = build(ROOT)
    if history['errors']: raise ValueError('history parse failure')
    used = {v['text'] for v in history['texts'].values()}
    rows = p.data.parquet_module().read_table(p.data.RAW / 'dev_fullwiki.parquet', columns=['id', 'question', 'context']).to_pylist()
    rows.sort(key=lambda c: sha256(('acquisition-independent-v1/' + c['id']).encode()).hexdigest())
    selected = [c for c in rows if normalize(c['question']) not in used][:16]
    if len(selected) != 16: raise ValueError('not enough new questions')
    cases = []
    for c in selected:
        docs = []
        for title, sents in zip(c['context']['title'], c['context']['sentences']):
            text = ''.join(sents); docs.append({'docid': 'native:' + sha256(title.encode()).hexdigest(),
                'title': title, 'text': text, 'text_sha256': sha256(text.encode()).hexdigest()})
        cases.append({'id': c['id'], 'question': c['question'], 'native_documents': docs})
    freeze(OUT / 'public_cases.json', cases)
    ids = {c['id'] for c in cases}
    labels = {c['id']:c['answer'] for c in p.data.parquet_module().read_table(p.data.RAW / 'dev_fullwiki.parquet', columns=['id','answer']).to_pylist() if c['id'] in ids}
    freeze(OUT / 'private_labels.json', labels)
    freeze(OUT / 'selection.json', {'history_files': len(history['files']), 'normalized_prior_overlap': 0,
        'not_semantic_or_pretraining_novelty': True, 'not_source_group_isolated': True,
        'history_sha256': {v['path']:v['sha256'] for v in history['files']}})
    files = [Path(__file__), DESIGN, ROOT / 'experiments/core/mechanism/evidence_acquisition.py',
        ROOT / 'experiments/mechanism/run_open_evidence_screen_v2.py',
        ROOT / 'experiments/tests/test_acquisition_replication.py',
        OUT / 'public_cases.json', OUT / 'private_labels.json', OUT / 'selection.json']
    freeze(OUT / 'protocol.json', {'version': 'independent-acquisition-v1', 'n':16, 'max_calls':96,
        'repeats':2, 'cumulative_cap_usd_proxy':5, 'model':'deepseek-flash', 'fp':p.base.FP,
        'design_status':'baseline, not a trained game or formal held-out test',
        'sha256': {str(f.relative_to(ROOT)):p.base.h(f) for f in files},
        'index_manifest_sha256':p.base.h(p.data.ENV / 'index_manifest.json')})


def verify():
    p.verify(); protocol=p.base.read(OUT / 'protocol.json')
    for path, expected in protocol['sha256'].items():
        if p.base.h(ROOT/path)!=expected: raise ValueError('replication frozen input changed:'+path)
    if p.base.h(p.data.ENV/'index_manifest.json')!=protocol['index_manifest_sha256']: raise ValueError('corpus changed')


def collect(c, transport):
    folder=OUT/c['id']
    with lock(folder):
        journal=CallJournal(folder,transport,6)
        if {a['key'] for a in read_rows(folder/'attempts.jsonl')}-{r['key'] for r in read_rows(folder/'responses.jsonl')}:
            raise RuntimeError('unresolved attempt, no automatic retry')
        def call(key, req, docs, stage):
            rows=read_rows(folder/'responses.jsonl')
            if key not in {r['key'] for r in rows}:
                paths=list(p.OUT.rglob('responses.jsonl'))+[repeat.OUT/'responses.jsonl',pool.OUT/'responses.jsonl']+list(OUT.rglob('responses.jsonl'))
                raws=[r for path in paths for r in read_rows(path)]
                spent=sum((json.loads(r['response']['body'])['usage']['prompt_tokens']*.3+json.loads(r['response']['body'])['usage']['completion_tokens']*1.2)/1e6 for r in raws)
                reserve=((len(p.base.encoded(req))+4096)*.3+req['max_tokens']*1.2)/1e6
                if spent+reserve>5: raise RuntimeError('cumulative cap exhausted')
            return p.parse(journal.call(key,req),stage,docs)
        planned=call('plan',p.request(c,'plan'),[],'plan')
        if not planned['valid']: raise RuntimeError('invalid query plan; no repair')
        queries=[c['question']]+planned['queries']
        environment_path=folder/'query_environments.json'
        corpus=Corpus(p.data.ENV/'wiki_fts.sqlite')
        try:
            if environment_path.exists(): environments=p.base.read(environment_path)
            else:
                environments=[]
                for q in queries:
                    ledger=Ledger(max_searches=1,max_unique_bytes=40000);ledger.acquire(corpus,q)
                    environments.append({'query':q,'documents':ledger.public_documents(),'searches':ledger.searches})
                freeze(environment_path,environments)
            initial,_=canonical(environments[0]['documents'])
            seq=call('seq_plan',p.request(c,'seq_plan',initial),initial,'seq_plan')
            if not seq['valid']: raise RuntimeError('invalid sequential query; no repair')
            next_path=folder/'next_environment.json'
            if next_path.exists(): next_env=p.base.read(next_path)
            else:
                ledger=Ledger(max_searches=1,max_unique_bytes=40000);ledger.acquire(corpus,seq['query'])
                next_env={'query':seq['query'],'documents':ledger.public_documents(),'searches':ledger.searches}
                freeze(next_path,next_env)
        finally: corpus.close()
        native,native_map=canonical(c['native_documents'])
        expanded,expanded_map=canonical(c['native_documents']+[d for e in environments for d in e['documents']]+next_env['documents'])
        freeze(folder/'material.json',{'native':native,'expanded':expanded,'native_origins':native_map,'expanded_origins':expanded_map})
        outputs={}
        for rep in range(2):
            outputs[str(rep)]={}
            arms=sorted(('native','expanded'),key=lambda a:sha256((c['id']+'/'+str(rep)+'/'+a).encode()).hexdigest())
            reused={}
            for arm in arms:
                docs=native if arm=='native' else expanded
                req=p.request(c,'native',docs);signature=sha256(p.base.encoded(req)).hexdigest()
                if signature in reused: value=reused[signature]
                else:
                    value=call(str(rep)+'/'+arm,req,docs,'native');reused[signature]=value
                outputs[str(rep)][arm]=value
        freeze(folder/'sealed_outputs.json',outputs)
        return outputs


def run(transport):
    verify(); outputs={}
    with lock(OUT):
        for c in p.base.read(OUT/'public_cases.json'):
            outputs[c['id']]=collect(c,transport);print('independent acquisition sealed',len(outputs),'/16',flush=True)
        freeze(OUT/'sealed_outputs.json',outputs)


def analyze():
    verify();outputs=p.base.read(OUT/'sealed_outputs.json');labels=p.base.read(OUT/'private_labels.json')
    if set(outputs)!=set(labels) or len(outputs)!=16: raise ValueError('incomplete seal; no scoring')
    metrics={}
    for rep in ('0','1'):
        cell={}
        for arm in ('native','expanded'):
            scores=[p.answer_score(outputs[c][rep][arm]['answer'],labels[c]) for c in labels]
            cell[arm]={'em_correct':sum(v['em'] for v in scores),'mean_f1':sum(v['f1'] for v in scores)/16,
                'invalid':sum(not outputs[c][rep][arm]['valid'] for c in labels)}
        cell['corrected']=[c for c in labels if p.answer_score(outputs[c][rep]['expanded']['answer'],labels[c])['em'] and not p.answer_score(outputs[c][rep]['native']['answer'],labels[c])['em']]
        cell['harmed']=[c for c in labels if not p.answer_score(outputs[c][rep]['expanded']['answer'],labels[c])['em'] and p.answer_score(outputs[c][rep]['native']['answer'],labels[c])['em']]
        metrics[rep]=cell
    raw=[r for path in OUT.rglob('responses.jsonl') for r in read_rows(path)]
    result={'n_independent_questions':16,'solver_repeats_not_extra_samples':2,'metrics':metrics,'physical_calls':len(raw),
        'physical_tokens':sum(json.loads(r['response']['body'])['usage']['total_tokens'] for r in raw),
        'sum_http_seconds':sum(r['wall_seconds'] for r in raw),
        'usd_proxy':sum((json.loads(r['response']['body'])['usage']['prompt_tokens']*.3+json.loads(r['response']['body'])['usage']['completion_tokens']*1.2)/1e6 for r in raw),
        'not_game_gain':True,'not_same_total_cost':True}
    freeze(OUT/'result.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','analyze','replay'));stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='analyze':analyze()
    elif stage=='replay':
        def blocked(req):raise AssertionError('replay API')
        run(blocked);analyze()
    else:
        from experiments.mechanism.study_explicit_thinking_control import transport
        run(transport);analyze()
