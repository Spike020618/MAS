"""Six new source documents, all four candidate claims, bounded interaction.

One adaptive response is not trained best response, a Stackelberg solution,
or proof of rationality. This is a protocol feasibility/value screen only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
import zipfile
from experiments.core.mechanism import targeted_challenge as method
from experiments.core.mechanism.document_verification import retrieve
from experiments.mechanism import run_document_verification as previous
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows
from experiments.mechanism.study_explicit_thinking_control import transport

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/resource_allocation/targeted_challenge_v1'
STOP=Event()


def h(path):return sha256(path.read_bytes()).hexdigest()


def prepare():
    with lock(OUT):
        if (OUT/'selection.json').exists():
            selection=json.loads((OUT/'selection.json').read_text())
            assert h(previous.ARCHIVE)==selection['archive_sha256']
            for name,value in selection['files'].items():assert h(OUT/name)==value
            return
        old=json.loads((previous.OUT/'public_cases.json').read_text())
        used={c['article_id'] for c in old};hashes={c['article_sha256'] for c in old}
        with zipfile.ZipFile(previous.ARCHIVE) as z:
            member='QuALITY.v1.0.1.htmlstripped.dev'
            rows=[json.loads(line) for line in z.read(member).decode().splitlines() if line.strip()]
        ranked=[]
        for row in rows:
            if str(row['article_id']) in used or sha256(row['article'].encode()).hexdigest() in hashes:continue
            for q in row['questions']:
                if len(q['options'])!=4 or len(set(q['options']))!=4:continue
                ranked.append((sha256(('targeted-challenge-v1/'+str(row['article_id'])+'/'+q['question']).encode()).hexdigest(),row,q))
        texts=[];journal_hashes={}
        for path in (ROOT/'experiments/results').rglob('responses.jsonl'):
            if OUT in path.parents:continue
            texts.append(path.read_text());journal_hashes[str(path.relative_to(ROOT))]=h(path)
        selected=[];selected_docs=set();selected_hashes=set()
        for ident,row,q in sorted(ranked,key=lambda x:x[0]):
            doc=str(row['article_id']);dh=sha256(row['article'].encode()).hexdigest()
            if doc in selected_docs or dh in selected_hashes:continue
            if any(q['question'] in text or json.dumps(q['question'],ensure_ascii=False)[1:-1] in text for text in texts):continue
            selected.append((ident,row,q));selected_docs.add(doc);selected_hashes.add(dh)
            if len(selected)==6:break
        assert len(selected)==6
        public=[{'id':'quality:'+ident[:24],'article_id':str(row['article_id']),'article':row['article'],
            'article_sha256':sha256(row['article'].encode()).hexdigest(),'question':q['question'],
            'choices':dict(zip('ABCD',q['options']))} for ident,row,q in selected]
        labels={c['id']:'ABCD'[int(q['gold_label'])-1] for c,(_,row,q) in zip(public,selected)}
        freeze(OUT/'public_cases.json',public);freeze(OUT/'private_labels.json',labels)
        freeze(OUT/'source_metadata.json',[{'id':c['id'],'url':row.get('url'),'license':row.get('license')}
            for c,(_,row,q) in zip(public,selected)])
        freeze(OUT/'selection.json',{'n':6,'candidate_claims':24,'source':'official QuALITY v1.0.1 dev',
            'archive_sha256':h(previous.ARCHIVE),'excluded_document_ids':sorted(used),'journal_sha256':journal_hashes,
            'sampling':'fixed input hash; new documents; all four options, not gold-selected claims; no hard-only filter',
            'limits':'developmental engineering pilot; exact source/document/request exclusion, not semantic or pretraining novelty',
            'files':{name:h(OUT/name) for name in ('public_cases.json','private_labels.json','source_metadata.json')}})
        print('prepared6 new documents,24 claims,222 calls maximum',flush=True)


def collect(case):
    directory=OUT/case['id'].replace(':','_')
    with lock(directory):
        journal=CallJournal(directory,transport,37)
        full=[{'id':'article','start':0,'end':len(case['article']),'text':case['article']}]
        def call(key,req,stage,windows):
            if STOP.is_set():raise RuntimeError('global stop after another worker failure')
            try:
                row=journal.call(key,req)
                if row['response']['http_status']!=200:raise RuntimeError('HTTP failure, no retry')
                body=json.loads(row['response']['body'])
                if (body.get('model'),body.get('system_fingerprint'))!=('deepseek-flash','e72d3c7f663d3e1383b00b3884cb4501'):
                    raise RuntimeError('backend identity changed')
                result=previous.parse(row,'full',windows) if stage=='native' else method.parse(row,stage,windows)
                if type(result['tokens']) is not int:raise RuntimeError('unknown usage')
                return result
            except Exception:
                STOP.set();raise
        native=call('native',previous.request(case,'full',full),'native',full)
        claims={}
        for candidate in 'ABCD':
            slots={}
            def stage(name,history,windows):
                slots[name]=call(candidate+'/'+name,method.request(case,candidate,name,history,windows),name,windows)
            stage('author',{},full)
            author=method.message(slots['author'],'author')
            stage('ordinary0',{'author':author},full)
            stage('ordinary1',{'author':author,'previous_review':method.message(slots['ordinary0'],'ordinary0')},full)
            stage('ordinary2',{'author':author,'previous_reviews':[method.message(slots[n],n) for n in ('ordinary0','ordinary1')]},full)
            stage('restatement',{'author':author},full)
            # Challenge cannot see the ordinary verdicts or gold. It must ask a
            # specific question before retrieving a window from the real source.
            stage('challenge',{'author':author},[])
            challenge=method.message(slots['challenge'],'challenge')
            query=slots['challenge'].get('query',case['question']+' '+' '.join(case['choices'].values()))
            windows=retrieve(case['article'],query)
            freeze(directory/(candidate+'_retrieval.json'),{'candidate':candidate,'article_sha256':case['article_sha256'],
                'query':query,'challenge_valid':slots['challenge']['valid'],'windows':windows})
            stage('reply',{'author':author,'challenge':challenge},full)
            stage('nonadaptive',{'author':author,'challenge':challenge,
                  'reply':method.message(slots['restatement'],'restatement')},windows)
            stage('interactive',{'author':author,'challenge':challenge,'reply':method.message(slots['reply'],'reply')},windows)
            claims[candidate]=slots
        result={'native':native,'claims':claims};freeze(directory/'sealed_outputs.json',result)
        print('sealed interactive document',case['article_id'],flush=True)
        return case['id'],result


def run():
    STOP.clear()
    with lock(OUT):
        selection=json.loads((OUT/'selection.json').read_text())
        for name,value in selection['files'].items():assert h(OUT/name)==value
        freeze(OUT/'protocol.json',{'version':'targeted-challenge-v1','n_questions':6,'claims':24,'attempt_limit':222,
            'primary':'interactive vs ordinary2 question-macro balanced verification error',
            'same_calls':'each starts from same author; interactive3 calls vs ordinary3 calls; not equal realized tokens',
            'controls':['native full-source QA','ordinary0 single verification','ordinary2 three-call review','nonadaptive restatement before challenge'],
            'truth':'official best answer label, not proof validity; label only offline',
            'adaptation':'one author response to a specific challenge; not trained best response or equilibrium',
            'claim_constraint':'registered candidate fixed across all stages; all options defended, no gold-aware assignment',
            'stage_order':'native; per candidate author ordinary0 ordinary1 ordinary2 restatement challenge reply nonadaptive interactive',
            'retrieval':'public query chosen before access; top6 windows220 words stride110; source hashes and offsets recorded',
            'parser':'complete required JSON fields; harmless extra metadata accepted; invalid/abstain counted as error',
            'retry':'none; ambiguous call stops automatic resume; concurrent workers stop before next call',
            'automatic_expansion':False,'selection_sha256':h(OUT/'selection.json'),
            'design_sha256':h(ROOT/'docs/research/针对性竞争核验_预定方案_2026-10-25.md'),
            'code_sha256':{str(p.relative_to(ROOT)):h(p) for p in (Path(__file__),Path(method.__file__),
                ROOT/'experiments/core/mechanism/document_verification.py',Path(previous.__file__))}})
        cases=json.loads((OUT/'public_cases.json').read_text());assert len(cases)==6
        with ThreadPoolExecutor(max_workers=3) as workers:outputs=dict(workers.map(collect,cases))
        freeze(OUT/'sealed_outputs.json',outputs)
        freeze(OUT/'sealed_readouts.json',{cid:{candidate:{m:s[m]['verdict'] for m in ('ordinary0','ordinary2','nonadaptive','interactive')}
            for candidate,s in r['claims'].items()} for cid,r in outputs.items()})
        analyze()


def analyze():
    outputs=json.loads((OUT/'sealed_outputs.json').read_text());decisions=json.loads((OUT/'sealed_readouts.json').read_text())
    labels=json.loads((OUT/'private_labels.json').read_text());records=[]
    assert set(outputs)==set(labels)==set(decisions)
    for cid,r in outputs.items():
        for candidate,s in r['claims'].items():
            records.append({'id':cid,'candidate':candidate,'truth':int(candidate==labels[cid]),
                'native':'accept' if r['native']['answer']==candidate else 'reject' if r['native']['answer'] in 'ABCD' else 'invalid',
                **decisions[cid][candidate]})
    metrics={m:method.balanced_errors(records,m) for m in ('native','ordinary0','ordinary2','nonadaptive','interactive')}
    for m,metric in metrics.items():
        selected={}
        for cid in outputs:
            rr=[r for r in records if r['id']==cid];accepted=[r['candidate'] for r in rr if r[m]=='accept']
            selected[cid]=accepted[0] if len(accepted)==1 and all(r[m] in ('accept','reject') for r in rr) else 'abstain'
        metric.update(unique_accept_qa_correct=sum(selected[cid]==labels[cid] for cid in labels),
                      unique_accept_qa_abstain=sum(v=='abstain' for v in selected.values()))
    for m,stages in (('ordinary0',('author','ordinary0')),('ordinary2',('author','ordinary0','ordinary1','ordinary2')),
                     ('nonadaptive',('author','restatement','challenge','nonadaptive')),
                     ('interactive',('author','challenge','reply','interactive'))):
        metrics[m].update(tokens=sum(s[stage]['tokens'] for r in outputs.values() for s in r['claims'].values() for stage in stages),
                          calls=len(records)*len(stages))
    metrics['native'].update(tokens=sum(r['native']['tokens'] for r in outputs.values()),calls=len(outputs))
    comparison={}
    for baseline in ('ordinary0','ordinary2','nonadaptive','native'):
        def correct(row,m):return row[m] in ('accept','reject') and (row[m]=='accept')==bool(row['truth'])
        comparison[baseline]={'corrected_claims':sum(correct(r,'interactive') and not correct(r,baseline) for r in records),
            'harmed_claims':sum(not correct(r,'interactive') and correct(r,baseline) for r in records),
            'scope':'dependent candidate claims, not independent statistical trials'}
    raw=[r for p in OUT.rglob('responses.jsonl') for r in read_rows(p)]
    result={'scope':'six-document prospective engineering screen; no equilibrium/game-contribution claim',
        'metrics':metrics,'interactive_vs':comparison,'records':records,'physical_calls':len(raw),
        'physical_tokens':sum(json.loads(r['response']['body'])['usage']['total_tokens'] for r in raw),
        'sum_http_seconds':sum(r['wall_seconds'] for r in raw),
        'invalid_stages':sum(not s[stage]['valid'] for r in outputs.values() for s in r['claims'].values() for stage in method.STAGES),
        'limits':['one adaptive response is not optimized opponent or Stackelberg equilibrium',
                  'candidate verification is not automatic final QA selection or rationale truth',
                  'ordinary verifier has full source, interactive final has queried windows; equal calls not equal information or cost',
                  'native QA baseline is much cheaper and must not be omitted',
                  'source literal quotation is not entailment; no utility rewards for citation formatting']}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','analyze']);args=p.parse_args()
    {'prepare':prepare,'run':run,'analyze':analyze}[args.stage]()
