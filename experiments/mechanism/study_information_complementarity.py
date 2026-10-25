"""Matched 3-backend x 3-repeat pilot. No personas, debate, rewards, or label access during collection."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import product, combinations
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.premise_review import decode
from experiments.mechanism.run_premise_review import parse_row
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows, utc

OUT = p.ROOT/'experiments/results/resource_allocation/information_complementarity_v1'
BACKENDS = ('deepseek', 'zhipu_flash', 'ali_qwen')
INSTRUCTION = ('Answer independently and carefully using your available knowledge. '
    'Respect the original question and its answer criterion; do not invent external verification. '
    'Return one complete JSON object with answer (one registered choice or abstain) '
    'and reasoning (sufficient concise reasoning). Put answer first. No text outside JSON.')


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare():
    from experiments.mechanism.prepare_routing_holdout import PINS
    from experiments.mechanism.prepare_upgrade_p2 import pool, native_mc
    from experiments.mechanism.prepare_upgrade_pilot import canonical, strings
    from experiments.mechanism.build_upgrade_history import build
    if (OUT/'provenance.json').exists():
        prov = json.loads((OUT/'provenance.json').read_text())
        for name, h in prov['files'].items():
            assert fingerprint(OUT/name) == h
        return prov
    dep = Path('/private/tmp/mas_p2_dependencies')
    if dep.is_dir():
        sys.path.insert(0, str(dep))
    args = SimpleNamespace(
        strategyqa=p.ROOT/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json',
        truthfulqa=p.ROOT/'experiments/data/raw/upgrade_p2_v1/truthfulqa_mc_authors.json',
        mmlu=p.ROOT/'experiments/data/raw/upgrade_p2_v1/mmlu_test.parquet')
    for d,h in PINS.items():
        assert fingerprint(getattr(args,d)) == h
    mmlu, args.mmlu = args.mmlu, None
    candidates, _ = pool(args)
    import pyarrow.parquet as pq
    for row in pq.read_table(mmlu).to_pylist():
        if len(set(row['choices'])) == len(row['choices']):
            candidates.append(native_mc('mmlu', p.digest([row['subject'],row['question']])[:24],
                                        row['question'],list(row['choices']),int(row['answer'])))
    print('native pools loaded', flush=True)
    history = build(p.ROOT)
    if history['errors']:
        raise ValueError('historical parsing failures')
    used = {canonical(v['text']) for v in history['texts'].values()}
    ranked = []
    for d in ('strategyqa','mmlu','truthfulqa'):
        seen = set()
        for r in sorted((r for r in candidates if r['domain']==d),
                        key=lambda r:p.digest(['complementarity-2026-10-24-v1',r['source_id']])):
            stem = canonical(r['base_question'])
            if stem in used or stem in seen:
                continue
            ranked.append(r)
            seen.add(stem)
            if len(seen)==120:
                break
    print('history indexed',len(history['files']),'files; backstop screening',flush=True)
    blocked, coverage = {}, []
    for f in history['files']:
        if f.get('role')=='verified_official_raw_pool_not_model_execution' or f.get('duplicate_of'):
            continue
        path = p.ROOT/f['path']
        assert fingerprint(path)==f['sha256']
        raw=path.read_text()
        data=[json.loads(s) for s in raw.splitlines() if s.strip()] if path.suffix=='.jsonl' else json.loads(raw)
        leaves=list(strings(data)); ids=set(leaves); content='\n'.join(canonical(s) for s in leaves)
        for r in ranked:
            ident=r['domain']+':'+r['source_id']
            if ident not in blocked and (r['source_id'] in ids or canonical(r['base_question']) in content):
                blocked[ident]=f['path']
        coverage.append({'path':f['path'],'sha256':f['sha256']})
    selected=[]
    for d in ('strategyqa','mmlu','truthfulqa'):
        available=[r for r in ranked if r['domain']==d and r['domain']+':'+r['source_id'] not in blocked]
        if len(available)<12:
            raise ValueError('insufficient isolated source groups '+d)
        selected.extend([{**r,'split':'calibration' if i<6 else 'evaluation'} for i,r in enumerate(available[:12])])
    assert len({canonical(r['base_question']) for r in selected})==36
    public=[{k:r[k] for k in ('question','choices','source_id','split')} |
            {'id':r['domain']+':'+r['source_id']} for r in selected]
    public.sort(key=lambda r:p.digest(['complementarity-order-v1',r['id']]))
    OUT.mkdir(parents=True,exist_ok=True)
    outputs={'public_cases.json':public,'private_labels.json':[{'id':r['domain']+':'+r['source_id'],'label':r['label']} for r in selected],
             'history_snapshot.json':history,'backstop_coverage.json':coverage}
    for name,obj in outputs.items():
        freeze(OUT/name,obj)
    prov={'n':36,'per_domain':12,'split_counts':{'calibration':18,'evaluation':18},'source_pins':PINS,
          'history_files':len(history['files']),'backstop_files':len(coverage),'excluded_candidates':blocked,
          'sampling':'input-only fixed hash order, no selection based on errors/disagreements',
          'limits':'exact normalized source-stem and ID isolation inside indexed repo; not semantic or pretraining novelty',
          'files':{name:fingerprint(OUT/name) for name in outputs}}
    freeze(OUT/'provenance.json',prov)
    print('prepared 36 isolated cases',flush=True)
    return prov


def request(case,backend,model,replicate):
    # Replicate is a ledger index only, not an API seed or a different instruction.
    return {'backend':backend,'model':model,'temperature':.7,'max_tokens':800,
            'response_format':{'type':'json_object'},'messages':[
                {'role':'system','content':INSTRUCTION},
                {'role':'user','content':json.dumps({'question':case['question'],'choices':case['choices']},sort_keys=True)}]}


def majority(answers):
    counts=Counter(x for x in answers if x not in (None,'abstain'))
    return counts.most_common(1)[0][0] if counts and counts.most_common(1)[0][1]>=2 else 'abstain'


def preflight(journal,models):
    for b in BACKENDS:
        req={'backend':b,'model':models[b],'temperature':.7,'max_tokens':100,
             'response_format':{'type':'json_object'},'messages':[
                 {'role':'user','content':'Connectivity fixture only. Return JSON {"answer":"yes","reasoning":"ready"}.'}]}
        row=journal.call('preflight/'+b,req)
        parsed=parse_row(row,['yes','no'])
        if not parsed['valid'] or parsed['answer']!='yes':
            raise RuntimeError('backend preflight failed, no silent substitution: '+b+' HTTP '+str(parsed['http_status']))
        print('preflight',b,'returned',parsed['model'],flush=True)


def collect():
    from experiments.core.llm import LLM_BACKENDS
    with lock(OUT):
        prepare()
        if not all(LLM_BACKENDS[b]['api_key'] for b in BACKENDS):
            raise ValueError('missing configured backend; do not silently change design')
        models={b:LLM_BACKENDS[b]['model'] for b in BACKENDS}
        cases=json.loads((OUT/'public_cases.json').read_text())
        protocol={'version':'complementarity-v1','created_utc':None,'cases_hash':p.digest(cases),
                  'models':models,'endpoints':{b:LLM_BACKENDS[b]['base_url'] for b in BACKENDS},
                  'n':36,'replicates':3,'temperature':.7,'max_tokens':800,
                  'max_http_attempts':654,'planned_task_calls':324,'preflight_calls':3,
                  'retry':'one identical retry for completed HTTP429/5xx; ambiguous transport stops',
                  'primary':'fixed replicate0 heterogeneous vote vs each same-model triple and calibration-selected same-model triple',
                  'selection':'calibration only; ties use BACKENDS order; no label-based per-question selection',
                  'secondary':'all27 replicate combinations are dependent sensitivity, not 27 independent trials',
                  'code_sha256':{str(f.relative_to(p.ROOT)):fingerprint(f) for f in [Path(__file__),Path(decode.__code__.co_filename)]},
                  'limits':['equal call count/output cap not equal tokens or money','no independent information ownership or equilibrium claimed']}
        freeze(OUT/'protocol.json',protocol)
        journal=CallJournal(OUT,p.transport,654)
        preflight(journal,models)
        sealed={}
        for i,case in enumerate(cases):
            slots={}
            schedules=sorted(product(BACKENDS,range(3)),key=lambda x:p.digest([case['id'],list(x)]))
            for b,rep in schedules:
                req=request(case,b,models[b],rep)
                key=case['id']+'/'+b+'/'+str(rep)
                row=journal.call(key,req)
                status=row['response']['http_status']
                if status==429 or status>=500:
                    row=journal.call(key+'/retry1',req)
                parsed=parse_row(row,case['choices'])
                if parsed['http_status'] in (401,402,403):
                    raise RuntimeError('API authorization/balance failure: '+b)
                slots[b+'/'+str(rep)]=parsed
            sealed[case['id']]=slots
            print('independent pool saved',i+1,'/36',flush=True)
        freeze(OUT/'sealed_outputs.json',sealed)
        return analyze()


def summarize(cases,sealed,labels):
    outcomes={};metrics={};percase=[]
    for b in BACKENDS:
        outcomes['single0/'+b]=[]
        outcomes['homogeneous/'+b]=[]
    outcomes['heterogeneous0']=[]
    outcomes['heterogeneous_all27_expected']=[]
    for case in cases:
        s=sealed[case['id']];gold=labels[case['id']]
        answers={b:[s[b+'/'+str(i)]['answer'] for i in range(3)] for b in BACKENDS}
        row={'id':case['id'],'split':case['split'],'gold':gold,'answers':answers,'decisions':{},'any_correct':{}}
        for b in BACKENDS:
            row['decisions']['single0/'+b]=answers[b][0]
            row['decisions']['homogeneous/'+b]=majority(answers[b])
            row['any_correct'][b]=any(a==gold for a in answers[b])
        row['decisions']['heterogeneous0']=majority([answers[b][0] for b in BACKENDS])
        row['heterogeneous0_any_correct']=any(answers[b][0]==gold for b in BACKENDS)
        row['any_of9_correct']=any(a==gold for xs in answers.values() for a in xs)
        row['heterogeneous0_wasted_correct']=row['heterogeneous0_any_correct'] and row['decisions']['heterogeneous0']!=gold
        row['heterogeneous0_unanimous']=len({answers[b][0] for b in BACKENDS})==1
        row['homogeneous_unanimous']={b:len(set(answers[b]))==1 for b in BACKENDS}
        for name,answer in row['decisions'].items():
            outcomes[name].append(answer==gold)
        outcomes['heterogeneous_all27_expected'].append(sum(majority([answers[b][i] for b,i in zip(BACKENDS,indices)])==gold
                                                          for indices in product(range(3),repeat=3))/27)
        percase.append(row)
    for name,values in outcomes.items():
        metrics[name]={'n':len(values),'correct':sum(values)}
        if name=='heterogeneous_all27_expected':
            metrics[name]['note']='analytic mean across dependent combinations; not observed extra paid trials'
        else:
            def slots(case):
                s=sealed[case['id']]
                if name=='heterogeneous0':return [s[b+'/0'] for b in BACKENDS]
                kind,b=name.split('/')
                return [s[b+'/'+str(i)] for i in ([0] if kind=='single0' else range(3))]
            records=[r for c in cases for r in slots(c)]
            metrics[name].update({'calls':len(records),'tokens':sum(r['tokens'] for r in records) if all(r['tokens'] is not None for r in records) else None,
                                  'sum_http_seconds':sum(r['wall_seconds'] for r in records),
                                  'invalid':sum(not r['valid'] for r in records)})
    diagnostics={'heterogeneous0_any_correct':sum(r['heterogeneous0_any_correct'] for r in percase),
                 'heterogeneous0_correct_available_but_vote_wrong':sum(r['heterogeneous0_wasted_correct'] for r in percase),
                 'any_of9_correct':sum(r['any_of9_correct'] for r in percase),
                 'heterogeneous0_unanimous':sum(r['heterogeneous0_unanimous'] for r in percase),
                 'homogeneous_unanimous':{b:sum(r['homogeneous_unanimous'][b] for r in percase) for b in BACKENDS},
                 'homogeneous_any_correct':{b:sum(r['any_correct'][b] for r in percase) for b in BACKENDS},
                 'unique_backend0_correct':{b:sum(r['answers'][b][0]==r['gold'] and all(r['answers'][other][0]!=r['gold'] for other in BACKENDS if other!=b) for r in percase) for b in BACKENDS},
                 'error_pairs_backend0':{a+'|'+b:{'both_wrong':sum(r['answers'][a][0]!=r['gold'] and r['answers'][b][0]!=r['gold'] for r in percase),
                                               'disagreement':sum(r['answers'][a][0]!=r['answers'][b][0] for r in percase)} for a,b in combinations(BACKENDS,2)}}
    return {'metrics':metrics,'diagnostics':diagnostics,'per_case':percase}


def analyze():
    cases=json.loads((OUT/'public_cases.json').read_text())
    sealed=json.loads((OUT/'sealed_outputs.json').read_text())
    assert len(cases)==36 and set(sealed)=={c['id'] for c in cases}
    assert all(set(s)=={b+'/'+str(i) for b in BACKENDS for i in range(3)} for s in sealed.values())
    labels={r['id']:r['label'] for r in json.loads((OUT/'private_labels.json').read_text())}
    cal=[c for c in cases if c['split']=='calibration']
    evaluation=[c for c in cases if c['split']=='evaluation']
    cal_result=summarize(cal,sealed,labels)
    best=min(BACKENDS,key=lambda b:(-cal_result['metrics']['homogeneous/'+b]['correct'],BACKENDS.index(b)))
    best_single=min(BACKENDS,key=lambda b:(-cal_result['metrics']['single0/'+b]['correct'],BACKENDS.index(b)))
    freeze(OUT/'calibration_selection.json',{'best_homogeneous':best,'best_single0':best_single,
           'rule':'calibration correct count descending, fixed backend-order tie break; no evaluation labels used in selection'})
    results={split:summarize(sub,sealed,labels) for split,sub in [('calibration',cal),('evaluation',evaluation),('all',cases)]}
    for name,result in results.items():
        contrasts={}
        for baseline in ['homogeneous/'+best,'single0/'+best_single]+['homogeneous/'+b for b in BACKENDS]:
            cells=Counter()
            for r in result['per_case']:
                a=r['decisions'][baseline]==r['gold'];b=r['decisions']['heterogeneous0']==r['gold']
                cells['corrections']+=not a and b;cells['harms']+=a and not b
            contrasts[baseline]=dict(cells)
        result['heterogeneous0_vs_baseline']=contrasts
    rows=read_rows(OUT/'responses.jsonl')
    final={'scope':'small source-isolated exploratory study; no claim of novel heterogeneous ensembling or equilibrium',
           'n':36,'new_task_calls':324,'actual_http_attempts':len(read_rows(OUT/'attempts.jsonl')),
           'models_returned':{b:sorted({r['model'] for s in sealed.values() for k,r in s.items() if k.startswith(b+'/') and r['model']}) for b in BACKENDS},
           'calibration_selected_homogeneous':best,'calibration_selected_single':best_single,'results':results,
           'all_collection_tokens_including_preflight':sum(json.loads(r['response']['body']).get('usage',{}).get('total_tokens',0) for r in rows),
           'response_journal_hash':p.digest(rows),'automatic_expansion':False}
    freeze(OUT/'result.json',final)
    print(json.dumps({k:v for k,v in final.items() if k!='results'},ensure_ascii=False,indent=2),flush=True)
    print('EVALUATION',json.dumps(results['evaluation']['metrics'],ensure_ascii=False),flush=True)
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.run:collect()
    elif args.analyze:analyze()
    else:
        with lock(OUT):prepare()
