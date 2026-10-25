"""One frozen 32-question development check of a generic source-pipeline repair."""
import argparse
from collections import Counter
import fcntl
import hashlib
from itertools import product
import json
from pathlib import Path

from experiments.mechanism.validate_locked_source_game import Caller,canonical,freeze,retrieve,valid_report
from experiments.mechanism.pilot_source_sharing import digest
from experiments.mechanism.pilot_source_sharing_neutral import prepare as neutral_prepare
from experiments.mechanism.sprint_source_pipeline import repair_sources

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/results/mechanism/stackelberg'


def prepare(directory):
    previous=BASE/'locked_source_game_test_v1'
    official_path=previous/'official_train.json'
    _,oldjobs=neutral_prepare()
    prompts=[next(r['messages'][0]['content'] for r in oldjobs if r['agent']==i) for i in range(3)]
    files=[Path(__file__),ROOT/'experiments/mechanism/sprint_source_pipeline.py',
           ROOT/'experiments/mechanism/pilot_grounded_evidence.py',
           ROOT/'experiments/mechanism/validate_locked_source_game.py']
    protocol={'version':'sprint-source-development-v1','n_questions':32,'n_agents':3,'n_repeats':1,
        'conditions':['D','R_legacy','R_repaired'],'max_successful_calls':320,
        'changes':['canonical-page dedup','reuse generic paragraph/table extraction'],
        'unchanged':['queries','model alias','report prompts','temperature','max output tokens','majority rule'],
        'scope':'new development questions only; one dataset StrategyQA; not a held-out confirmation',
        'source_file_sha256':hashlib.sha256(official_path.read_bytes()).hexdigest(),
        'seed':'MAS-sprint-source-dev-2026-10-20-v1','prompts':prompts,
        'source_failure_policy':'keep all questions, no manual supplementation',
        'primary_diagnostic':'question-level paired team quality-cost outcomes; no reward tuning',
        'decision_policy':'advance only on an observable development benefit; otherwise no larger reward scan',
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    freeze(directory/'protocol.json',protocol)
    case_path=directory/'cases.json'
    if case_path.exists():return protocol,json.loads(case_path.read_text())
    history=json.loads((previous/'history_audit.json').read_text())
    excluded=set(history['canonical_question_sha256'])
    for row in json.loads((previous/'cases.json').read_text()):
        excluded.add(hashlib.sha256(canonical(row['question']).encode()).hexdigest())
    preview=set(json.loads((previous/'external_preview_disclosure.json').read_text())['previewed_official_ids'])
    raw=json.loads(official_path.read_text())
    candidates=[{'question_id':'StrategyQA-official/'+r['qid'],'question':r['question']}
        for r in raw if r['qid'] not in preview and hashlib.sha256(canonical(r['question']).encode()).hexdigest() not in excluded]
    candidates.sort(key=lambda r:hashlib.sha256((protocol['seed']+canonical(r['question'])).encode()).hexdigest())
    selected=[];seen=set()
    for row in candidates:
        key=canonical(row['question'])
        if key not in seen:selected.append(row);seen.add(key)
        if len(selected)==32:break
    if len(selected)!=32:raise ValueError('insufficient new development cases')
    freeze(case_path,selected)
    freeze(directory/'exposure.json',{'prior_case_overlap':0,'prior_preview_overlap':0,'n':32,
        'exclusion_sha256':digest(sorted(excluded)),'scope':'previous execution/development records, not model pretraining'})
    return protocol,selected


def majority(stances):
    count=Counter(stances)
    return next((s for s in ['yes','no'] if count[s]>=2),'abstain')


def analyze(directory,protocol,cases,caller):
    # Answers first enter only after every response is complete.
    official=json.loads((BASE/'locked_source_game_test_v1/official_train.json').read_text())
    gold={'StrategyQA-official/'+r['qid']:'yes' if r['answer'] else 'no' for r in official}
    rows=[]
    for case in cases:
        qid=case['question_id'];conditions={}
        for condition in protocol['conditions']:
            reports=[caller.calls[f'{qid}/{condition}/{i}'] for i in range(3)]
            stances=[r['parsed']['C'] if valid_report(r) else 'abstain' for r in reports]
            conditions[condition]={'stance':majority(stances),'correct':majority(stances)==gold[qid],
                'votes':stances,'tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in reports),
                'wall_sum':sum(r['wall_seconds'] for r in reports),
                'schema_failures':sum(not valid_report(r) for r in reports)}
        rows.append({'question_id':qid,'conditions':conditions})
    summary={c:{'correct':sum(r['conditions'][c]['correct'] for r in rows),'n':len(rows),
        'mean_tokens':sum(r['conditions'][c]['tokens'] for r in rows)/len(rows),
        'schema_failures':sum(r['conditions'][c]['schema_failures'] for r in rows)} for c in protocol['conditions']}
    pairs=[]
    for c in ['R_legacy','R_repaired']:
        pairs.append({'condition':c,'corrected_vs_D':sum(not r['conditions']['D']['correct'] and r['conditions'][c]['correct'] for r in rows),
            'harmed_vs_D':sum(r['conditions']['D']['correct'] and not r['conditions'][c]['correct'] for r in rows)})
    # All action profiles for the repaired environment remain composable: no peer reports.
    profiles=[]
    for profile in product((0,1),repeat=3):
        correct=0;tokens=0
        for case in cases:
            qid=case['question_id']
            reports=[caller.calls[f'{qid}/{"R_repaired" if a else "D"}/{i}'] for i,a in enumerate(profile)]
            stances=[r['parsed']['C'] if valid_report(r) else 'abstain' for r in reports]
            correct+=majority(stances)==gold[qid]
            tokens+=sum((r.get('usage') or {}).get('total_tokens',0) for r in reports)
        profiles.append({'profile':list(profile),'correct':correct,'n':len(cases),'mean_tokens':tokens/len(cases)})
    result={'scope':protocol['scope'],'summary':summary,'paired_diagnostics':pairs,'repaired_profile_surface':profiles,
        'n_calls':len(caller.calls),'tokens_all_calls':sum((r.get('usage') or {}).get('total_tokens',0) for r in caller.calls.values()),
        'returned_models':sorted({str(r.get('returned_model')) for r in caller.calls.values()}),
        'source_sufficiency':'not inferred from accuracy or existence; requires separate content review',
        'task_specific_checker_used':False,'rows':rows}
    freeze(directory/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out-dir',type=Path,required=True);parser.add_argument('--collect',action='store_true')
    args=parser.parse_args();directory=args.out_dir;directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ['banks','snapshots','repaired_banks']:(directory/name).mkdir(exist_ok=True)
    protocol,cases=prepare(directory)
    print('Frozen source-only development pilot: 32 questions, at most 320 new successful calls',flush=True)
    if not args.collect:return
    caller=Caller(directory)
    for index,case in enumerate(cases):
        qid=case['question_id'];bank=retrieve(directory,case,caller)
        legacy=bank['sources']
        sources=[dict(s,snapshot_path=str(directory/'snapshots'/f'{s["source_hash"]}.html')) for s in legacy]
        repaired=repair_sources(case['question'],sources,ROOT)
        freeze(directory/'repaired_banks'/f'{qid.split("/")[-1]}.json',repaired)
        conditions=list(protocol['conditions'])
        # Predefined hash order; no outcomes used for scheduling.
        conditions.sort(key=lambda c:hashlib.sha256((qid+c).encode()).hexdigest())
        for condition in conditions:
            available=[] if condition=='D' else legacy if condition=='R_legacy' else repaired
            bundles=[available[:1],available[1:2],available]
            for i in range(3):
                public_sources=[{key:s[key] for key in ['source_hash','title','url','text']} for s in bundles[i]]
                messages=[{'role':'system','content':protocol['prompts'][i]},
                    {'role':'user','content':json.dumps({'question':case['question'],'sources':public_sources},ensure_ascii=False)}]
                caller.call(f'{qid}/{condition}/{i}',messages)
        print(f'Completed development question {index+1}/32',flush=True)
    if len(caller.calls)!=320:raise ValueError('unexpected final call count')
    analyze(directory,protocol,cases,caller)


if __name__=='__main__':main()
