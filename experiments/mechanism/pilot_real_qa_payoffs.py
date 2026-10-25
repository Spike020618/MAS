"""Paired real-QA action measurements, before contract selection.

Thirty training-split questions, two repeats, independent direct and draft+audit
paths. Correctness is scored after generation; no answer labels enter prompts.
All successful API calls are checkpointed before parsing and never resampled.
"""
import argparse
import fcntl
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.core.llm import LLMClient, AGENT_PERSONAS
from experiments.core.trust import extract_stance
from experiments.mechanism.collect import load_local_questions, parse_aeic, _call_json


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--compact-schema-v2',action='store_true',
                        help='New development protocol: answer-first, typed and bounded AEIC')
    args=parser.parse_args()
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another collector is using this directory')
    cases=[]
    for dataset in ['StrategyQA','MMLU-Hard','TruthfulQA']:
        for i,q in enumerate(load_local_questions(dataset,10,'train')):
            cases.append({'dataset':dataset,'phase':'calibration' if i<6 else 'development_check',**q})
    protocol={'version':'real-qa-payoffs-pilot-v1','split':'train',
        'historical_exposure':'existing training cache; no claim of globally pristine unseen questions',
        'n_questions':30,'repeats':2,'actions':['direct','self_check'],
        'backend':'deepseek','model_alias':'from configured DeepSeek backend',
        'temperature':0.3,'max_tokens_per_call':800,'seed':2026101801,
        'expected_api_calls':180,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'case_sha256':digest(cases),
        'gates':'estimate action differences and uncertainty first; no equilibrium claim',
        'verification_scope':'self-check by same LLM; not independent factual verification',
        'reward':'no incentives applied during forced action measurement'}
    if args.compact_schema_v2:
        protocol.update(version='real-qa-payoffs-pilot-v2-compact',
                        format='C-first; bounded typed A/E/I; identical constraints for both actions',
                        purpose='development rerun on same questions; not independent final validation')
    text=json.dumps(protocol,ensure_ascii=False,indent=2)+'\n'
    target=out/'protocol.json'
    if target.exists() and target.read_text()!=text:
        raise ValueError('frozen protocol mismatch')
    target.write_text(text)
    (out/'private_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    cache={}
    raw_path=out/'calls.jsonl'
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            item=json.loads(line)
            if item['key'] in cache: raise ValueError('duplicate call key')
            cache[item['key']]=item
    client=LLMClient(backend='deepseek',max_tokens=800)

    def call(key,messages,stage):
        prompt_hash=digest(messages)
        if key in cache:
            if cache[key]['prompt_sha256']!=prompt_hash:
                raise ValueError('resume prompt mismatch')
            return cache[key]
        try:
            raw,trace=_call_json(client,messages,0.3,800,stage)
        except Exception as exc:
            with (out/'errors.jsonl').open('a') as f:
                f.write(json.dumps({'key':key,'utc':datetime.now(timezone.utc).isoformat(),
                    'error':str(exc)},ensure_ascii=False)+'\n')
            raise
        item={'key':key,'utc':datetime.now(timezone.utc).isoformat(),'prompt_sha256':prompt_hash,
              'messages':messages,'raw_response':raw,'trace':trace}
        with raw_path.open('a') as f:
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
            f.flush()
        cache[key]=item
        print(f'{len(cache)}/180 {key}',flush=True)
        return item

    rows=[]
    schedule=[]
    for case in cases:
        for repeat in [0,1]:
            for action in ['direct','self_check']:
                schedule.append((case,repeat,action))
    random.Random(protocol['seed']).shuffle(schedule)
    for case,repeat,action in schedule:
        base=f'{case["dataset"]}/{case["question_id"]}/rep{repeat}/{action}'
        system={'role':'system','content':AGENT_PERSONAS[repeat]['system']}
        prompt=('Answer this real QA question carefully. Return strict JSON with A (assumptions), '
                'E (a list of specific evidence), I (inference), C (a concise answer beginning Yes or No). '
                'Only the question is provided; do not assume access to external tools.\nQuestion:\n'+case['question'])
        if args.compact_schema_v2:
            prompt=(
                'Evaluate the question and its proposed answer carefully. No external tools are available. '
                'Return exactly one valid compact JSON object with keys in this order: C, A, E, I. '
                'C: a string beginning Yes or No and a concise answer, at most 25 words. '
                'A: a string of at most 35 words, not an array. '
                'E: an array of at most two strings, each at most 45 words. '
                'I: one string of at most 65 words, not a list. '
                'Do not add a type key, markdown, or commentary outside JSON. '
                'Report the final judgment first; keep supporting fields within these limits so the JSON completes. '
                'If no proposed answer is present, directly answer the yes/no question.\nQuestion:\n'+case['question'])
        draft=call(base+'/draft',[system,{'role':'user','content':prompt}],'draft')
        traces=[draft['trace']]
        final=draft['raw_response']
        if action=='self_check':
            revision=call(base+'/revision',[system,{'role':'user','content':prompt},
                {'role':'assistant','content':final},{'role':'user','content':
                 'Check the draft for factual mistakes, unsupported assumptions and logical contradictions. '
                 'Retain the answer if warranted or correct it when necessary. '
                 'Return the same AEIC JSON schema. This is self-review, not independent verification.'}], 'self_check')
            final=revision['raw_response']
            traces.append(revision['trace'])
        error=None
        try:
            report=parse_aeic(final)
            stance=extract_stance(report.C)
        except Exception as exc:
            report=None
            stance=None
            error=str(exc)
        rows.append({'dataset':case['dataset'],'question_id':case['question_id'],
            'phase':case['phase'],'repeat':repeat,'action':action,
            'ground_truth':case['ground_truth'],'stance':stance,
            'correct':stance==case['ground_truth'], 'parse_error':error,
            'report':report.__dict__ if report else None,
            'tokens':sum(t['prompt_tokens']+t['completion_tokens'] for t in traces),
            'api_calls':len(traces),'api_latency_s':sum(t['latency_s'] for t in traces)})
    (out/'outcomes.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    summary={}
    for phase in ['calibration','development_check']:
        for action in ['direct','self_check']:
            group=[r for r in rows if r['phase']==phase and r['action']==action]
            summary[f'{phase}/{action}']={'n':len(group),
                'accuracy':sum(r['correct'] for r in group)/len(group),
                'mean_tokens':sum(r['tokens'] for r in group)/len(group),
                'parse_failures':sum(bool(r['parse_error']) for r in group)}
    result={'scope':'forced-action real-QA development pilot; repeated questions not independent samples',
        'completed_units':len(rows),'api_calls':len(cache),'groups':summary,
        'total_tokens':sum(x['trace']['prompt_tokens']+x['trace']['completion_tokens'] for x in cache.values()),
        'next_step':'estimate question-cluster uncertainty before deciding whether to optimize contracts'}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
