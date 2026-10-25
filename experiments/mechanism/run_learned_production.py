"""Bounded DeepSeek production-control experiment: 12 train + 12 frozen test.

Five calls/case: 3 initial fast reports, two optional thinking upgrades.
Test game selection is committed after initials, before any upgrades; remaining
upgrades are collected afterwards solely for common-menu baseline replay.
No Qwen, model weight training, automatic retries, or task-specific prompts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys

from experiments.core.mechanism import learned_production_game as game
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows,utc
from experiments.mechanism.run_premise_review import parse_row
from experiments.mechanism import study_explicit_thinking_control as api

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/resource_allocation/learned_production_v1'
DATA=OUT/'source_data'


def h(path):return sha256(path.read_bytes()).hexdigest()


def prepare():
    from experiments.mechanism import study_information_complementarity as source
    previous=source.OUT
    try:
        source.OUT=DATA;source.prepare()
    finally:source.OUT=previous
    public=json.loads((DATA/'public_cases.json').read_text())
    labels={r['id']:r['label'] for r in json.loads((DATA/'private_labels.json').read_text())}
    for split,source_split in (('train','calibration'),('test','evaluation')):
        chosen=[]
        for domain in ('strategyqa','mmlu','truthfulqa'):
            available=sorted((c for c in public if c['split']==source_split and c['id'].startswith(domain+':')),
                             key=lambda c:sha256(('production-v1/'+c['id']).encode()).hexdigest())
            chosen.extend(available[:4])
        chosen.sort(key=lambda c:sha256(c['id'].encode()).hexdigest())
        assert len(chosen)==12
        freeze(OUT/(split+'_public.json'),chosen)
        freeze(OUT/(split+'_labels.json'),{c['id']:labels[c['id']] for c in chosen})
    print('prepared 12 train +12 test, 120 calls maximum',flush=True)


def protocol():
    paths=[Path(__file__),Path(game.__file__),Path(api.__file__),Path(parse_row.__code__.co_filename)]
    value={'version':'learned-production-v1','train_n':12,'test_n':12,'max_calls':120,
        'model_requested':'deepseek-flash','model_expected':'deepseek-flash',
        'fingerprint_expected':'e72d3c7f663d3e1383b00b3884cb4501',
        'actions':['retain initial fast report','independent thinking-enabled report'],
        'controllers':2,'fixed_anchor_reports':1,'initial_calls_per_case':3,
        'token_cap_both':4096,'sampling':'temperature omitted for both modes; backend default and mode behavior can differ',
        'cost_per_upgrade_token':str(game.COST_PER_TOKEN),
        'learning':'full-information empirical payoff tables by initial agreement state; state n<3 falls back global',
        'game':'strict unilateral empirical best responses, start00, role0 then role1, retain on exact ties',
        'non_game':'centralized exhaustive empirical optimization of identical potential over same four profiles',
        'primary':'test net objective game vs centralized_erm; correctness and realized costs also disclosed',
        'controls':['fixed00','fixed01','fixed10','fixed11','uniform_random_expected','same_upgrade_count_random_expected','centralized_erm'],
        'test_protocol':'freeze train policy before any test API calls; commit game before test upgrades; test labels only after readouts sealed',
        'semantics':'software controllers with defined team payoff and resource cost; not natural LLM economic agents or a new payment proof',
        'retry':'none; unknown transport stops; all invalid complete outputs retained as abstain',
        'scope':'ECON-inspired minimal controller, not ECON implementation or BNE reproduction',
        'automatic_expansion':False,
        'code_sha256':{str(p.relative_to(ROOT)):h(p) for p in paths},
        'design_sha256':h(ROOT/'docs/research/可学习信息生产协调_锁定方案_2026-10-25.md'),
        'data_sha256':{name:h(OUT/name) for name in ('train_public.json','train_labels.json','test_public.json','test_labels.json')}}
    freeze(OUT/'protocol.json',value)
    return value


def collect_case(case,split,policy,p):
    directory=OUT/split/sha256(case['id'].encode()).hexdigest()
    with lock(directory):
        journal=CallJournal(directory,api.transport,5);slots={}
        def call(slot):
            req=api.request(case,'enabled' if slot.startswith('think') else 'disabled')
            row=journal.call(slot,req)
            if row['response']['http_status']!=200:raise RuntimeError('HTTP failure; no retry')
            body=json.loads(row['response']['body'])
            if body.get('model')!=p['model_expected'] or body.get('system_fingerprint')!=p['fingerprint_expected']:
                raise RuntimeError('backend identity changed; stop without silent substitution')
            parsed=parse_row(row,case['choices'])
            if parsed['tokens'] is None:raise RuntimeError('usage missing; cost estimation cannot proceed')
            slots[slot]=parsed
        for i in range(3):call('fast'+str(i))
        if policy is not None:
            initial={k:slots[k] for k in ('fast0','fast1','fast2')}
            decisions={'context':game.context(initial),'game':game.decide(policy,initial,'game'),
                       'centralized_erm':game.decide(policy,initial,'centralized_erm'),
                       'policy_sha256':h(OUT/'frozen_policy.json'),
                       'initial_sha256':sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()}
            freeze(directory/'decision.json',decisions)
            if not (directory/'decision_time.json').exists():freeze(directory/'decision_time.json',{'utc':utc()})
            preferred=[i+1 for i,a in enumerate(decisions['game']) if a]
        else:preferred=[]
        for i in preferred+[i for i in (1,2) if i not in preferred]:call('think'+str(i))
        freeze(directory/'sealed_outputs.json',slots)
        print('sealed',split,case['id'],flush=True)
        return case['id'],slots


def execute(split):
    from experiments.core.llm import LLM_BACKENDS
    if not LLM_BACKENDS['deepseek']['api_key']:raise RuntimeError('DeepSeek configuration missing')
    with lock(OUT):
        p=protocol()
        cases=json.loads((OUT/(split+'_public.json')).read_text())
        assert len(cases)==12 and len({c['id'] for c in cases})==12
        train_ids={c['id'] for c in json.loads((OUT/'train_public.json').read_text())}
        test_ids={c['id'] for c in json.loads((OUT/'test_public.json').read_text())}
        assert not train_ids & test_ids
        policy=json.loads((OUT/'frozen_policy.json').read_text()) if split=='test' else None
        with ThreadPoolExecutor(max_workers=3) as pool:
            outputs=dict(pool.map(lambda c:collect_case(c,split,policy,p),cases))
        freeze(OUT/(split+'_sealed_outputs.json'),outputs)
        if split=='train':
            labels=json.loads((OUT/'train_labels.json').read_text())
            records=[{'slots':outputs[c['id']],'label':labels[c['id']],
                      'context':game.context({k:outputs[c['id']][k] for k in ('fast0','fast1','fast2')})} for c in cases]
            freeze(OUT/'frozen_policy.json',game.fit(records))
            print('training complete; policy frozen',flush=True)
        else:analyze()


def analyze():
    cases=json.loads((OUT/'test_public.json').read_text());outputs=json.loads((OUT/'test_sealed_outputs.json').read_text())
    policy=json.loads((OUT/'frozen_policy.json').read_text());decisions={}
    for c in cases:
        slots=outputs[c['id']];initial={k:slots[k] for k in ('fast0','fast1','fast2')}
        selected={name:game.decide(policy,initial,name) for name in ('game','centralized_erm')}
        selected.update({'fixed'+''.join(map(str,p)):list(p) for p in game.PROFILES})
        decisions[c['id']]={'profiles':selected,'answers':{k:game.outcome(slots,p) for k,p in selected.items()}}
    freeze(OUT/'test_sealed_readouts.json',decisions)
    labels=json.loads((OUT/'test_labels.json').read_text())
    metrics={};rows=[]
    for c in cases:
        slots=outputs[c['id']];gold=labels[c['id']];d=decisions[c['id']]
        base_tokens=sum(slots['fast'+str(i)]['tokens'] for i in range(3))
        base_seconds=sum(slots['fast'+str(i)]['wall_seconds'] for i in range(3))
        distributions={name:[(tuple(profile),1.)] for name,profile in d['profiles'].items()}
        distributions['uniform_random_expected']=[(p,.25) for p in game.PROFILES]
        same=[p for p in game.PROFILES if sum(p)==sum(d['profiles']['game'])]
        distributions['same_upgrade_count_random_expected']=[(p,1/len(same)) for p in same]
        record={'id':c['id'],'gold':gold,'methods':{}}
        for name,dist in distributions.items():
            cell={'correct':0.,'tokens':float(base_tokens),'calls':3.,'sum_http_seconds':base_seconds,'upgrades':0.,'objective':0.}
            for profile,weight in dist:
                correct=game.outcome(slots,profile)==gold
                tokens=sum(slots['think'+str(i+1)]['tokens']*a for i,a in enumerate(profile))
                cell['correct']+=weight*correct;cell['tokens']+=weight*tokens
                cell['calls']+=weight*sum(profile);cell['upgrades']+=weight*sum(profile)
                cell['sum_http_seconds']+=weight*sum(slots['think'+str(i+1)]['wall_seconds']*a for i,a in enumerate(profile))
                cell['objective']+=weight*(correct-float(game.COST_PER_TOKEN)*tokens)
            record['methods'][name]=cell
            total=metrics.setdefault(name,{k:0. for k in cell}|{'n':len(cases)})
            for k,v in cell.items():total[k]+=v
        rows.append(record)
    physical=[r for path in OUT.rglob('responses.jsonl') for r in read_rows(path)]
    result={'scope':'small prospective frozen-policy production control; no general equilibrium or game superiority claim',
        'metrics':metrics,'rows':rows,'physical_calls':len(physical),
        'physical_tokens':sum(json.loads(r['response']['body'])['usage']['total_tokens'] for r in physical),
        'sum_physical_http_seconds':sum(r['wall_seconds'] for r in physical),
        'invalid_outputs':sum(not s['valid'] for r in outputs.values() for s in r.values()),
        'limits':['random controls are analytic expectations, not extra independent API trials',
                  'all controls pay three common initial calls plus their selected upgrades',
                  'net objective excludes fixed initial cost shared across these controls; report total tokens separately',
                  'centralized optimizer uses same training feedback; necessary baseline, not an inferior straw man',
                  'only one response per action and role; potential-outcome replay is not repeated deployment',
                  'ECON-inspired finite empirical table, not neural belief training or original ECON replication']}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','test','analyze']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='analyze':analyze()
    else:execute(args.stage)
