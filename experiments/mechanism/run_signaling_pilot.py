"""12 reserved, previously unexecuted questions; DEVELOPMENT only, no tuning.

Fixed lambda=.5, G-then-D, contrastive normalization, T1/nonthinking DeepSeek.
One shared score pool, six predeclared readouts, no new evidence or payment.
Each question has K+2 main calls plus2 reversed-code G diagnostics.
No test labels opened until all outputs have been sealed.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from experiments.core.mechanism.scored_signaling import compare,decode_score
from experiments.mechanism import probe_signaling_interface as interface
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/resource_allocation/signaling_pilot_v1'
SOURCE=ROOT/'experiments/results/resource_allocation/reliable_readout_data_v1'
NAMES=('generation','contrastive_generation','verification','raw_product','contrastive_product','game')


def fingerprint(path):return sha256(path.read_bytes()).hexdigest()


def prepare():
    if (OUT/'selection.json').exists():
        prov=json.loads((OUT/'selection.json').read_text())
        for rel,h in prov['source_hashes'].items():assert fingerprint(ROOT/rel)==h
        assert fingerprint(OUT/'public_cases.json')==prov['public_sha256']
        return
    public=json.loads((SOURCE/'public_cases.json').read_text())
    executed=json.loads((SOURCE.with_name('reliable_readout_v1')/'public_cases.json').read_text())
    ids={c['id'] for c in executed};cases=[]
    for domain in ('strategyqa','mmlu','truthfulqa'):
        available=sorted((c for c in public if c['id'].startswith(domain+':') and c['id'] not in ids),
                         key=lambda c:sha256(('signaling-pilot-v1/'+c['id']).encode()).hexdigest())
        assert len(available)==6
        cases.extend(available[:4])
    cases.sort(key=lambda c:sha256(('signaling-order-v1/'+c['id']).encode()).hexdigest())
    assert len(cases)==12 and len({c['id'] for c in cases})==12
    # Check actual API request journals rather than treating prepared data as execution.
    journals=[]
    for path in sorted((ROOT/'experiments/results').rglob('responses.jsonl')):
        if path.parent==OUT:continue
        text=path.read_text()
        for c in cases:
            assert c['id'] not in text,'selected source ID already in execution journal: '+str(path)
            assert c['question'] not in text,'selected rendered question already in journal: '+str(path)
            assert json.dumps(c['question'],ensure_ascii=False)[1:-1] not in text,'escaped question already executed'
        journals.append({'path':str(path.relative_to(ROOT)),'sha256':fingerprint(path)})
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'public_cases.json',cases)
    freeze(OUT/'selection.json',{'n':12,'per_domain':4,'scope':'developmental screen, not confirmatory test',
        'source':'previously prepared but reserved unexecuted calibration half; input-only hash selection',
        'source_hashes':{str(p.relative_to(ROOT)):fingerprint(p) for p in
            [SOURCE/'public_cases.json',SOURCE/'private_labels.json',SOURCE/'provenance.json',
             SOURCE.with_name('reliable_readout_v1')/'public_cases.json']},
        'public_sha256':fingerprint(OUT/'public_cases.json'),'execution_journals_checked':journals,
        'limits':'prior source-preparation isolation plus exact execution-journal checks, not semantic or pretraining novelty'})
    print('prepared12 reserved cases; scheduled calls',sum(len(c['choices'])+4 for c in cases),flush=True)


def request(case,slot,model):
    choices=case['choices'];codes=list('ABCDEFGHIJKLMNOPQRST'[:len(choices)])
    if len(choices)>20:raise ValueError('too many choices for token interface')
    reverse=slot.endswith('_reverse');mapped=list(reversed(choices)) if reverse else choices
    if slot.startswith('g_'):
        objective='incorrect' if slot.startswith('g_negative') else 'correct'
        system='Choose the '+objective+' option using your knowledge. Output exactly one registered code character and nothing else.'
        content={'question':case['question'],'code_to_registered_choice':dict(zip(codes,mapped))}
    elif slot.startswith('d/') and slot[2:] in choices:
        system='Assess the proposed answer using your knowledge. Output exactly A for supported or B for unsupported, nothing else.'
        content={'question':case['question'],'proposed_registered_choice':slot[2:]}
    else:raise ValueError('invalid slot')
    return {'backend':'deepseek','model':model,'max_tokens':8,'temperature':1.,
            'thinking':{'type':'disabled'},'logprobs':True,'top_logprobs':20,
            'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(content,ensure_ascii=False,sort_keys=True)}]}


def parsed(row,case,slot):
    codes=list('ABCDEFGHIJKLMNOPQRST'[:len(case['choices'])]) if slot.startswith('g_') else ['A','B']
    return decode_score(row['response'],codes)


def collect():
    from experiments.core.llm import LLM_BACKENDS
    with lock(OUT):
        prepare();cases=json.loads((OUT/'public_cases.json').read_text())
        planned=sum(len(c['choices'])+4 for c in cases);model=LLM_BACKENDS['deepseek']['model']
        code=[Path(__file__),ROOT/'experiments/core/mechanism/scored_signaling.py',
              ROOT/'experiments/core/mechanism/signaling_game.py']
        freeze(OUT/'protocol.json',{'version':'signaling-pilot-v1','n':12,'model_requested':model,
            'expected_returned_model':'deepseek-flash','expected_fingerprint':'e72d3c7f663d3e1383b00b3884cb4501',
            'temperature':1.,'thinking':'disabled','max_tokens':8,'lambda':.5,'normalization':'contrastive',
            'update_order':'generator then discriminator','primary':'game vs contrastive_product on ALL12 cases',
            'controls':list(NAMES),'format_or_missing_score':'no repairs; unavailable readout abstains; report per-method availability',
            'reversal_diagnostic':'two reverse-code generation calls; not chosen by labels, not extra independent questions',
            'planned_task_calls':planned,'attempt_limit':planned,'retry':'none; unknown transport stops',
            'code_sha256':{str(p.relative_to(ROOT)):fingerprint(p) for p in code},
            'public_sha256':fingerprint(OUT/'public_cases.json'),
            'scope':'developmental restricted-token score readout; no full reasoning or equal-FLOPs advantage claimed'})
        protocol=json.loads((OUT/'protocol.json').read_text())
        for rel,h in protocol['code_sha256'].items():assert fingerprint(ROOT/rel)==h
        journal=CallJournal(OUT,interface.transport,planned);sealed={}
        for index,case in enumerate(cases):
            slots=['g_positive','g_negative','g_positive_reverse','g_negative_reverse']+['d/'+c for c in case['choices']]
            slots.sort(key=lambda s:sha256((case['id']+'/'+s).encode()).hexdigest())
            outputs={}
            for slot in slots:
                row=journal.call(case['id']+'/'+slot,request(case,slot,model))
                outputs[slot]=parsed(row,case,slot)
                p=outputs[slot]
                if row['response']['http_status']!=200:raise RuntimeError('HTTP failure preserved; no silent retry')
                if p.get('model')!=protocol['expected_returned_model'] or p.get('fingerprint')!=protocol['expected_fingerprint']:
                    raise RuntimeError('model identity/fingerprint changed; stop before further questions')
            sealed[case['id']]=outputs
            print('sealed scoring pool',index+1,'/12',flush=True)
        freeze(OUT/'sealed_outputs.json',sealed)
        return analyze()


def decisions(case,slots):
    choices=case['choices'];codes=list('ABCDEFGHIJKLMNOPQRST'[:len(choices)])
    answers={name:'abstain' for name in NAMES};details={}
    gp,gn=slots['g_positive'],slots['g_negative'];ds=[slots['d/'+c] for c in choices]
    from experiments.core.mechanism.scored_signaling import rank
    if gp['usable']:answers['generation']=rank(choices,[gp['probabilities'][k] for k in codes])
    if all(d['usable'] for d in ds):answers['verification']=rank(choices,[d['probabilities']['A'] for d in ds])
    if gp['usable'] and all(d['usable'] for d in ds):
        answers['raw_product']=rank(choices,[gp['probabilities'][k]*d['probabilities']['A'] for k,d in zip(codes,ds)])
    if gp['usable'] and gn['usable']:
        ratios=[gp['probabilities'][k]/(gp['probabilities'][k]+gn['probabilities'][k]) for k in codes]
        answers['contrastive_generation']=rank(choices,ratios)
    if gp['usable'] and gn['usable'] and all(d['usable'] for d in ds):
        details=compare(choices,[gp['probabilities'][k] for k in codes],[gn['probabilities'][k] for k in codes],
                        [(d['probabilities']['A'],d['probabilities']['B']) for d in ds])
        answers.update(details['answers'])
    reverse={}
    for arm in ('positive','negative'):
        normal,flipped=slots['g_'+arm],slots['g_'+arm+'_reverse']
        if normal['usable'] and flipped['usable']:
            a=rank(choices,[normal['probabilities'][k] for k in codes])
            b=rank(list(reversed(choices)),[flipped['probabilities'][k] for k in codes])
            reverse[arm]={'normal':a,'reverse':b,'changed':a!=b}
    return {'answers':answers,'details':details,'reversed_code_diagnostics':reverse}


def analyze():
    cases=json.loads((OUT/'public_cases.json').read_text());sealed=json.loads((OUT/'sealed_outputs.json').read_text())
    assert set(sealed)=={c['id'] for c in cases}
    # Decisions and their commitment exist BEFORE labels are read by this phase.
    readouts={c['id']:decisions(c,sealed[c['id']]) for c in cases}
    freeze(OUT/'sealed_readouts.json',readouts)
    labels={r['id']:r['label'] for r in json.loads((SOURCE/'private_labels.json').read_text())}
    metrics={name:{'n':len(cases),'correct':sum(readouts[c['id']]['answers'][name]==labels[c['id']] for c in cases),
                  'abstain':sum(readouts[c['id']]['answers'][name]=='abstain' for c in cases)} for name in NAMES}
    for name in NAMES:
        selected=[]
        for c in cases:
            keys=(['g_positive'] if name=='generation' else
                  ['g_positive','g_negative'] if name=='contrastive_generation' else
                  ['d/'+x for x in c['choices']] if name=='verification' else
                  ['g_positive']+['d/'+x for x in c['choices']] if name=='raw_product' else
                  ['g_positive','g_negative']+['d/'+x for x in c['choices']])
            selected.extend(sealed[c['id']][key] for key in keys)
        metrics[name].update(required_scoring_calls=len(selected),
            scoring_tokens=sum(r['tokens'] for r in selected) if all(type(r['tokens']) is int for r in selected) else None,
            unusable_required_responses=sum(not r['usable'] for r in selected))
    paired={name:{'corrections':sum(readouts[c['id']]['answers'][name]!=labels[c['id']] and readouts[c['id']]['answers']['game']==labels[c['id']] for c in cases),
                  'harms':sum(readouts[c['id']]['answers'][name]==labels[c['id']] and readouts[c['id']]['answers']['game']!=labels[c['id']] for c in cases)}
            for name in NAMES if name!='game'}
    from math import comb
    for cell in paired.values():
        b,c=cell['corrections'],cell['harms'];n=b+c
        cell['descriptive_exact_mcnemar_p']=min(1.,2*sum(comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
    rows=read_rows(OUT/'responses.jsonl');usage=[json.loads(r['response']['body']).get('usage',{}).get('total_tokens') for r in rows]
    result={'scope':'12-case developmental readout comparison, not confirmatory efficacy','metrics':metrics,'game_vs':paired,
        'all_cases_retained':True,'fully_usable_games':sum(bool(r['details']) for r in readouts.values()),
        'reversed_code_changes':{arm:sum(r['reversed_code_diagnostics'].get(arm,{}).get('changed',False) for r in readouts.values()) for arm in ('positive','negative')},
        'http_attempts':len(read_rows(OUT/'attempts.jsonl')),'http_responses':len(rows),
        'known_tokens_lower_bound':sum(u for u in usage if type(u)is int and u>=0),'unknown_usage_responses':sum(type(u)is not int or u<0 for u in usage),
        'sum_http_seconds':sum(r['wall_seconds'] for r in rows),
        'records':[{'id':c['id'],'gold':labels[c['id']],'answers':readouts[c['id']]['answers']} for c in cases],
        'limits':['same-pool readout ablation, not equal minimal deployment costs across controls',
                  'no independent vote/full-reasoning budget baseline yet; no superiority or novelty declaration',
                  'candidate coding, prompt objective and truncated top20 can affect references'],
        'automatic_expansion':False}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--analyze',action='store_true');args=p.parse_args()
    if args.run:collect()
    elif args.analyze:analyze()
    else:
        with lock(OUT):prepare()
