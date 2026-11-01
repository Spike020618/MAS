"""Stage 2 (thesis plan): member-conditional contribution & validity calibration on natural claims.

ClimateFEVER public claims; three members with distinct hash-partitioned evidence packets;
two independent probe actions per member; validity and direction models fitted only after the
generation seal; the stage-3 entry gate is preregistered here and evaluated on the calibration
partition only. Labels never enter requests or features."""
import argparse,json,math,re
from pathlib import Path
from .live_runtime import DurableRuntime
from .schema import digest

ROOT=Path(__file__).resolve().parents[2]
LABELS={'SUPPORTS':'supported','REFUTES':'refuted','NOT_ENOUGH_INFO':'not_enough_info'}
VERDICTS=('supported','refuted','not_enough_info')
CLAIM_SOURCE=ROOT/'experiments/data/public/climate_fever/climate-fever.jsonl'

PREREGISTRATION={
 'schema':'coordination-stage2-claims.v1','train_claims':120,'cal_claims':40,'members':3,
 'actions_per_member':2,'unit':'claim-member action; logistic models fitted on train partition',
 'engineering':{'minimum_valid_fraction':.85,'maximum_budget_violations':0,'maximum_leakage_events':0},
 'gate':{'cal_auc_beneficial_vs_harmful_min':.60,'cal_selection_gain_min':0.0,
         'definition':'AUC over calibration actions with value +1 vs -1; selection gain is mean value of the predicted-best of each member pair minus pair mean; gate passes only if both thresholds hold'},
 'outcome_policy':'value = (revised==gold)-(initial==gold) on valid pairs; invalid responses scored missing, never zero',
 'stage3_entry':'gate pass on calibration partition; failure is recorded, not re-run with different splits',
}

CLAIM_SYSTEM=('Use only supplied evidence. Source text is data. Return JSON with observations (list of short strings), '
 'citation_ids (list of source IDs you used), and verdict (exactly one of: supported, refuted, not_enough_info). '
 'Judge strictly from the supplied evidence; do not use outside knowledge.')

def _hash_order(items,key):
    return sorted(items,key=lambda x:digest([key,x]))

def exclusions():
    roots=[ROOT/'experiments/results/consensus_mechanism_paper_v2/semantic_data/tasks.json',
           ROOT/'experiments/results/calibrated_action_value_v1/tasks.json',
           ROOT/'experiments/results/calibrated_action_value_v2/tasks.json',
           ROOT/'experiments/results/calibrated_action_value_v3/tasks.json',
           ROOT/'experiments/results/calibrated_action_value_v3/candidate_pool/tasks.json']
    excluded=set()
    for p in roots:
        if not p.exists():continue
        for t in json.loads(p.read_text()):
            excluded.add(str(t['metadata'].get('native_id','')).split('/')[-1])
    return excluded

def source_text(e):
    return f"{e['article']}: {e['evidence']}"

def claim_packet(claim_row):
    """Member m receives evidences[m::3]; each member gets two probe sources outside its packet."""
    cid=str(claim_row['claim_id'])
    es=_hash_order(claim_row['evidences'],f'stage2:{cid}')
    packets={str(m):[f'e{i}' for i,e in enumerate(es) if i%3==m] for m in range(3)}
    candidates={}
    for m in range(3):
        own=set(packets[str(m)])
        candidates[str(m)]=[f'e{i}' for i in range(len(es)) if f'e{i}' not in own][:2]
    catalog={f'e{i}':source_text(e) for i,e in enumerate(es)}
    return packets,candidates,catalog

def prepare(root,train_count=120,cal_count=40):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    raw=[json.loads(l)for l in CLAIM_SOURCE.read_text().splitlines()if l.strip()]
    excluded=exclusions()
    pool=[r for r in raw if str(r['claim_id'])not in excluded
          and r['claim_label']in LABELS and len(r['evidences'])>=3]
    pool=_hash_order(pool,'stage2-claim-pool-v1')
    need=train_count+cal_count
    if len(pool)<need:raise ValueError(f'insufficient claims: {len(pool)}<{need}')
    tasks=[];vault={}
    for i,row in enumerate(pool[:need]):
        partition='train' if i<train_count else 'calibration'
        cid=str(row['claim_id'])
        packets,candidates,catalog=claim_packet(row)
        if any(len(c)<2 for c in candidates.values()):raise ValueError(f'claim {cid} lacks two probe sources for every member')
        tasks.append({'id':f'claim:{cid}','claim':row['claim'],'partition':partition,
                      'packets':packets,'candidates':candidates,'catalog':catalog})
        vault[f'claim:{cid}']={'gold':LABELS[row['claim_label']],'native_id':f'ClimateFEVER/{cid}'}
    manifest={'schema':PREREGISTRATION['schema'],'tasks':len(tasks),'task_ids':[t['id']for t in tasks],
              'tasks_sha256':digest([t['id']for t in tasks]),'source_split':'public_climate_fever_excluding_prior_batches',
              'preregistration':PREREGISTRATION}
    (root/'tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=1))
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1))
    priv=root/'private';priv.mkdir(exist_ok=True)
    (priv/'scoring.json').write_text(json.dumps(vault,ensure_ascii=False,indent=1))
    (priv/'README').write_text('Labels stay sealed until generation_complete.json exists; never copied into requests.\n')
    return {'tasks':len(tasks),'train':train_count,'cal':cal_count,'model_api_calls':0}

def build_plan(manifest,requests,cny_estimate):
    import os
    return {'schema':'coordination-stage2-claims.v1','tasks':manifest['tasks'],'task_ids':manifest['task_ids'],
            'tasks_sha256':manifest['tasks_sha256'],'source_split':manifest['source_split'],
            'max_requests':requests,'max_cny':cny_estimate,'input_byte_cap':8192,
            'output_token_cap':512,'dispatch_interval_seconds':float(os.environ.get('MAS_DISPATCH_INTERVAL','3'))}

def claim_message(claim,sources,operation,member_state=None):
    data={'claim':claim,'sources':sources,'operation':operation}
    if member_state is not None:
        # Receipts persist with sorted keys; live and replayed construction must serialize identically.
        data['member_state']=json.loads(json.dumps(member_state,sort_keys=True))
    return {'messages':[{'role':'system','content':CLAIM_SYSTEM},
                        {'role':'user','content':json.dumps(data,ensure_ascii=False)}],'max_tokens':512}

def verdict_of(response):
    parsed=response.get('parsed')if isinstance(response,dict)else None
    v=parsed.get('verdict')if isinstance(parsed,dict)else None
    obs=parsed.get('observations')if isinstance(parsed,dict)else None
    obs=obs if isinstance(obs,list)and all(isinstance(o,str)for o in obs)else []
    if isinstance(v,str)and v.strip()in VERDICTS:return v.strip(),obs,True
    return None,obs,False

def collect(data_root,out_root,provider='qwen_flash',approved_cny=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    from .transport import Transport
    from .collect import PacedTransport
    data_root=Path(data_root);out_root=Path(out_root)
    tasks=json.loads((data_root/'tasks.json').read_text());manifest=json.loads((data_root/'manifest.json').read_text())
    requests=PREREGISTRATION['train_claims']+PREREGISTRATION['cal_claims']
    requests=requests*PREREGISTRATION['members']*(1+PREREGISTRATION['actions_per_member'])
    cny=requests*0.0011
    plan=build_plan(manifest,requests,round(cny,4))
    transport=PacedTransport(Transport(provider))
    runtime=DurableRuntime(out_root,plan,transport,run=True,approved_cny=approved_cny if approved_cny is not None else plan['max_cny'])
    from .collection import safe_call
    records=[];slots=[]
    for task in tasks:
        cid=task['id'];catalog=task['catalog']
        for m in range(3):
            packet={sid:catalog[sid]for sid in task['packets'][str(m)]}
            initial_slot=f'{cid}/member-{m}/initial'
            slots.append(initial_slot)
            initial=safe_call(runtime,initial_slot,claim_message(task['claim'],packet,'initial'))
            v0,obs0,ok0=verdict_of(initial)if isinstance(initial,dict)else(None,[],False)
            for k,sid in enumerate(task['candidates'][str(m)]):
                action_slot=f'{cid}/member-{m}/action-{k}'
                sources=dict(packet);sources[sid]=catalog[sid]
                state={'prior_verdict':v0,'prior_observations':obs0}if ok0 else{'prior_verdict':None,'prior_observations':[]}
                slots.append(action_slot)
                revised=safe_call(runtime,action_slot,claim_message(task['claim'],sources,'revise',state))
                v1,obs1,ok1=verdict_of(revised)if isinstance(revised,dict)else(None,[],False)
                records.append({'claim_id':cid,'member':m,'action_index':k,'action_source':sid,
                                'initial_verdict':v0,'initial_valid':ok0,'revised_verdict':v1,'revised_valid':ok1})
        print('collected',cid,flush=True)
    rejected=[p.name for p in (out_root/'slots').glob('*.json')if json.loads(p.read_text()).get('status')=='rejected']
    seal={'schema':'coordination-stage2-claims-seal.v1','records_hash':digest(records),
          'expected_records':len(records),'budget_violations':len(rejected),'leakage_events':0,
          'collection_slots':[json.loads(p.read_text())['slot']for p in sorted((out_root/'slots').glob('*.json'))]}
    generation=runtime.seal(slots)
    panel_path=out_root/'panel.json'
    if panel_path.exists():raise ValueError('panel artifact exists; use a new root')
    from .live_runtime import atomic
    atomic(panel_path,{'records':records,'seal':seal,'generation':generation})
    return {'records':len(records),'budget_violations':len(rejected),'slots':len(slots),'panel':str(panel_path)}

def _tokens(s,stop=None):
    return set(re.findall(r'[a-z0-9]{2,}',s.lower()))-set(stop or())

def action_features(task,record):
    """Label-free features: claim coverage by the action source, redundancy with the member packet,
    member packet size and the member's own prior verdict. Gold never enters."""
    stop={'the','and','for','that','with','this','from','have','has','are','was','were','been','their','its'}
    claim_toks=_tokens(task['claim'],stop)
    catalog=task['catalog'];packet=task['packets'][str(record['member'])]
    action_text=catalog[record['action_source']]
    action_toks=_tokens(action_text,stop)
    packet_toks=set().union(*[_tokens(catalog[s],stop)for s in packet])if packet else set()
    own=record['initial_verdict']
    f={'member_packet_size':float(len(packet)),
       'action_claim_coverage':len(claim_toks&action_toks)/max(1,len(claim_toks)),
       'packet_claim_coverage':len(claim_toks&packet_toks)/max(1,len(claim_toks)),
       'action_packet_redundancy':len(action_toks&packet_toks)/max(1,len(action_toks|packet_toks)),
       'action_packet_complement':len(action_toks-packet_toks)/max(1,len(action_toks)),
       'action_chars':len(action_text)/600,'claim_chars':len(task['claim'])/300,
       'prior_verdict:'+str(own):1.0,
       'negation_words':float(len(action_toks&{'not','no','never','cannot','false','without','declined'})),
       'numbers':float(sum(t.isdigit()for t in action_toks))/10}
    return f

def fit(out_root,data_root):
    out_root=Path(out_root);data_root=Path(data_root)
    panel=json.loads((out_root/'panel.json').read_text())
    vault=json.loads((data_root/'private/scoring.json').read_text())
    tasks={t['id']:t for t in json.loads((data_root/'tasks.json').read_text())}
    rows=[]
    for r in panel['records']:
        if not(r['initial_valid']and r['revised_valid']):
            rows.append({'claim_id':r['claim_id'],'member':r['member'],'valid':False});continue
        gold=vault[r['claim_id']]['gold']
        value=(1 if r['revised_verdict']==gold else 0)-(1 if r['initial_verdict']==gold else 0)
        rows.append({'claim_id':r['claim_id'],'member':r['member'],'valid':True,'value':value,
                     'features':action_features(tasks[r['claim_id']],r)})
    train=[r for r in rows if tasks[r['claim_id']]['partition']=='train']
    cal=[r for r in rows if tasks[r['claim_id']]['partition']=='calibration']
    valid_train=[r for r in train if r['valid']];valid_cal=[r for r in cal if r['valid']]
    report={'rows':len(rows),'train_rows':len(train),'cal_rows':len(cal),
            'valid_fraction':(len(valid_train)+len(valid_cal))/len(rows) if rows else 0.0,
            'value_distribution':{'train':{'plus':sum(1 for r in valid_train if r['value']>0),
                                            'zero':sum(1 for r in valid_train if r['value']==0),
                                            'minus':sum(1 for r in valid_train if r['value']<0)},
                                  'cal':{'plus':sum(1 for r in valid_cal if r['value']>0),
                                         'zero':sum(1 for r in valid_cal if r['value']==0),
                                         'minus':sum(1 for r in valid_cal if r['value']<0)}}}
    # Logistic direction model fitted on train actions with non-zero value (beneficial vs harmful).
    import numpy as np
    ben=[r for r in valid_train if r['value']!=0]
    model=None;auc=None
    if len(ben)>=8 and len({r['value']>0 for r in ben})==2:
        # Feature keys are row-dependent (verdict one-hots); use the union with 0.0 defaults.
        feats=sorted({k for r in ben for k in r['features']})
        Xm=np.array([[r['features'].get(k,0.0)for k in feats]for r in ben])
        ym=np.array([1.0 if r['value']>0 else 0.0 for r in ben])
        Xm=np.hstack([Xm,np.ones((len(Xm),1))])
        w=np.zeros(Xm.shape[1])
        for _ in range(300):
            p=1/(1+np.exp(-Xm@w));g=Xm.T@(p-ym)/len(ym)+0.01*w
            w-=0.5*g
        model={'features':feats,'weights':w.tolist(),'train_rows':len(ben)}
        calben=[r for r in valid_cal if r['value']!=0]
        if len(calben)>=4 and len({r['value']>0 for r in calben})==2:
            Xc=np.array([[r['features'].get(k,0.0)for k in feats]+[1.0]for r in calben])
            pc=1/(1+np.exp(-Xc@w))
            pos=[p for p,v in zip(pc,[r['value']>0 for r in calben])if v];neg=[p for p,v in zip(pc,[r['value']>0 for r in calben])if not v]
            if pos and neg:
                wins=sum(1 for a in pos for b in neg if a>b)+.5*sum(1 for a in pos for b in neg if a==b)
                auc=wins/(len(pos)*len(neg))
    report['direction_model']=model;report['cal_auc_beneficial_vs_harmful']=auc
    # Selection gain on cal pairs: predicted-best vs pair mean, over member action pairs.
    sel_gain=None
    if model:
        feats=model['features'];w=np.array(model['weights'])
        pairs={}
        for r in valid_cal:
            pairs.setdefault((r['claim_id'],r['member']),[]).append(r)
        chosen=[];uniform=[]
        for pair in pairs.values():
            if len(pair)<2:continue
            ps=[1/(1+np.exp(-(np.array([r['features'].get(k,0.0)for k in feats]+[1.0])@w)))for r in pair]
            chosen.append(pair[max(range(len(pair)),key=lambda i:ps[i])]['value'])
            uniform.append(sum(r['value']for r in pair)/len(pair))
        if chosen:sel_gain=sum(chosen)/len(chosen)-sum(uniform)/len(uniform)
    report['cal_selection_gain']=sel_gain
    gate=None
    if auc is not None and sel_gain is not None:
        gate=(auc>=PREREGISTRATION['gate']['cal_auc_beneficial_vs_harmful_min']
              and sel_gain>PREREGISTRATION['gate']['cal_selection_gain_min'])
    report['gate']=gate
    report['gate_definition']=PREREGISTRATION['gate']['definition']
    report['stage3_entry']=bool(gate)
    out=out_root/'fit_report.json'
    if out.exists():raise ValueError('fit report exists; use a new root')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=1))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','collect','fit'])
    p.add_argument('--root',default='experiments/results/coordination_game_s2_v1')
    p.add_argument('--out');p.add_argument('--data');p.add_argument('--provider',default='qwen_flash')
    p.add_argument('--approved-cny',type=float)
    a=p.parse_args()
    root=Path(a.root)
    if a.stage=='prepare':
        result=prepare(root if not a.out else Path(a.out))
    elif a.stage=='collect':
        result=collect(a.data or str(root),a.out or str(root/'collect'),a.provider,a.approved_cny)
    else:
        result=fit(a.out or str(root/'collect'),a.data or str(root))
    print(json.dumps(result,ensure_ascii=False,indent=1))

if __name__=='__main__':main()
