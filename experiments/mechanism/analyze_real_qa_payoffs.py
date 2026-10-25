"""Question-cluster analysis; conservative recovery of intact C strings only."""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.core.trust import extract_stance


def recover_c(raw):
    # Do not complete truncated text or call an LLM to invent missing content.
    matches=list(re.finditer(r'"C"\s*:\s*',raw))
    if len(matches)!=1:
        return None
    try:
        value,_=json.JSONDecoder().raw_decode(raw[matches[0].end():])
    except (ValueError,json.JSONDecodeError):
        return None
    return value if isinstance(value,str) else None


def paired_summary(rows, seed=2026101901):
    units=defaultdict(dict)
    for row in rows:
        key=(row['dataset'],row['question_id'],row['repeat'])
        if row['action'] in units[key]:
            raise ValueError('duplicate paired action unit')
        units[key][row['action']]=row
    if any(set(v)!={'direct','self_check'} for v in units.values()):
        raise ValueError('incomplete action pairs')
    pairs=[v for v in units.values() if set(v)=={'direct','self_check'}]
    differences=defaultdict(list)
    gained=lost=0
    for pair in pairs:
        left,right=pair['direct'],pair['self_check']
        diff=int(right['correct'])-int(left['correct'])
        gained+=diff==1
        lost+=diff==-1
        differences[(left['dataset'],left['question_id'])].append(diff)
    values=np.array([np.mean(v) for v in differences.values()])
    rng=np.random.default_rng(seed)
    # Paired question-level resampling; repeats remain inside their question.
    boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(10000)])
    return {'question_clusters':len(values),'paired_repeat_units':len(pairs),
        'accuracy_difference':float(values.mean()),
        'cluster_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
        'repeat_discordances_descriptive_only':{'direct_wrong_check_right':int(gained),'direct_right_check_wrong':int(lost)},
        'inference_scope':'exploratory question-cluster bootstrap; no repeat-level significance claim'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    d=args.directory
    rows=json.loads((d/'outcomes.json').read_text())
    calls={x['key']:x for x in map(json.loads,(d/'calls.jsonl').read_text().splitlines())}
    derived=[]
    recovered=[]
    for row in rows:
        copy=dict(row)
        if row['parse_error']:
            key=f'{row["dataset"]}/{row["question_id"]}/rep{row["repeat"]}/{row["action"]}/'+('draft' if row['action']=='direct' else 'revision')
            value=recover_c(calls[key]['raw_response'])
            if value is not None:
                copy['stance']=extract_stance(value)
                copy['correct']=copy['stance']==row['ground_truth']
                recovered.append({'key':key,'intact_C':value,'correct':copy['correct']})
        derived.append(copy)
    reports={}
    for name,data in [('original',rows),('intact_C_sensitivity',derived)]:
        groups={'pooled':data}
        for dataset in sorted({x['dataset'] for x in data}):
            groups[dataset]=[x for x in data if x['dataset']==dataset]
        for phase in ['calibration','development_check']:
            groups[phase]=[x for x in data if x['phase']==phase]
        reports[name]={k:paired_summary(v) for k,v in groups.items()}
    output={'reports':reports,'recovered_intact_conclusions':recovered,
        'original_records_unchanged':True,
        'decision':'no demonstrated overall self-check benefit; repair interface before new task measurement',
        'limitations':['small exploratory sample','repeats vary persona and are not independent questions',
            'same-model self-check is not external verification','no conclusion reconstructed from truncated output']}
    (d/'paired_analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
