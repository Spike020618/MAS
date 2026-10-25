"""Post-hoc answer-prefix sensitivity, never alters primary sealed outputs.

If JSON is invalid but a completed response starts with a complete answer field,
recover ONLY that explicit registered choice; do not repair/score its reasoning.
"""
import copy
import json
import re
from pathlib import Path

from experiments.mechanism import study_two_backend_complementarity as parent
from experiments.mechanism import study_explicit_thinking_control as mode
from experiments.mechanism import confirm_thinking_control as confirm
from experiments.mechanism.run_upgrade_pilot import freeze

PREFIX=re.compile(r'^\s*\{\s*"answer"\s*:\s*"([^"\\]+)"')
OUT=parent.OUT/'extraction_sensitivity'


def answer(slot,choices):
    if slot['valid']:return slot['answer']
    if slot['finish_reason']!='stop' or slot['http_status']!=200:return 'abstain'
    m=PREFIX.match(slot['raw'])
    return m.group(1) if m and m.group(1) in (*choices,'abstain') else 'abstain'


def main():
    result={'scope':'post-hoc explicit completed answer-field sensitivity, not repaired primary outputs',
            'new_model_calls':0,'changes':[],'results':{}}
    cases=json.loads((parent.OUT/'public_cases.json').read_text());choices={c['id']:c['choices'] for c in cases}
    labels={r['id']:r['label'] for r in json.loads((parent.OUT/'private_labels.json').read_text())}
    sealed=json.loads((parent.OUT/'sealed_outputs.json').read_text());sensitive=copy.deepcopy(sealed)
    for qid,slots in sealed.items():
        for key,slot in slots.items():
            value=answer(slot,choices[qid])
            sensitive[qid][key]['answer']=value
            if value!=slot['answer']:result['changes'].append({'study':'two_backend','id':qid,'slot':key,'old':slot['answer'],'new':value,'raw':slot['raw']})
    result['results']['two_backend']={split:parent.summarize([c for c in cases if split=='all' or c['split']==split],sensitive,labels)['metrics']
                                    for split in ('calibration','evaluation','all')}
    for name,directory,data in [('thinking_discovery',mode.OUT,parent.OUT),('thinking_confirmation',confirm.OUT,confirm.DATA)]:
        if not (directory/'sealed_outputs.json').exists():continue
        cs=json.loads((data/'public_cases.json').read_text());ch={c['id']:c['choices'] for c in cs}
        gold={r['id']:r['label'] for r in json.loads((data/'private_labels.json').read_text())}
        slots=json.loads((directory/'sealed_outputs.json').read_text());new={}
        for qid,pair in slots.items():
            new[qid]={}
            for key,slot in pair.items():
                value=answer(slot,ch[qid]);new[qid][key]=value
                if value!=slot['answer']:result['changes'].append({'study':name,'id':qid,'slot':key,'old':slot['answer'],'new':value,'raw':slot['raw']})
        result['results'][name]={scope:{key:sum(new[c['id']][key]==gold[c['id']] for c in cs if scope=='all' or c['split']==scope)
                                              for key in mode.MODES} for scope in ('all','calibration','evaluation')}
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='changes'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
