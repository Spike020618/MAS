"""Offline assembly of explored light claims; null ID echo is metadata only."""
import copy
import json
from pathlib import Path
from collections import Counter

from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.state_d2 import StateD2
from experiments.paired_deliberation.explore_light_actor import register


def null_id_row(response,cid):
    if isinstance(response,dict) and set(response)=={'claim_id','kind','scope'} and response['claim_id'] is None:
        return {'claim_id':cid,'kind':response['kind'],'scope':copy.deepcopy(response['scope'])}
    return None


def run(output,source):
    output,source=Path(output),Path(source);previous=json.loads((source/'summary.json').read_text())
    contexts={r['id']:r for r in json.loads((ROOT/'experiments/results/paired_deliberation_d2_engineering_r1/inputs.json').read_text())}
    reports=[];groups={}
    for prior in previous['records']:
        raw=prior['raw'];item=contexts[prior['id']];reg=EvidenceRegistry(item['evidence'],item['training_view'])
        rows={r['claim_id']:copy.deepcopy(r) for r in prior['parsed_rows']};ops=[]
        for failure in prior['failures']:
            row=null_id_row(failure.get('response'),failure['claim_id'])
            if row is not None:
                rows[row['claim_id']]=row;ops.append({'claim_id':row['claim_id'],'operation':'omit null ID echo; runtime reattaches original ID for single-text request'})
        wanted=['c'+str(i+1) for i in range(len(raw['claims']))]
        registered=register(raw,{'claims':[rows[c] for c in wanted if c in rows]},reg)
        entry={'actor':prior['actor'],'id':prior['id'],'owner':prior['owner'],'raw':raw,'registrations':registered,'metadata_operations':ops}
        reports.append(entry)
        key=(prior['actor'],prior['id']);group=groups.setdefault(key,{'registry':reg,'typed_reports':{},'unregistered':[],'raw_reports':{}})
        group['typed_reports'][prior['owner']]={'ranking':copy.deepcopy(raw['ranking']),
            'claims':[c['registered'] for c in registered if c['registered'] is not None],'updates':[]}
        group['raw_reports'][prior['owner']]=raw
        group['unregistered'] += [{'owner':prior['owner'],**copy.deepcopy(c)} for c in registered if c['registered'] is None]
    states=[]
    for (actor,cid),group in groups.items():
        state=StateD2(group['typed_reports'],group['registry'])
        assert not state.payload()['source_supported_claims'] and not state.payload()['hard_exclusions']
        entry={'actor':actor,'id':cid,'registry_hash':group['registry'].integrity,'raw_initial':group['raw_reports'],
               'typed_initial':group['typed_reports'],'unregistered_originals':group['unregistered'],'state':state.payload(),
               'public_parsing_is_not_truth_verification':True}
        entry['integrity']=fingerprint(entry);states.append(entry)
    summary={}
    for actor in ('glm_flash','qwen_flash'):
        selected=[r for r in reports if r['actor']==actor];claims=[c for r in selected for c in r['registrations']]
        summary[actor]={'reports':len(selected),'raw_claims':len(claims),'registered':sum(c['registered'] is not None for c in claims),
             'unregistered':sum(c['registered'] is None for c in claims),'complete_claim_accounting':True,
             'kinds':dict(Counter(c['registered']['claim_kind'] for c in claims if c['registered'] is not None))}
    result={'exploratory':True,'source':str(source),'source_sha256':sha(source/'summary.json'),'summary':summary,
       'groups':len(states),'states':states,'records':reports,'new_api_calls':0,'semantic_truth_validated':False,
       'recommendation_gain_validated':False,'all_unregistered_originals_retained':True,
       'limits':['development prompt iteration and mixed cached parser versions','one producer report was regenerated',
                 'one timed-out original Qwen request has separate billing-risk record',
                 'six initial states are not completed five-condition trajectories']}
    create(output/'assembly.json',result)
    return {k:result[k] for k in ('summary','groups','new_api_calls')}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True);a=p.parse_args()
    print(json.dumps(run(a.output,a.source),ensure_ascii=False,indent=2))
