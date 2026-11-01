"""Exploratory v3: reuse complete parser rows, single-claim calls for missing rows."""
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.preflight_d2 import ACTORS
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.transport_d2 import TransportD2,parse_d2_raw
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.explore_light_actor import ProjectClient,register
from experiments.paired_deliberation.explore_light_parser import request_v2


def complete_rows(report,wanted):
    rows=report.get('claims',[]) if isinstance(report,dict) else []
    if isinstance(report,dict) and set(report)=={'claim_id','kind','scope'}:rows=[report]
    if not isinstance(rows,list):return {}
    counts={cid:sum(isinstance(r,dict) and r.get('claim_id')==cid for r in rows) for cid in wanted}
    return {r['claim_id']:copy.deepcopy(r) for r in rows if isinstance(r,dict) and set(r)=={'claim_id','kind','scope'}
            and r['claim_id'] in wanted and counts[r['claim_id']]==1}


def one_request(text):
    original=request_v2([{'text':text}])
    schema=json.loads(original['messages'][1]['content'])['schema']['claims'][0]
    system=original['messages'][0]['content']
    system+=(' This call contains exactly ONE original claim. Return exactly kind and scope at the ROOT. '
             'Do not add claim_id or wrap in claims, schema or answer. The runtime assigns IDs. '
             'Kind is always required; scope is null for non-dialogue and unsupported kinds.')
    return RequestsV5.envelope('register_single_light',{'original_claim':{'text':text},
        'schema':{'kind':schema['kind'],'scope':schema['scope']}},system,max_tokens=512)


def run(output,source):
    output,source=Path(output),Path(source);old=json.loads((source/'summary.json').read_text())
    contexts={r['id']:r for r in json.loads((ROOT/'experiments/results/paired_deliberation_d2_engineering_r1/inputs.json').read_text())}
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    profile=ACTORS['qwen_flash'];meta={'exploratory':True,'variant':'single-claim-register-v3','source_sha256':sha(source/'summary.json'),'code_sha256':sha(Path(__file__))}
    provider=DurableProvider(output/'qwen_flash/slots',meta,TransportD2('qwen_flash',profile,output/'qwen_flash/slots',100,2.0),parse_d2_raw,True)
    client=ProjectClient(provider);records=[]
    for prev in old['records']:
        prefix=prev['actor']+'_'+prev['id'].replace(':','_')+'_'+prev['owner'];raw=prev['raw']
        item=contexts[prev['id']];reg=EvidenceRegistry(item['evidence'],item['training_view'])
        wanted=['c'+str(i+1) for i in range(len(raw['claims']))]
        parsed=complete_rows(prev.get('public_parser'),wanted);origins={cid:'complete_saved_v2_row' for cid in parsed}
        failures=[]
        for i,cid in enumerate(wanted):
            if cid in parsed:continue
            req=one_request(raw['claims'][i]['text']);record,event=client.complete(prefix+'_'+cid+'_single',req)
            obj=record['report']
            if not isinstance(obj,dict) or set(obj)!={'kind','scope'}:failures.append({'claim_id':cid,'error':'single parser root fields','response':obj});continue
            parsed[cid]={'claim_id':cid,**obj};origins[cid]='new_single_claim_v3'
        row={'actor':prev['actor'],'id':prev['id'],'owner':prev['owner'],'raw':raw,'structure_valid':prev['structure_valid'],
             'parsed_rows':[parsed[cid] for cid in wanted if cid in parsed],'parse_origins':origins,'failures':failures}
        if not failures and len(parsed)==len(wanted):
            try:row.update(registrations=register(raw,{'claims':row['parsed_rows']},reg),parser_structure_valid=True)
            except (ValueError,KeyError,TypeError) as exc:row.update(parser_structure_valid=False,error=str(exc))
        else:row['parser_structure_valid']=False
        records.append(row);create(output/'records'/(prefix+'.json'),row)
        print(prefix+' parse='+str(row['parser_structure_valid'])+' registered='+str(sum(r['registered'] is not None for r in row.get('registrations',[]))),flush=True)
    summary={}
    for actor in ACTORS:
        rows=[r for r in records if r['actor']==actor];regs=[c for r in rows for c in r.get('registrations',[])]
        summary[actor]={'reports':len(rows),'structure_valid':sum(r['structure_valid'] for r in rows),
             'public_parser_structure_valid':sum(r['parser_structure_valid'] for r in rows),
             'raw_claims':sum(len(r['raw']['claims']) for r in rows),'registered_claims':sum(r['registered'] is not None for r in regs),
             'unregistered_claims':sum(r['registered'] is None for r in regs),
             'reused_parse_rows':sum(v=='complete_saved_v2_row' for r in rows for v in r['parse_origins'].values()),
             'new_parse_rows':sum(v=='new_single_claim_v3' for r in rows for v in r['parse_origins'].values())}
    raws=[json.loads(p.read_text()) for p in (output/'qwen_flash/slots').glob('*.raw.json')]
    bodies=[json.loads(r['raw']['body']) for r in raws];prompt=sum(b['usage']['prompt_tokens'] for b in bodies);completion=sum(b['usage']['completion_tokens'] for b in bodies)
    physical={'qwen_requests':len(raws),'prompt_tokens':prompt,'completion_tokens':completion,
              'estimated_list_price_cny':(prompt*.15+completion*1.5)/1_000_000,
              'pending':len(list((output/'qwen_flash/slots').glob('*.intent.json')))}
    create(output/'summary.json',{'exploratory':True,'variant':'single-claim-register-v3','summary':summary,'physical':physical,'records':records,
        'source':str(source),'semantic_truth_validated':False,'recommendation_gain_validated':False,'reserved_data_read':False,
        'limits':['reuses complete rows without selecting by truth or outcome','supplements only missing parser fields via new calls; no guessed filling',
                  'different parser prompt versions on development reports; not independent validation','all original unregistered claims retained']})
    return {'summary':summary,'physical':physical}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True);args=p.parse_args()
    print(json.dumps(run(args.output,args.source),ensure_ascii=False,indent=2))
