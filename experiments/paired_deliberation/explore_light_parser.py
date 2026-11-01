"""Unfrozen cheap-model debugging: parser v2 on saved light reports, one producer retry."""
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.preflight_d2 import ACTORS
from experiments.paired_deliberation.transport_d2 import TransportD2,parse_d2_raw
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.explore_light_actor import ProjectClient,request,parser_request,check_raw,register


def request_v2(claims):
    req=parser_request(claims)
    req['messages'][0]['content']+=(
        ' Every row MUST contain exactly all THREE keys claim_id, kind, scope. '
        'kind is mandatory even when scope is null. Never omit kind. '
        'Only seeker wants/likes meanings fit the two dialogue kinds. Other predicates such as '
        'expressed interest or discussed are unsupported unless the original text actually asserts wants/likes. '
        'Do not translate interest into liking to satisfy the schema. Unsupported uses kind=unsupported, scope=null.')
    return req


def run(output,source):
    output,source=Path(output),Path(source)
    previous=json.loads((source/'summary.json').read_text())
    contexts={r['id']:r for r in json.loads((ROOT/'experiments/results/paired_deliberation_d2_engineering_r1/inputs.json').read_text())}
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    meta={'exploratory':True,'variant':'public-parser-v2','code_sha256':sha(Path(__file__)),
          'source_summary_sha256':sha(source/'summary.json'),'not_a_confirmatory_study':True}
    pp={name:ProjectClient(DurableProvider(output/name/'slots',meta,
        TransportD2(name,profile,output/name/'slots',200,5.0),parse_d2_raw,True)) for name,profile in ACTORS.items()}
    results=[]
    for old in previous['records']:
        item=contexts[old['id']];reg=EvidenceRegistry(item['evidence'],item['training_view'])
        prefix=old['actor']+'_'+old['id'].replace(':','_')+'_'+old['owner']
        row={'actor':old['actor'],'id':old['id'],'owner':old['owner'],'source_record':prefix,
             'source_v1_structure_valid':old['structure_valid'],'raw':copy.deepcopy(old.get('raw')),'origin':'same_saved_v1_report'}
        if not old['structure_valid']:
            retry=request(reg,old['owner'])
            retry['messages'][0]['content']+=' candidate_ids must be copied ID STRINGS from the shared candidate list, never titles or outside IDs. If affected candidates are unclear, use [].'
            answer,event=pp[old['actor']].complete(prefix+'_producer_retry',retry)
            row.update(raw=answer['report'],origin='one_producer_retry_v2',producer_adaptation=event)
        try:check_raw(row['raw'],reg);row['structure_valid']=True
        except (ValueError,KeyError,TypeError) as exc:
            row.update(structure_valid=False,error=str(exc));results.append(row);create(output/'records'/(prefix+'.json'),row)
            print(prefix+' producer still invalid: '+str(exc),flush=True);continue
        req=request_v2(row['raw']['claims']);row['parser_prompt_hash']=fingerprint(req)
        parsed,event=pp['qwen_flash'].complete(prefix+'_parse_v2',req)
        row.update(public_parser=parsed['report'],parser_adaptation=event)
        try:
            row['registrations']=register(row['raw'],parsed['report'],reg);row['parser_valid']=True
        except (ValueError,KeyError,TypeError) as exc:row.update(parser_valid=False,parser_error=str(exc))
        results.append(row);create(output/'records'/(prefix+'.json'),row)
        print(prefix+' parser='+str(row['parser_valid'])+' registered='+str(sum(x['registered'] is not None for x in row.get('registrations',[]))),flush=True)
    summary={}
    for name,profile in ACTORS.items():
        rows=[r for r in results if r['actor']==name];regs=[x for r in rows for x in r.get('registrations',[])]
        raws=[json.loads(p.read_text()) for p in (output/name/'slots').glob('*.raw.json')];bodies=[json.loads(r['raw']['body']) for r in raws if r['raw']['status_code']==200]
        prompt=sum(b.get('usage',{}).get('prompt_tokens',0) for b in bodies);completion=sum(b.get('usage',{}).get('completion_tokens',0) for b in bodies)
        summary[name]={'reports':len(rows),'structure_valid':sum(r.get('structure_valid',False) for r in rows),
            'public_parser_valid':sum(r.get('parser_valid',False) for r in rows),'registered_claims':sum(x['registered'] is not None for x in regs),
            'unsupported_or_authority_failed_claims':sum(x['registered'] is None for x in regs),
            'model_calls_including_shared_parser_work':len(raws),'prompt_tokens':prompt,'completion_tokens':completion,
            'estimated_list_price_cny':(prompt*profile['input_cny_per_million']+completion*profile['output_cny_per_million'])/1_000_000,
            'pending':len(list((output/name/'slots').glob('*.intent.json')))}
    create(output/'summary.json',{'exploratory':True,'variant':'public-parser-v2','source':str(source),'summary':summary,'records':results,
        'semantic_truth_validated':False,'recommendation_gain_validated':False,'reserved_data_read':False,
        'limits':['prompt iterated on observed development errors; not independent validation',
                  '17 old reports plus one explicit producer retry, not a paired comparison of two new actors',
                  'public parsing/registration is not factual verification','unsupported originals retained without semantic repair']})
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True);args=p.parse_args()
    print(json.dumps(run(args.output,args.source),ensure_ascii=False,indent=2))
