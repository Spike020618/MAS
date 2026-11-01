"""Resume exploratory v2; at most one logged retry of each uncertain cheap call."""
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.preflight_d2 import ACTORS
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.transport_d2 import TransportD2,parse_d2_raw
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.explore_light_actor import ProjectClient,request,check_raw,register
from experiments.paired_deliberation import explore_light_parser as original


def run(output,source):
    output,source=Path(output),Path(source)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    previous=json.loads((source/'summary.json').read_text())
    contexts={r['id']:r for r in json.loads((ROOT/'experiments/results/paired_deliberation_d2_engineering_r1/inputs.json').read_text())}
    meta={'exploratory':True,'variant':'public-parser-v2','code_sha256':sha(Path(original.__file__)),
          'source_summary_sha256':sha(source/'summary.json'),'not_a_confirmatory_study':True}
    create(output/'original_run_metadata.json',meta)
    providers={name:DurableProvider(output/name/'slots',meta,TransportD2(name,profile,output/name/'slots',200,5.0),parse_d2_raw,True)
               for name,profile in ACTORS.items()}
    def complete(name,slot,req):
        provider=providers[name];intent=provider.root/(slot+'.intent.json');raw=provider.root/(slot+'.raw.json')
        if intent.exists() and not raw.exists():
            retry=slot+'__uncertain_retry1'
            create(output/'recovery'/(slot+'.json'),{'original_slot':slot,'retry_slot':retry,'provider':name,
                'original_result_and_billing_unknown':True,'possible_duplicate_billing':True,
                'authorization':'user permits freely using cheap GLM/Qwen for exploratory debugging',
                'original_intent_retained':True,'max_retries_for_this_uncertain_slot':1})
            slot=retry
        return ProjectClient(provider).complete(slot,req)
    results=[]
    for old in previous['records']:
        prefix=old['actor']+'_'+old['id'].replace(':','_')+'_'+old['owner'];path=output/'records'/(prefix+'.json')
        if path.exists():results.append(json.loads(path.read_text()));continue
        item=contexts[old['id']];reg=EvidenceRegistry(item['evidence'],item['training_view'])
        row={'actor':old['actor'],'id':old['id'],'owner':old['owner'],'source_record':prefix,
             'source_v1_structure_valid':old['structure_valid'],'raw':old.get('raw'),'origin':'same_saved_v1_report'}
        if not old['structure_valid']:
            req=request(reg,old['owner']);req['messages'][0]['content']+=' candidate_ids must be copied ID STRINGS from the shared candidate list, never titles or outside IDs. If affected candidates are unclear, use [].'
            answer,event=complete(old['actor'],prefix+'_producer_retry',req)
            row.update(raw=answer['report'],origin='one_producer_retry_v2',producer_adaptation=event)
        try:check_raw(row['raw'],reg);row['structure_valid']=True
        except (ValueError,KeyError,TypeError) as exc:
            row.update(structure_valid=False,error=str(exc));results.append(row);create(path,row);continue
        req=original.request_v2(row['raw']['claims']);row['parser_prompt_hash']=fingerprint(req)
        answer,event=complete('qwen_flash',prefix+'_parse_v2',req)
        row.update(public_parser=answer['report'],parser_adaptation=event)
        try:row.update(registrations=register(row['raw'],answer['report'],reg),parser_valid=True)
        except (ValueError,KeyError,TypeError) as exc:row.update(parser_valid=False,parser_error=str(exc))
        results.append(row);create(path,row);print('resumed '+prefix+' parser='+str(row['parser_valid']),flush=True)
    summary={}
    for name,profile in ACTORS.items():
        rows=[r for r in results if r['actor']==name];regs=[x for r in rows for x in r.get('registrations',[])]
        raws=[json.loads(p.read_text()) for p in (output/name/'slots').glob('*.raw.json')];bodies=[json.loads(r['raw']['body']) for r in raws if r['raw']['status_code']==200]
        prompt=sum(b.get('usage',{}).get('prompt_tokens',0) for b in bodies);completion=sum(b.get('usage',{}).get('completion_tokens',0) for b in bodies)
        summary[name]={'reports':len(rows),'structure_valid':sum(r.get('structure_valid',False) for r in rows),
            'public_parser_valid':sum(r.get('parser_valid',False) for r in rows),'registered_claims':sum(x['registered'] is not None for x in regs),
            'unsupported_or_authority_failed_claims':sum(x['registered'] is None for x in regs),
            'successful_responses_including_parser':len(raws),'unresolved_original_intents':len(list((output/name/'slots').glob('*.intent.json'))),
            'known_prompt_tokens':prompt,'known_completion_tokens':completion,
            'estimated_known_list_price_cny':(prompt*profile['input_cny_per_million']+completion*profile['output_cny_per_million'])/1_000_000,
            'actual_bill_and_timed_out_usage_unknown':True}
    report={'exploratory':True,'variant':'public-parser-v2','source':str(source),'summary':summary,'records':results,
        'retry_records':[json.loads(p.read_text()) for p in (output/'recovery').glob('*.json')],
        'semantic_truth_validated':False,'recommendation_gain_validated':False,'reserved_data_read':False,
        'limits':['development prompt iteration, not independent validation','one producer retry explicitly identified',
                  'one possible duplicate bill from timed-out original retained','registration is not truth verification']}
    create(output/'summary.json',report)
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True);args=p.parse_args()
    print(json.dumps(run(args.output,args.source),ensure_ascii=False,indent=2))
