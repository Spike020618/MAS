"""One JSON connectivity probe per supplied provider; no credential artifacts."""
import json
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.domestic_transport import DomesticTransport,PROVIDERS
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.client_v5 import parse_raw
from experiments.paired_deliberation.requests_v5 import RequestsV5


def run(root):
    root=Path(root)
    request=RequestsV5.envelope('connectivity',{'schema':{'connection':'ok'}},
        'Return exactly this JSON object: {"connection":"ok"}. No other text.',max_tokens=32)
    plan={'version':'domestic_connectivity_r1','providers':PROVIDERS,'maximum_requests':2,
          'request_hash':fingerprint(request),'code_sha256':{p:sha(Path(__file__).with_name(p)) for p in
          ('probe_domestic.py','domestic_transport.py','durable_provider.py','client_v5.py')},
          'keys_in_manifest':False,'maximum_estimated_cny_per_provider':0.01}
    plan['plan_hash']=fingerprint(plan);create(root/'plan.json',plan)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    results={}
    for name in PROVIDERS:
        slots=root/name/'slots'
        transport=DomesticTransport(name,slots,max_attempts=1,max_cny=0.01)
        provider=DurableProvider(slots,plan,transport,parse_raw,True)
        try:
            record=provider.complete('connectivity',request)
            results[name]={'connected':record['report']=={'connection':'ok'},'record':record}
        except (ValueError,RuntimeError,KeyError,TypeError) as exc:
            results[name]={'connected':False,'error':str(exc)}
            path=slots/'connectivity.raw.json'
            if path.exists():
                saved=json.loads(path.read_text());results[name]['http_status']=saved['raw']['status_code']
                try:results[name]['provider_error']=json.loads(saved['raw']['body']).get('error',{})
                except ValueError:pass
        results[name]['pending_requests']=len(list(slots.glob('*.intent.json')))
        print(name+': '+('connected' if results[name]['connected'] else 'failed'),flush=True)
    create(root/'probe_result.json',results)
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    print(json.dumps(run(parser.parse_args().root),ensure_ascii=False,indent=2))
