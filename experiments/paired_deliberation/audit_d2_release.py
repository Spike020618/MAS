"""Close D2 release engineering attempts; preserve raw and normalized outcomes."""
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.client_v5 import parse_raw
from experiments.paired_deliberation.output_adapter_d2 import adapt
from experiments.paired_deliberation.source_alias_d2 import canonicalize
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry


def audit(output):
    output=Path(output);base=ROOT/'experiments/results'
    roots=[base/('paired_deliberation_d2_purpose_validation_r'+str(i)) for i in range(1,6)]
    plans=[json.loads((r/'plan.json').read_text()) for r in roots]
    assert len({p['inputs_hash'] for p in plans})==1 and len({p['labels_hash'] for p in plans})==1
    assert len({fingerprint(p['thresholds']) for p in plans})==1
    physical=[];pending=0
    for root in roots:
        pending+=len(list((root/'slots').glob('*.intent.json')))
        for path in (root/'slots').glob('*.raw.json'):
            saved=json.loads(path.read_text())
            assert saved['integrity']==fingerprint(saved['raw'])
            if 'reused_from' not in saved:physical.append((path,saved))
    normalized=roots[-1];alias_ops=0;alias_rows=[]
    for path in (normalized/'slots').glob('*.raw.json'):
        saved=json.loads(path.read_text());record=parse_raw(saved['raw'],saved['request'])
        shaped,_=adapt(record,saved['request']);changed,ops=canonicalize(shaped,saved['request'])
        if ops:
            field='relations' if saved['request']['kind']=='atomic_source_relations' else 'witnesses'
            before,after=copy.deepcopy(shaped['report']),copy.deepcopy(changed['report'])
            for row in before[field]:row.pop('source_id',None)
            for row in after[field]:row.pop('source_id',None)
            assert before==after,'alias projection altered semantic content'
            alias_ops+=len(ops);alias_rows.append({'slot':path.name,'operations':ops,'raw_receipt_sha256':sha(path)})
    engineering=base/'paired_deliberation_d2_engineering_r1'
    engineering_score=json.loads((engineering/'engineering_acceptance.json').read_text())
    items={x['id']:x for x in json.loads((engineering/'inputs.json').read_text())};producer_failures=[]
    for actor in ('glm_flash','qwen_flash'):
        for path in (engineering/'slots'/actor).glob('*.raw.json'):
            saved=json.loads(path.read_text());reg_payload=json.loads(saved['request']['messages'][1]['content'])
            item=items[reg_payload['evidence']['case_id']];reg=EvidenceRegistry(item['evidence'],item['training_view'])
            obj=parse_raw(saved['raw'],saved['request'])['report'];report_error=None
            try:reg.validate_report(obj)
            except (ValueError,KeyError,TypeError) as exc:report_error=str(exc)
            claims=[]
            for claim in obj.get('claims',[]):
                error=None
                try:reg.validate_claim(claim)
                except (ValueError,KeyError,TypeError) as exc:error=str(exc)
                claims.append({'claim_id':claim.get('claim_id'),'error':error})
            producer_failures.append({'actor':actor,'slot':path.name,'report_error':report_error,'claim_count':len(obj.get('claims',[])),
                                      'claims':claims,'raw_receipt_sha256':sha(path)})
    successful=[json.loads(saved['raw']['body']) for _,saved in physical if saved['raw']['status_code']==200]
    score=json.loads((normalized/'release_score.json').read_text())
    report={'kind':'d2_purpose_and_real_engineering_closure','inputs_labels_thresholds_unchanged':True,
        'calibration_physical_attempts':len(physical),'calibration_successful_responses':len(successful),
        'calibration_http_failures':len(physical)-len(successful),'calibration_pending':pending,
        'calibration_known_prompt_tokens':sum(b['usage']['prompt_tokens'] for b in successful),
        'calibration_known_completion_tokens':sum(b['usage']['completion_tokens'] for b in successful),
        'failed_http_usage_unknown_not_zero':True,'normalized_calibration':{k:v for k,v in score.items() if k!='details'},
        'source_alias_operations':alias_ops,'source_alias_semantics_unchanged':True,'source_alias_rows':alias_rows,
        'engineering':{k:v for k,v in engineering_score.items() if k!='initial_failures'},'producer_failures':producer_failures,
        'total_new_physical_attempts':len(physical)+sum(r['responses'] for r in engineering_score['physical'].values()),
        'pilot_allowed':False,'new_recommendation_effect_validated':False,
        'normalization_is_posthoc':True,'sampling_and_thresholds_not_tuned':True,
        'limits':['new semantic labels are mostly current wants for four genres, not general natural preference coverage',
                  '144 paid responses normalized offline; no independent post-normalization semantic recollection',
                  'false support/refutation remain; purpose pass permits only small engineering',
                  'both actors stopped at first initial report; 0 of 30 method paths completed'],
        'audit_code_sha256':sha(Path(__file__))}
    create(output,report)
    return {k:v for k,v in report.items() if k not in ('normalized_calibration','source_alias_rows','producer_failures','engineering')}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(audit(p.parse_args().output),ensure_ascii=False,indent=2))
