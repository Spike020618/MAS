"""Zero-call D2 regression against real rejected reports and saved rankings."""
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.client_v5 import parse_raw
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.output_adapter_d2 import adapt
from experiments.paired_deliberation.state_d2 import StateD2
from experiments.paired_deliberation.verification_d2 import VerificationD2
from experiments.paired_deliberation.contracts_v6 import check_ranking

SOURCE=ROOT/'experiments/results/paired_deliberation_domestic_screen_r1'


def replay(output):
    output=Path(output)
    inputs={r['id']:r for r in json.loads((SOURCE/'inputs.json').read_text())}
    predictions=[];rankings={};leaks=[]
    for provider in ('glm_flash','qwen_flash'):
        rows=[]
        for path in sorted((SOURCE/provider/'rankings').glob('*.json')):
            old=json.loads(path.read_text());cid,condition=old['id'],old['condition'];item=inputs[cid]
            raw_path=SOURCE/provider/'slots'/(cid.replace(':','_')+'_'+condition+'.raw.json')
            saved=json.loads(raw_path.read_text())
            if saved['integrity']!=fingerprint(saved['raw']):raise ValueError('raw response changed')
            record,event=adapt(parse_raw(saved['raw'],saved['request']),saved['request'])
            try:check_ranking(record['report'],item['evidence']);valid=True
            except (ValueError,KeyError,TypeError):valid=False
            rows.append({'id':cid,'condition':condition,'original_valid':old['valid'],'d2_representation_valid':valid,
                         'raw_receipt_sha256':sha(raw_path),'adaptation':event})
        rankings[provider]={'n':len(rows),'valid':sum(r['d2_representation_valid'] for r in rows),'rows':rows}
    for path in sorted((SOURCE/'qwen_flash/reports').glob('*.json')):
        old=json.loads(path.read_text())
        if old['valid']:continue
        item=inputs[old['id']];registry=EvidenceRegistry(item['evidence'],item['training_view'])
        for legacy in old['response']['report']['claims']:
            if legacy['claim_type'] not in {'preference_hypothesis','recommendation_prediction'}:continue
            if not any(s.startswith('redial:') for s in legacy['source_ids']):continue
            claim=registry.migrate_legacy(legacy)
            one={'ranking':old['response']['report']['ranking'],'claims':[claim],'updates':[]}
            state=StateD2({old['owner']:one},registry)
            target=old['owner']+':'+claim['claim_id']
            check=state.observe('central',target,fingerprint(state.claims[target]),VerificationD2(registry))['verification']
            # Keep the same historical references, but attempt to grant them
            # factual authority. This counterexample MUST be rejected.
            forbidden=copy.deepcopy(claim)
            forbidden.update(claim_kind='current_preference',scope={'scope':'preference_state','subject':'seeker',
                'predicate':'wants','object':'Horror','polarity':True,'time':'current'})
            forbidden['evidence_refs']=[{'source_id':r['source_id'],'use':'supports'} for r in claim['evidence_refs']]
            denied=False
            try:registry.validate_claim(forbidden)
            except ValueError:denied=True
            predictions.append({'owner':old['owner'],'claim_id':legacy['claim_id'],'source_report_sha256':sha(path),
                'original_report_valid':False,'d2_prediction_basis_accepted':check['verdict']=='not_evaluated',
                'registered_canonical_refs':claim['evidence_refs'],'historical_fact_promotion_rejected':denied,
                'factual_projection_empty':not state.payload()['source_supported_claims'],
                'hard_exclusions_empty':not state.payload()['hard_exclusions']})
            if registry.dialogue().keys() & {r['source_id'] for r in claim['evidence_refs']}:leaks.append(target)
    assert predictions and all(r['d2_prediction_basis_accepted'] and r['historical_fact_promotion_rejected'] and
        r['factual_projection_empty'] and r['hard_exclusions_empty'] for r in predictions)
    report={'version':'d2_contract_offline_replay_r1','new_model_calls':0,'runtime_labels_read':False,
        'reserved_data_read':False,'online_release':False,'historical_results_unchanged':True,
        'prediction_claims':predictions,'prediction_claim_count':len(predictions),
        'historical_to_first_party_leaks':leaks,'ranking_representation':rankings,
        'source_score_sha256':sha(SOURCE/'score.json'),
        'source_hashes':{n:sha(Path(__file__).with_name(n)) for n in
          ('evidence_d2.py','output_adapter_d2.py','state_d2.py','verification_d2.py','replay_d2_contracts.py')},
        'limits':['legal predictive reference does not prove claim text, frequency or user preference true',
                  'only prediction claims migrated; no semantic purpose guessed for legacy explicit claims',
                  'representation replay is offline and does not upgrade original online scores',
                  'no source-level semantic accuracy or recommendation-gain validation']}
    create(output/'contract_replay.json',report)
    return {'prediction_claims_accepted':len(predictions),'fact_promotions_rejected':sum(r['historical_fact_promotion_rejected'] for r in predictions),
            'rankings':{name:{'n':row['n'],'valid':row['valid']} for name,row in rankings.items()},'new_model_calls':0}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(replay(p.parse_args().output),ensure_ascii=False,indent=2))
