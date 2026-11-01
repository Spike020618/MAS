"""Offline representation sensitivity and isolation audit; original scores retained."""
import argparse
import json
from pathlib import Path

from experiments.consensus_recommendation.common import OUTPUT,create,sha
from experiments.paired_deliberation.screen_domestic import verify,VALIDATION
from experiments.paired_deliberation.domestic_transport import PROVIDERS
from experiments.paired_deliberation.client_v5 import parse_raw
from experiments.paired_deliberation.screen_output_projection import project
from experiments.paired_deliberation.contracts_v6 import check_ranking
from experiments.paired_deliberation.atomic_evidence import validate_atom,dialogue_sources
from experiments.paired_deliberation.atomic_relations_v4 import reduce_relations
from experiments.paired_deliberation.pipeline_v5 import source_map
from experiments.paired_deliberation.validate_atomic_evidence import faithful
from experiments.paired_deliberation.evaluate_training_signals import metrics
from experiments.consensus_recommendation.common import read_rows


def run(root):
    root=Path(root);plan=verify(root)
    inputs={r['id']:r for r in json.loads((root/'inputs.json').read_text())}
    semantic_inputs={r['id']:r for r in json.loads((root/'semantic_inputs.json').read_text())}
    if sha(OUTPUT/'data/dev.labels.jsonl')!=plan['expected_dev_labels_sha256']:raise ValueError('labels changed')
    labels={r['case_id']:r['relevant_movie_ids'] for r in read_rows(OUTPUT/'data/dev.labels.jsonl')}
    semantic_labels={r['id']:r for r in json.loads((VALIDATION/'labels.json').read_text())}
    results={}
    for name in PROVIDERS:
        folder=root/name;rankings=[];semantic=[];citation_mismatches=[]
        for path in (folder/'rankings').glob('*.json'):
            original=json.loads(path.read_text());cid=original['id'];condition=original['condition']
            raw=json.loads((folder/'slots'/(cid.replace(':','_')+'_'+condition+'.raw.json')).read_text())
            record=parse_raw(raw['raw'],raw['request'])
            candidate,ops=project(record,raw['request'],inputs[cid]['evidence'])
            try:check_ranking(candidate['report'],inputs[cid]['evidence']);valid=True
            except (ValueError,KeyError,TypeError):valid=False
            rankings.append({'id':cid,'condition':condition,'original_valid':original['valid'],'projected_valid':valid,
                             'operations':ops,'projected_ranking':candidate['report'].get('ranking') if valid else None,
                             'metrics':metrics(candidate['report']['ranking'],labels[cid]) if valid else None})
        for cid in plan['semantic_ids']:
            parsed_path=folder/'slots'/(cid+'_parse.raw.json');judged_path=folder/'slots'/(cid+'_sources.raw.json')
            if not parsed_path.exists():continue
            raw=json.loads(parsed_path.read_text());parsed=parse_raw(raw['raw'],raw['request'])
            candidate,ops=project(parsed,raw['request'])
            row={'id':cid,'parser_operations':ops,'source_response_available':judged_path.exists(),
                 'projected_parser_correct':faithful(candidate['report'],semantic_labels[cid]['expected_atom'])}
            try:
                atom=validate_atom(candidate['report'])
                if judged_path.exists():
                    raw=json.loads(judged_path.read_text());payload=json.loads(raw['request']['messages'][1]['content'])
                    if payload['atom']!={k:v for k,v in atom.items() if k!='reason'}:raise ValueError('source response bound to a different atom')
                    candidate,ops=project(parse_raw(raw['raw'],raw['request']),raw['request'])
                    check=reduce_relations(atom,dialogue_sources(source_map(semantic_inputs[cid]['evidence'])),{**candidate['report'],'usage':candidate['usage']})
                    row.update(source_operations=ops,projected_source_valid=True,
                               projected_verdict_correct=check['verdict']==semantic_labels[cid]['verdict'],
                               projected_relations_correct={r['source_id']:r['relation'] for r in check['source_relations']}==semantic_labels[cid]['relations'])
            except (ValueError,KeyError,TypeError) as exc:row.update(projected_source_valid=False,error=str(exc))
            semantic.append(row)
        for path in (folder/'reports').glob('*.json'):
            row=json.loads(path.read_text())
            if row['valid'] or 'response' not in row:continue
            item=inputs[row['id']]
            known={x['training_case_id'] for x in item['training_view']['historical_examples']}
            training_refs={sid for c in row['response']['report'].get('claims',[]) for sid in c.get('source_ids',[]) if sid in known}
            if training_refs:citation_mismatches.append({'owner':row['owner'],'historical_ids_exist_in_supplied_view':sorted(training_refs),
                'issue':'historical prediction evidence has no registered claim citation channel; not first-party user evidence',
                'report_not_repaired':True})
        ranking_summary={}
        for condition in ('raw_evidence','training_exposed'):
            selected=[r for r in rankings if r['condition']==condition];scorable=[r for r in selected if r['metrics'] is not None]
            ranking_summary[condition]={'projected_valid':sum(r['projected_valid'] for r in selected),'n':len(selected),
                                        'valid_annotated':len(scorable),'hits_on_valid_annotated':sum(r['metrics']['Hit@5'] for r in scorable)}
        pairs=[]
        for cid in plan['case_ids']:
            pair={r['condition']:r for r in rankings if r['id']==cid}
            if len(pair)==2 and all(r['metrics'] is not None for r in pair.values()):
                pairs.append(pair['training_exposed']['metrics']['Hit@5']-pair['raw_evidence']['metrics']['Hit@5'])
        results[name]={'projected_ranking_summary':ranking_summary,
                       'projected_paired_valid_annotated':{'n':len(pairs),'wins':pairs.count(1),'losses':pairs.count(-1),'ties':pairs.count(0)},
                       'projected_parser_correct':sum(r['projected_parser_correct'] for r in semantic),'parser_n':len(semantic),
                       'existing_source_responses':sum(r['source_response_available'] for r in semantic),
                       'historical_citation_channel_mismatches':citation_mismatches,'ranking_details':rankings,'semantic_details':semantic}
    report={'plan_hash':plan['plan_hash'],'kind':'posthoc_label_independent_representation_projection','providers':results,
            'original_score_sha256':sha(root/'score.json'),'new_api_calls':0,'pilot_allowed':False,
            'limits':['posthoc projected diagnostics, original strict scores unchanged',
                      'absent source responses not generated or imputed','no citation or semantic decision repair','not MAS gain'],
            'audit_code_sha256':sha(Path(__file__)),'projection_code_sha256':sha(Path(__file__).with_name('screen_output_projection.py'))}
    create(root/'posthoc_representation_audit.json',report)
    return {name:{k:v for k,v in r.items() if k not in ('ranking_details','semantic_details')} for name,r in results.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    print(json.dumps(run(parser.parse_args().root),ensure_ascii=False,indent=2))
