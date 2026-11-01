"""Post-collection isolation and exact metadata projection; never changes release."""
import argparse
import json
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint, sha
from experiments.paired_deliberation.validate_commit_v4 import verify_plan
from experiments.paired_deliberation.client_v5 import parse_raw
from experiments.paired_deliberation.response_format_metadata import project
from experiments.paired_deliberation.atomic_relations_v4 import reduce_relations
from experiments.paired_deliberation.atomic_evidence import dialogue_sources
from experiments.paired_deliberation.pipeline_v5 import source_map


def run(root):
    root=Path(root);plan=verify_plan(root);score=json.loads((root/'release_score.json').read_text())
    if score['plan_hash']!=plan['plan_hash']:raise ValueError('score mismatch')
    inputs={r['id']:r for r in json.loads((root/'inputs.json').read_text())}
    labels={r['id']:r for r in json.loads((root/'labels.json').read_text())}
    if fingerprint(list(labels.values()))!=plan['labels_hash']:raise ValueError('labels changed')
    forbidden={'expected_atom','expected_verdict','labels','label','gold','corrupted','compatible_ids'}
    leaks=[];requests=[]
    def visit(value,path):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in forbidden:leaks.append(path+'/'+key)
                visit(item,path+'/'+key)
        elif isinstance(value,list):
            for i,item in enumerate(value):visit(item,path+'/'+str(i))
    for path in sorted((root/'slots').glob('*.raw.json')):
        saved=json.loads(path.read_text())
        if saved['integrity']!=fingerprint(saved['raw']):raise ValueError('raw receipt changed')
        request=saved['request'];payload=json.loads(request['messages'][1]['content'])
        visit(payload,path.name);requests.append(request['kind'])
        if request['kind']=='atomic_interpretation' and set(payload)!={'original_claim','schema'}:
            raise ValueError('parser received unexpected information')
    recoveries=[]
    for row in score['details']:
        if not row['verification'].get('verification_failed'):continue
        path=root/'slots'/(row['id']+'_sources.raw.json')
        if not path.exists():continue
        saved=json.loads(path.read_text());parsed=parse_raw(saved['raw'],saved['request'])
        candidate,changed=project(parsed,saved['request'])
        if changed:
            sources=dialogue_sources(source_map(inputs[row['id']]['evidence']))
            result=reduce_relations(row['verification']['atom'],sources,{**candidate['report'],'usage':candidate['usage']})
            recoveries.append({'id':row['id'],'source_receipt_sha256':sha(path),'operation':'remove exact schema=json_object transport marker',
                               'projected_verdict':result['verdict'],'verdict_matches_label':result['verdict']==labels[row['id']]['verdict'],
                               'relations_match_label':{r['source_id']:r['relation'] for r in result['source_relations']}==labels[row['id']]['relations']})
    result={'kind':'posthoc_commit_validation_audit','plan_hash':plan['plan_hash'],
            'release_score_sha256':sha(root/'release_score.json'),'physical':score['physical'],
            'request_kinds':{kind:requests.count(kind) for kind in sorted(set(requests))},
            'structured_label_leaks':leaks,'parser_evidence_isolated':True,
            'original_parser_correct':sum(r['atom_correct'] for r in score['details']),
            'original_verdict_correct':sum(r['verification']['verdict']==r['expected_verdict'] for r in score['details']),
            'original_joint_correct':sum(r['correct'] for r in score['details']),
            'false_support':sum(r['verification']['verdict']=='supported' and r['expected_verdict']!='supported' for r in score['details']),
            'false_refutation':sum(r['verification']['verdict']=='refuted' and r['expected_verdict']!='refuted' for r in score['details']),
            'metadata_projection_recoveries':recoveries,'original_release_pass':score['pass'],
            'online_release':False,'new_api_calls':0,'reserved_files_read':False,
            'limits':['projection is posthoc and is not an online calibration score',
                      'same-source opposition omission remains unresolved; original failed gate retained'],
            'audit_code_sha256':sha(Path(__file__))}
    create(root/'posthoc_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    print(json.dumps(run(parser.parse_args().root),ensure_ascii=False,indent=2))
