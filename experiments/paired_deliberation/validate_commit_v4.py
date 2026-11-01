"""One frozen joint parser/witness batch, with no automatic retry or expansion."""
import argparse
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.prepare_commit_calibration import cases
from experiments.paired_deliberation.validate_atomic_evidence import regressions, faithful
from experiments.paired_deliberation.atomic_relations_v4 import AtomicEvidencePipeline
from experiments.paired_deliberation.evidence_commit_state import EvidenceCommitState
from experiments.paired_deliberation.pipeline_v5 import source_map
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw
from experiments.paired_deliberation.schema_echo_client import SchemaEchoClient


def prepare(root):
    root = Path(root)
    inputs, labels = [], []
    for row in cases():
        evidence = {'case_id': row['id'], 'observations': [dict(source_id=sid, **source) for sid, source in row['sources'].items()],
                    'previous_mentions_excluded_not_asserted_seen': [],
                    'candidates_in_shared_retrieval_order': [{'source_id': 'movie:' + str(i), 'movie_id': str(i),
                        'title': 'Control ' + str(i), 'year': 2000, 'genres': []} for i in range(1, 6)]}
        sentinel = {'claim_id': 'sentinel', 'text': 'Movie 5 has catalog year 2000.', 'source_ids': ['movie:5'],
                    'candidate_ids': ['5'], 'stance': 'support', 'claim_type': 'item_fact', 'depends_on': []}
        inputs.append({'id': row['id'], 'evidence': evidence, 'reports': {'current_request': {
            'ranking': [str(i) for i in range(1, 6)], 'claims': [row['claim'], sentinel], 'updates': []}}})
        labels.append({'id': row['id'], 'group': 'new', 'expected_atom': {k:v for k,v in row['expected_atom'].items() if k != 'reason'},
                       'verdict': row['expected_verdict'], 'relations': {r['source_id']:r['relation'] for r in row['expected_relations']['relations']}})
    old = regressions()
    for row in old:
        inputs.append({'id': row['id'], **row['input']})
        labels.append({'id': row['id'], 'group': 'regression', 'expected_atom': row['expected_atom'],
                       'verdict': row['label'], 'relations': row['relations']})
    names = ('validate_commit_v4.py', 'prepare_commit_calibration.py', 'validate_atomic_evidence.py',
             'atomic_relations_v4.py', 'atomic_evidence.py', 'evidence_commit_state.py', 'state_v6.py',
             'contracts_v6.py', 'pipeline_v5.py', 'verified_state.py', 'budget_v6.py', 'durable_provider.py',
             'client_v5.py', 'requests_v5.py', 'schema_echo_client.py', 'commit_engine.py')
    paths = [Path(__file__).with_name(n) for n in names] + [ROOT/'experiments/consensus_recommendation/common.py']
    maximum = 2 * len(inputs)
    plan = {'version': 'commit_v4_joint_validation_r1', 'contract_version':'commit_v4', 'model':'deepseek-chat',
            'new_cases':48, 'regressions':len(old), 'maximum_requests':maximum,
            'inputs_hash':fingerprint(inputs), 'labels_hash':fingerprint(labels),
            'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
            'limits':{'max_calls':maximum, 'per_call_prompt_tokens':12000, 'per_call_completion_tokens':4096,
                      'total_prompt_tokens':maximum*12000, 'total_completion_tokens':maximum*4096},
            'gates':{'new_joint_errors_max':0,'regression_joint_errors_max':0,'invalid_max':0,'sentinel_changed_max':0},
            'pilot_allowed':False, 'runtime_labels_available':False,
            'scope':'48 constructed object/polarity cases plus prior development failures; joint original parsing and source witnesses, not independent natural dialogue accuracy',
            'stop':'one fixed batch; no automatic retry, prompt change or sample expansion'}
    plan['plan_hash'] = fingerprint(plan)
    create(root/'inputs.json',inputs);create(root/'labels.json',labels);create(root/'plan.json',plan)
    for p in paths:
        dst=root/'source_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and dst.read_bytes()!=p.read_bytes():raise ValueError('snapshot changed')
        dst.write_bytes(p.read_bytes())
    return plan


def verify_plan(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    if plan['plan_hash']!=fingerprint({k:v for k,v in plan.items() if k!='plan_hash'}):raise ValueError('plan changed')
    for name,wanted in plan['code_sha256'].items():
        if sha(ROOT/name)!=wanted or sha(root/'source_snapshot'/name)!=wanted:raise ValueError('source changed: '+name)
    if fingerprint(json.loads((root/'inputs.json').read_text()))!=plan['inputs_hash']:raise ValueError('inputs changed')
    return plan


def infer(row,client):
    state=EvidenceCommitState(row['reports'],row['evidence']);target='current_request:c1'
    sentinel=copy.deepcopy(state.claims['current_request:sentinel'])
    pipe=AtomicEvidencePipeline(source_map(row['evidence']),client)
    try:check=pipe.verify(state.claims[target]['claim'],row['id'])
    except (ValueError,KeyError,TypeError) as exc:
        check={'verdict':'insufficient','kind':'invalid_proxy','verification_failed':True,'atom':pipe.last_atom,'reason':str(exc)}
    class Recorded:
        def verify(self,claim):return copy.deepcopy(check)
    state.observe('current_request',target,fingerprint(state.claims[target]),Recorded())
    return {'verification':check,'target_status':state.claims[target]['status'],
            'sentinel_preserved':state.claims['current_request:sentinel']==sentinel}


def score(root,plan,usage,echoes):
    root=Path(root);labels=json.loads((root/'labels.json').read_text())
    if fingerprint(labels)!=plan['labels_hash']:raise ValueError('labels changed')
    details=[];counts={'new_joint_errors':0,'regression_joint_errors':0,'invalid':0,'sentinel_changed':0}
    for label in labels:
        pred=json.loads((root/'predictions'/(label['id']+'.json')).read_text());check=pred['verification']
        atom_ok=faithful(check.get('atom'),label['expected_atom']);valid=not check.get('verification_failed',False)
        rel={r['source_id']:r['relation'] for r in check.get('source_relations',[])}
        relations_ok=rel==label['relations']
        correct=check['verdict']==label['verdict'] and atom_ok and relations_ok and valid and pred['sentinel_preserved']
        counts['new_joint_errors' if label['group']=='new' else 'regression_joint_errors']+=not correct
        counts['invalid']+=not valid;counts['sentinel_changed']+=not pred['sentinel_preserved']
        details.append({'id':label['id'],'group':label['group'],'correct':correct,'atom_correct':atom_ok,
                        'relations_correct':relations_ok,'expected_verdict':label['verdict'],**pred})
    raw=[json.loads(p.read_text()) for p in (root/'slots').glob('*.raw.json')]
    bodies=[json.loads(r['raw']['body']) for r in raw];models=sorted({b.get('model') for b in bodies})
    gates={k:counts[k]<=v for k,v in ((k.removesuffix('_max'),v) for k,v in plan['gates'].items())}
    gates['one_model_no_pending']=len(models)==1 and not list((root/'slots').glob('*.intent.json'))
    result={'pass':all(gates.values()),'gates':gates,'counts':counts,'plan_hash':plan['plan_hash'],
            'contract_version':'commit_v4','pilot_allowed':False,'details':details,'usage':usage,'schema_echoes':echoes,
            'physical':{'responses':len(raw),'pending_intents':len(list((root/'slots').glob('*.intent.json'))),
                        'models':models,'prompt_tokens':sum(b['usage']['prompt_tokens'] for b in bodies),
                        'completion_tokens':sum(b['usage']['completion_tokens'] for b in bodies)},'scope':plan['scope']}
    create(root/'release_score.json',result)
    return {k:v for k,v in result.items() if k not in {'details','schema_echoes'}}


def execute(root):
    root=Path(root);plan=verify_plan(root)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False)
    client=SchemaEchoClient(AttemptClient(DurableProvider(root/'slots',plan,RawDeepSeek(model=plan['model']),parse_raw,True),plan['limits']))
    for row in json.loads((root/'inputs.json').read_text()):
        create(root/'predictions'/(row['id']+'.json'),infer(row,client))
        print('collected '+row['id'],flush=True)
    return score(root,plan,client.summary(),client.ignored_echoes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--execute-authorized',action='store_true');args=parser.parse_args()
    result=execute(args.root) if args.execute_authorized else prepare(args.root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
