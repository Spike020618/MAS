"""Native task adapters and strictly post-generation labels for unified E1.

Preparation reads public questions and cached independent web excerpts, or sealed
recommendation prefixes/predictions. Official answer/next-response labels are read
only by load_evaluation_labels after the caller's complete task seal is checked.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

from experiments.consensus_recommendation.common import ROOT, fingerprint, sha

QA_ROOT = ROOT / 'experiments/results/mechanism/stackelberg/locked_source_game_test_v1'
REC_ROOT = ROOT / 'experiments/results/native_review_diagnostic_packets_v1'
QA_LABELS = QA_ROOT / 'official_train.json'
REC_LABELS = ROOT / 'experiments/results/native_review_calibration_v2/fold-serving-data/valid_data.json'
ENTITIES = ROOT / 'experiments/data/raw/crslab_redial_native/released/entity2id.json'


def read(path):
    return json.loads(Path(path).read_text())


def text_hash(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def adapt_source(sid, card, task_id, *, subject=None):
    """Preserve provenance while binding permission to this exact serving task."""
    out = copy.deepcopy(card)
    out['source_id'] = sid
    if not isinstance(out.get('text'), str) or not out['text'].strip():
        raise ValueError('nonempty source text required')
    role = out.get('role', 'reference')
    use = out.get('allowed_use', 'prediction_basis')
    if subject is None:
        subject = ('other_user' if out.get('kind') == 'historical_prediction_basis'
                   else 'current_user' if use == 'current_expression'
                   else 'model_forecast' if out.get('kind') == 'model_forecast'
                   else 'catalog' if out.get('kind') == 'knowledge_graph'
                   else 'current_context' if out.get('kind') == 'dialogue'
                   else 'prediction_basis')
    out['scope'] = {'task_id': task_id, 'subject': subject, 'role': role}
    out['allowed_uses'] = list(out.get('allowed_uses', [use]))
    out['text_sha256'] = text_hash(out['text'])
    out['semantic_truth_certified'] = False
    return out


def qa_tasks(count=12, source_root=QA_ROOT):
    """Select native questions solely by public order and usable source presence."""
    source_root = Path(source_root)
    public = read(source_root/'cases.json')
    rows, skipped = [], []
    for case in public:
        native = case['question_id']
        path = source_root/'banks'/(native.rsplit('/', 1)[-1]+'.json')
        if not path.exists():
            skipped.append({'id': native, 'reason': 'missing_cached_bank'})
            continue
        bank = read(path)
        if bank.get('question_id') != native:
            raise ValueError('source bank question mismatch')
        sources = [c for c in bank.get('sources', [])
                   if isinstance(c.get('text'), str) and c['text'].strip() and c.get('url')]
        if not sources:
            skipped.append({'id': native, 'reason': 'no_usable_external_excerpt'})
            continue
        task_id = 'e1:qa:'+native.rsplit('/', 1)[-1]
        catalog = {}
        for i, source in enumerate(sources):
            full = source['text']
            recorded = source.get('excerpt_sha256')
            if recorded and recorded != text_hash(full):
                raise ValueError('cached excerpt hash mismatch')
            excerpt = full[:7000]
            sid = 'web:'+str(i)+':'+text_hash(full)[:12]
            card = {'source_id': sid, 'kind': 'cached_web_reference',
                    'title': source.get('title', ''), 'url': source['url'],
                    'text': excerpt, 'allowed_use': 'reference_material',
                    'origin_case_id': native, 'snapshot_hash': source.get('source_hash'),
                    'original_excerpt_sha256': text_hash(full),
                    'excerpt_truncated': len(full)>len(excerpt),
                    'snapshot_scope': 'historical cached independent web retrieval, not current live verification'}
            catalog[sid] = adapt_source(sid, card, task_id, subject='external_reference')
        rows.append({'id': task_id, 'kind': 'qa', 'prompt': case['question'],
                     'options': ['yes', 'no'], 'candidates': [], 'initial': 'abstain',
                     'source_catalog': catalog,
                     'metadata': {'native_dataset': 'StrategyQA', 'native_id': native,
                                  'native_answer_format': 'boolean yes/no; abstain is a failure outcome, not a gold option',
                                  'public_question_source': str(source_root/'cases.json'),
                                  'public_question_source_sha256': sha(source_root/'cases.json'),
                                  'source_bank': str(path), 'source_bank_sha256': sha(path),
                                  'query_labels_available': False,
                                  'history_scope': 'previously used public development source tasks; engineering smoke, not independent confirmation',
                                  'source_selection': 'first usable public cases in stored order, no answer/outcome read'}})
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f'only {len(rows)} native QA tasks with independent cached sources')
    return rows, skipped


def movie_title(uri):
    return unquote(uri.removeprefix('<http://dbpedia.org/resource/').removesuffix('>')).replace('_',' ')


def rec_tasks(count=12, packets=REC_ROOT, entity_path=ENTITIES):
    packets = Path(packets)
    rows = read(packets/'pair64.inputs.json')
    states = read(packets/'pair_states.json')
    names = {str(v): movie_title(k) for k,v in read(entity_path).items()}
    output = []
    for row in rows[:count]:
        native = row['case_id'];task_id = 'e1:rec:'+native
        ranking = list(map(str, states[row['task_id']]))
        if fingerprint(ranking) != row['target']['base_ranking_hash']:
            raise ValueError('frozen recommendation prior mismatch')
        if len(set(ranking)) != len(ranking) or any(m not in names for m in ranking):
            raise ValueError('duplicate/unknown native candidate')
        candidates = [{'movie_id': m, 'title': names[m]} for m in ranking]
        catalog = {sid: adapt_source(sid, c, task_id) for sid,c in row['sources'].items()}
        sid = 'forecast:shared_prior:'+native
        table = [{'movie_id':m,'prior_rank':i+1,'title':names[m]} for i,m in enumerate(ranking)]
        catalog[sid] = adapt_source(sid, {'source_id':sid, 'kind':'model_forecast',
            'allowed_use':'prediction_basis', 'origin_case_id':'frozen_training_derived_count_ranker',
            'text':'Shared training-derived expected-count ranking of native candidate pool; NOT observed next-turn targets, verified preferences, or probability. '+json.dumps(table,ensure_ascii=False,separators=(',',':'))}, task_id)
        output.append({'id':task_id,'kind':'rec','prompt':row['text'],
                       'options':[], 'candidates':candidates,'initial':ranking,
                       'source_catalog':catalog,
                       'metadata':{'native_dataset':'CRSLab-ReDial','native_id':native,
                                   'query_labels_available':False,
                                   'source_inputs':str(packets/'pair64.inputs.json'),
                                   'source_inputs_sha256':sha(packets/'pair64.inputs.json'),
                                   'prior_sha256':fingerprint(ranking),
                                   'candidate_pool':'unchanged blind Sparse100 union KBRD100; no gold insertion',
                                   'history_scope':'reused train-side diagnostic visible packets, development engineering only; not untouched confirmation',
                                   'prefix_only_inputs':True}})
    if len(output)!=count:
        raise ValueError('not enough recommendation packets')
    return output


def source_index(task):
    """Discovery metadata only. A fetch action must actually expand source text."""
    return [{'source_id': sid, 'kind': c.get('kind'), 'title': c.get('title',''),
             'scope':c['scope'],'allowed_uses':c['allowed_uses'],
             'characters':len(c['text']), 'text_sha256':c['text_sha256']}
            for sid,c in task['source_catalog'].items()]


def validate_generation_seal(tasks, generation_complete):
    seal = read(generation_complete)
    ids = seal.get('task_ids', seal.get('tasks'))
    expected = [t['id'] for t in tasks]
    if not isinstance(ids,list) or len(ids)!=len(set(ids)) or sorted(ids)!=sorted(expected):
        raise ValueError('all-task generation seal required before evaluation labels')
    if seal.get('generation_complete') is False:
        raise ValueError('incomplete generation seal')
    return seal


def load_evaluation_labels(tasks, *, generation_complete,
                           qa_path=QA_LABELS, rec_path=REC_LABELS, entity_path=ENTITIES):
    """The ONLY adapter function that reads current-query future labels."""
    validate_generation_seal(tasks, generation_complete)
    wanted_qa={t['metadata']['native_id']:t['id'] for t in tasks if t['kind']=='qa'}
    wanted_rec={t['metadata']['native_id']:t['id'] for t in tasks if t['kind']=='rec'}
    labels={}
    if wanted_qa:
        for row in read(qa_path):
            key='StrategyQA-official/'+row['qid']
            if key in wanted_qa:
                if type(row['answer']) is not bool:raise ValueError('native StrategyQA boolean answer required')
                labels[wanted_qa[key]]={'kind':'qa','answer':'yes' if row['answer'] else 'no'}
    if wanted_rec:
        from experiments.paired_deliberation.native_review_calibration_data import read_fit_targets
        targets=read_fit_targets(read(rec_path),set(wanted_rec),read(entity_path))
        labels.update({wanted_rec[k]:{'kind':'rec','targets':list(map(str,v))}for k,v in targets.items()})
    if set(labels)!={t['id']for t in tasks}:
        raise ValueError('missing native evaluation labels')
    return labels


def evaluate_decision(task, decision, label):
    if task['kind']=='qa':
        return {'correct':isinstance(decision,str) and decision.strip().lower()==label['answer'],
                'answer':decision,'gold':label['answer']}
    ranking = list(map(str,decision)) if isinstance(decision,list) else []
    candidates={str(c['movie_id'])for c in task['candidates']}
    if len(set(ranking))!=len(ranking) or any(m not in candidates for m in ranking):
        raise ValueError('invalid native ranking')
    gold=label['targets']
    return {'hits_at_10':sum(m in ranking[:10] for m in gold),'targets':len(gold),
            'recall_at_10':sum(m in ranking[:10] for m in gold)/len(gold)}
