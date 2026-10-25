"""All eight frozen suites: source-group / literal-parser sensitivity.

Condition, seed and suite remain separate; no pooling as independent subjects.
This rescores saved reports only. Historical reputation feedback is unchanged.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json

from experiments.mechanism.attribute_legacy_gain import ROOT, OUT, vote, transitions
from experiments.mechanism.audit_legacy_source_groups import normalize, split_question
from experiments.mechanism.audit_legacy_conclusion_parsing import conclusion

MODES = ['original', 'literal_yes_no_only', 'unresolved_placeholder_abstains']


def source_screen(cache, splits, target):
    """Input-only group exclusions, first target occurrence in frozen order."""
    by_id = {str(q['question_id']): q for q in cache}
    other = {normalize(split_question(by_id[i]['question'])[0])
             for part, ids in splits.items() if part != target for i in ids}
    keep, excluded, seen = [], [], set()
    for qid in splits[target]:
        stem = normalize(split_question(by_id[qid]['question'])[0])
        if stem in other or stem in seen:
            excluded.append({'id': qid, 'reason': 'source_stem_in_other_split' if stem in other else 'later_target_stem_variant'})
        else:
            keep.append(qid)
        seen.add(stem)
    return {'keep': keep, 'excluded': excluded}


def outcome(row, mode):
    last = max(t['round_id'] for t in row['rounds'])
    turns = sorted([t for t in row['rounds'] if t['round_id']==last],key=lambda t:t['agent_id'])
    if len({t['agent_id'] for t in turns}) != len(turns):
        raise ValueError('duplicate agent in final round')
    stances = [t['stance'] if mode=='original' else conclusion(t['report']['C'],t['stance'],mode) for t in turns]
    decision = vote(stances)
    if mode=='original' and decision!=row['final_decision']:
        raise ValueError('original final aggregation mismatch')
    return decision==row['ground_truth']


def main():
    split_path = ROOT/'experiments/mechanism/splits.json'
    splits = json.loads(split_path.read_text())['datasets']
    label_proof_path = OUT/'source_groups.json'
    label_proof = json.loads(label_proof_path.read_text())
    caches, gold, screens = {}, {}, {}
    hashes = {str(split_path.relative_to(ROOT)):sha256(split_path.read_bytes()).hexdigest(),
              str(label_proof_path.relative_to(ROOT)):sha256(label_proof_path.read_bytes()).hexdigest()}
    for d, info in splits.items():
        path = ROOT/info['source']
        digest = sha256(path.read_bytes()).hexdigest()
        if label_proof['inputs_sha256'][info['source']] != digest:
            raise ValueError('cache changed after native source-label verification')
        caches[d] = json.loads(path.read_text())['questions']
        gold[d] = {str(q['question_id']):q['ground_truth'] for q in caches[d]}
        proof = label_proof['datasets'][d]['native_label_check']['rows']
        if len(proof)!=len(caches[d]) or any(p['status']!='label_matches' for p in proof):
            raise ValueError('unverified native labels in cache')
        screens[d] = {part:source_screen(caches[d],info['splits'],part) for part in ['dev','test']}
        hashes[info['source']] = digest
    suites = {}
    checked = Counter()
    frozen = ROOT/'experiments/results/mechanism/frozen'
    for folder in sorted(frozen.iterdir()):
        path = folder/'question_level.jsonl'
        if not path.exists() or folder.name.endswith('.recomputed'):
            continue
        matrix = json.loads((folder/'matrix.json').read_text())
        split = matrix['defaults']['split']
        groups = defaultdict(list)
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r['completed'] or r.get('error'):
                raise ValueError('incomplete frozen record')
            if r['ground_truth'] != gold[r['dataset']][str(r['question_id'])]:
                raise ValueError('frozen label differs from native-verified cache')
            checked['records_native_label_consistent'] += 1
            groups[(r['condition'],r['seed'])].append(r)
        cells = []
        for (condition,seed), rs in sorted(groups.items()):
            keys = [(r['dataset'],str(r['question_id'])) for r in rs]
            if len(keys)!=len(set(keys)):
                raise ValueError('duplicate paired condition seed key')
            clean = [r for r in rs if str(r['question_id']) in screens[r['dataset']][split]['keep']]
            cells.append({'condition':condition,'seed':seed,'split':split,
                'by_policy': {mode:{name:{'n':len(sub), 'correct':sum(outcome(r,mode) for r in sub),
                    'by_domain':{d:{'n':sum(r['dataset']==d for r in sub),
                        'correct':sum(outcome(r,mode) for r in sub if r['dataset']==d)}
                        for d in sorted(gold)}} for name,sub in [('all_records',rs),('source_group_clean',clean)]}
                    for mode in MODES}})
        # Only actual no-mechanism / static pairs are compared. Do not invent
        # omitted baseline conditions from a different suite or seed.
        contrasts = []
        seeds = sorted({seed for c,seed in groups})
        for seed in seeds:
            for lc in sorted(c for c,s in groups if s==seed and '__no_mechanism@' in c):
                rc = lc.replace('__no_mechanism@','__static_aeic@',1)
                if (rc,seed) not in groups:
                    continue
                left = {(r['dataset'],str(r['question_id'])):r for r in groups[(lc,seed)]}
                right = {(r['dataset'],str(r['question_id'])):r for r in groups[(rc,seed)]}
                if left.keys()!=right.keys():
                    raise ValueError('unpaired reference contrast')
                for mode in MODES:
                    for screen in ['all_records','source_group_clean']:
                        keys = [k for k in sorted(left) if screen=='all_records' or k[1] in screens[k[0]][split]['keep']]
                        contrasts.append({'left':lc,'right':rc,'seed':seed,'mode':mode,'screen':screen,
                            **transitions([outcome(left[k],mode) for k in keys],[outcome(right[k],mode) for k in keys])})
        suites[folder.name] = {'cells':cells,'static_vs_none_where_present':contrasts}
        hashes[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
        checked['suites'] += 1
    result = {'scope':'post-hoc all-frozen-suite sensitivity; no APIs; no independent sample pooling',
        'checks':dict(checked),'screens':screens,'suites':suites,'inputs_sha256':hashes,
        'limits':['dev screen excludes train AND test stems; not an original preregistered dev selection',
                  'does not undo prior exposure or reputation effects from excluded questions',
                  'native label verification is source agreement, not independent factual adjudication',
                  'parser changes do not replay settlement, selection, response generation or dynamic history']}
    (OUT/'all_suite_sensitivity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':result['checks'],'contrasts':{s:x['static_vs_none_where_present'] for s,x in suites.items() if x['static_vs_none_where_present']}},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
