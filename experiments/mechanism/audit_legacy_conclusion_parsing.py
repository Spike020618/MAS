"""Label-blind sensitivity to schema-placeholder prefixes in saved conclusions.

Never infer intent from A/E/I, gold, model or condition. Existing labels and
stances stay untouched. Ambiguous boolean lists are not silently repaired.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from experiments.mechanism.attribute_legacy_gain import ROOT, OUT, vote, transitions

PREFIX = re.compile(r'^(?:yes\s*(?:/|or)\s*no|no\s*(?:/|or)\s*yes)\b', re.I)
EXPLICIT = re.compile(r'^(?:yes\s*(?:/|or)\s*no|no\s*(?:/|or)\s*yes)\s*:\s*(yes|no)[.!]?$', re.I)


def conclusion(text, original_stance, mode):
    if mode not in {'literal_yes_no_only', 'unresolved_placeholder_abstains'}:
        raise ValueError('unknown fixed sensitivity policy')
    match = EXPLICIT.fullmatch(text.strip())
    if match:
        return match[1].lower()
    if mode == 'unresolved_placeholder_abstains' and PREFIX.match(text.strip()):
        return 'neutral'
    return original_stance


def analyze(rows):
    summaries, changed = {}, []
    for mode in ['original', 'literal_yes_no_only', 'unresolved_placeholder_abstains']:
        groups = defaultdict(dict)
        details = defaultdict(Counter)
        for r in rows:
            key = (r['dataset'], str(r['question_id']))
            stances = []
            for t in sorted(r['rounds'], key=lambda t: (t['round_id'], t['agent_id'])):
                c = t['report']['C']
                v = t['stance'] if mode == 'original' else conclusion(c, t['stance'], mode)
                details[r['condition']]['placeholder_prefix_slots'] += bool(PREFIX.match(c.strip()))
                details[r['condition']]['explicit_single_yes_no_suffix_slots'] += bool(EXPLICIT.fullmatch(c.strip()))
                details[r['condition']]['stance_changes'] += v != t['stance']
                if v != t['stance']:
                    changed.append({'mode': mode, 'dataset': r['dataset'], 'id': r['question_id'],
                        'condition': r['condition'], 'round': t['round_id'], 'agent': t['agent_id'],
                        'raw_C': c, 'old': t['stance'], 'new': v})
                if t['round_id'] == 3:
                    stances.append(v)
            groups[r['condition']][key] = vote(stances) == r['ground_truth']
        keys = sorted(groups['independent__no_mechanism@0.2'])
        summaries[mode] = {'correct': {c:sum(v.values()) for c,v in groups.items()},
            'static_vs_none': transitions([groups['independent__no_mechanism@0.2'][k] for k in keys],
                                           [groups['independent__static_aeic@0.2'][k] for k in keys]),
            'slots': {c:dict(ct) for c,ct in details.items()}}
    return {'scope': 'post-hoc label-blind literal parsing sensitivity, not historical corrected execution',
            'no_api_calls': True, 'summaries': summaries, 'changed_slots': changed,
            'limits': ['ambiguous True/False lists intentionally not interpreted as valid Yes/No judgments',
                       'credit and later reputation were based on historical stances: no dynamic feedback replay claimed',
                       'field validity / source validity / factual report quality are separate questions',
                       'none of these checks proves correct interpretation of every free-form C']}


def main():
    path = ROOT / 'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl'
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    result = analyze(rows)
    result['source_sha256'] = sha256(path.read_bytes()).hexdigest()
    grouping = OUT/'source_groups.json'
    if grouping.exists():
        groups = json.loads(grouping.read_text())
        excluded = {(d, str(x['id'])) for d, ds in groups['datasets'].items()
                    for x in ds['grouping']['excluded_records']}
        clean = [r for r in rows if (r['dataset'], str(r['question_id'])) not in excluded]
        result['grouped_clean_summaries'] = analyze(clean)['summaries']
        result['grouping_sha256'] = sha256(grouping.read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'conclusion_parsing.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['summaries'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
