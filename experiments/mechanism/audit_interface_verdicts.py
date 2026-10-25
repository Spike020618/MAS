"""Label-blind, offline sensitivity audit; never modifies historical observations."""
import json
import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from experiments.mechanism.extend_legacy_sensitivity import source_screen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/results/audit/interface_verdicts_2026-10-24'


def verdict(raw, protocol):
    """Extract explicit verdicts only. Missing/conflicting declarations abstain.

    No labels, question content, or semantic correctness rules enter extraction.
    For prose, final-answer markers supersede initial/paragraph declarations.
    Without a marker all paragraph-leading declarations must agree.
    """
    if protocol not in ('free_text', 'generic_structure', 'aeic'):
        raise ValueError(protocol)
    text = raw
    if protocol != 'free_text':
        try:
            text = json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0][
                'answer' if protocol == 'generic_structure' else 'C']
            if not isinstance(text, str):
                raise ValueError('non-string verdict')
        except (ValueError, KeyError, TypeError):
            return {'stance': None, 'rule': 'invalid_json_or_field', 'declarations': []}
        match = re.match(r'^[\s*#_`]*(yes|no)\b', text, re.I)
        hits = [match] if match else []
        rule = 'json_field_prefix'
    else:
        hits = list(re.finditer(r'\bfinal\s+answer[\s:*_`]*(yes|no)\b', text, re.I))
        rule = 'explicit_final_marker'
        if not hits:
            hits = list(re.finditer(r'(?:\A|\n\s*\n)[\s*#_`]*(yes|no)\b', text, re.I))
            standalone = list(re.finditer(r'^[ \t*#_`]*(yes|no)[ \t.*_`]*$', text, re.I | re.M))
            hits = sorted({m.start(1): m for m in hits + standalone}.values(), key=lambda m: m.start(1))
            rule = 'paragraph_or_standalone_declarations'
    declarations = [{'stance': m.group(1).lower(), 'start': m.start(1),
                     'end': m.end(1), 'location': 'raw' if protocol == 'free_text' else
                     ('answer' if protocol == 'generic_structure' else 'C')}
                    for m in hits]
    values = {m['stance'] for m in declarations}
    return {'stance': next(iter(values)) if len(values) == 1 else None,
            'rule': rule if len(values) == 1 else ('conflict' if values else 'no_explicit_verdict'),
            'declarations': declarations}


def run():
    path = ROOT / 'experiments/results/mechanism/interface/report_interface_v1.jsonl'
    split_path = ROOT / 'experiments/mechanism/splits.json'
    splits = json.loads(split_path.read_text())['datasets']
    inputs = [path, split_path, Path(__file__)]
    questions, clean = {}, {}
    for d, s in splits.items():
        cp = ROOT / s['source']
        inputs.append(cp)
        qs = json.loads(cp.read_text())['questions']
        questions[d] = {str(q['question_id']): q for q in qs}
        clean[d] = set(source_screen(qs, s['splits'], 'test')['keep'])
    groups = defaultdict(dict)
    records = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        assert r['completed'] and not r.get('error')
        qid, d, p = str(r['question_id']), r['dataset'], r['protocol']
        assert r['ground_truth'] == questions[d][qid]['ground_truth']
        key = (d, qid, r['seed'])
        assert key not in groups[p]
        entry = {**r, 'question': questions[d][qid]['question'],
                 'source_clean': qid in clean[d], 'audit': verdict(r['raw_response'], p)}
        groups[p][key] = entry
        records.append(entry)
    assert len(records) == 270 and len(groups) == 3
    assert all(set(g) == set(groups['aeic']) for g in groups.values())
    results = {}
    for scope in ('all', 'source_clean'):
        keys = [k for k, r in groups['aeic'].items() if scope == 'all' or r['source_clean']]
        summary = {}
        for p, rows in groups.items():
            rs = [rows[k] for k in keys]
            summary[p] = {'n': len(rs),
                'historical_correct': sum(r['observation']['stance'] == r['ground_truth'] for r in rs),
                'explicit_correct_lower': sum(r['audit']['stance'] == r['ground_truth'] for r in rs),
                'unresolved': sum(r['audit']['stance'] is None for r in rs),
                'changed': sum(r['audit']['stance'] != r['observation']['stance'] for r in rs),
                'input_tokens': sum(r['token_input'] for r in rs),
                'output_tokens': sum(r['token_output'] for r in rs)}
            summary[p]['explicit_correct_upper'] = summary[p]['explicit_correct_lower'] + summary[p]['unresolved']
        contrasts = {}
        for p in ('generic_structure', 'free_text'):
            cells = defaultdict(int)
            for k in keys:
                a, b = groups[p][k], groups['aeic'][k]
                if a['audit']['stance'] is None or b['audit']['stance'] is None:
                    cells['unresolved_pairs'] += 1
                    continue
                ac = a['audit']['stance'] == a['ground_truth']
                bc = b['audit']['stance'] == b['ground_truth']
                cells['corrections' if bc and not ac else 'harms' if ac and not bc else 'both_correct' if ac else 'both_wrong'] += 1
            contrasts[p] = dict(cells)
        results[scope] = {'summary': summary, 'AEIC_vs_baseline_resolved_pairs_only': contrasts}
    discordant = []
    for k, a in groups['aeic'].items():
        b = groups['generic_structure'][k]
        if (a['observation']['stance'] == a['ground_truth']) != (b['observation']['stance'] == b['ground_truth']):
            discordant.append({'key': k, 'question': a['question'], 'gold': a['ground_truth'],
                               'records': {p: groups[p][k] for p in groups}})
    return {'scope': 'post-hoc extraction sensitivity, not a new confirmatory experiment',
            'new_model_calls': 0, 'results': results, 'original_generic_AEIC_discordances': discordant,
            'changed_or_unresolved': [r for r in records if r['audit']['stance'] != r['observation']['stance'] or r['audit']['stance'] is None],
            'all_records': records,
            'input_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in inputs}}


if __name__ == '__main__':
    result = run()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'audit_v2.json'
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if target.exists() and target.read_text() != payload:
        raise ValueError('refusing to overwrite incompatible audit')
    if not target.exists():
        target.write_text(payload)
    print(json.dumps(result['results'], ensure_ascii=False, indent=2))
