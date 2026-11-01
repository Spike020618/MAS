"""Separate offline scorer; refuses to load labels before the complete seal."""
import json
from math import comb
from experiments.mechanism import run_action_comparison as r
from experiments.core.mechanism.evidence_acquisition import answer_score
from experiments.mechanism.run_upgrade_pilot import freeze


def exact_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def main():
    r.verify()
    cases = r.read(r.OUT / 'public_cases.json')
    outputs = r.read(r.OUT / 'sealed_outputs.json')
    if len(cases) != 24 or set(outputs) != {c['id'] for c in cases}:
        raise ValueError('all24 sealed before reading private labels')
    for cid, output in outputs.items():
        if set(output['arms']) != set(r.ARMS) or r.read(r.OUT / cid / 'sealed_outputs.json') != output:
            raise ValueError('incomplete/mismatched case seal')
        r.replay_case(next(c for c in cases if c['id'] == cid), output,
                      r.read_rows(r.OUT / cid / 'responses.jsonl'))
    receipt = r.read(r.OUT / 'collection_receipt.json')
    if receipt['sealed_outputs_sha256'] != r.sha(r.OUT / 'sealed_outputs.json'):
        raise ValueError('sealed answers changed')
    selection = r.read(r.OUT / 'selection.json')
    if r.sha(r.OUT / 'private_labels.json') != selection['labels_sha256']:
        raise ValueError('labels changed')
    # This is the first private label parse. Collector never calls this module.
    labels = r.read(r.OUT / 'private_labels.json')
    splits = r.read(r.OUT / 'splits.json')
    all_raw = r.ledger_rows(r.OUT)
    if len(all_raw) != receipt['calls'] or sum(r.usage(v)['total_tokens'] for v in all_raw) != receipt['tokens']:
        raise ValueError('receipt disagrees with journals')
    per_case = {}
    for cid, output in outputs.items():
        raw = {v['key']: v for v in r.read_rows(r.OUT / cid / 'responses.jsonl')}
        per_case[cid] = {}
        for arm in ('direct', *r.ARMS):
            branch = output['arms'].get(arm)
            answer = output['initial'] if arm == 'direct' else branch['final']
            keys = [output['initial']['source_key']]
            if branch:
                if branch['action']:
                    keys.append(branch['action']['source_key'])
                if answer:
                    keys.append(answer['source_key'])
            chosen = [raw[key] for key in dict.fromkeys(keys)]
            parsed = answer['parsed'] if answer else {'valid': False, 'answer': 'invalid'}
            per_case[cid][arm] = {'answer': parsed['answer'], 'valid': parsed['valid'],
                'score': answer_score(parsed['answer'], labels[cid]['answer']),
                'tokens': sum(r.usage(v)['total_tokens'] for v in chosen), 'calls': len(chosen),
                'http_seconds': sum(v['wall_seconds'] for v in chosen),
                'search_seconds': branch['retrieval']['event']['wall_seconds'] if branch and branch['retrieval'] else 0.0}
    cells = {}
    for split in ('development', 'confirmation', 'all'):
        ids = [cid for cid in outputs if split == 'all' or splits[cid] == split]
        arms = {}
        for arm in ('direct', *r.ARMS):
            values = [per_case[cid][arm] for cid in ids]
            arms[arm] = {'n': len(ids), 'correct': sum(v['score']['em'] for v in values),
                'mean_f1': sum(v['score']['f1'] for v in values) / len(ids),
                'invalid': sum(not v['valid'] for v in values),
                **{k: sum(v[k] for v in values) for k in ('tokens', 'calls', 'http_seconds', 'search_seconds')},
                'corrections': [cid for cid in ids if per_case[cid]['direct']['score']['em'] == 0 and per_case[cid][arm]['score']['em'] == 1],
                'harms': [cid for cid in ids if per_case[cid]['direct']['score']['em'] == 1 and per_case[cid][arm]['score']['em'] == 0]}
        pairs = []
        for control in ('review', 'resample'):
            b = sum(per_case[cid]['check']['score']['em'] > per_case[cid][control]['score']['em'] for cid in ids)
            c = sum(per_case[cid]['check']['score']['em'] < per_case[cid][control]['score']['em'] for cid in ids)
            pairs.append({'control': control, 'check_wins': b, 'control_wins': c, 'p': exact_p(b, c)})
        ordered = sorted(pairs, key=lambda p: p['p'])
        previous = 0.0
        for i, pair in enumerate(ordered):
            previous = min(1.0, max(previous, (len(ordered) - i) * pair['p']))
            pair['holm_p'] = previous
        exclusive = {arm: [cid for cid in arms[arm]['corrections'] if cid not in arms[other]['corrections']]
                     for arm, other in (('retrieve', 'check'), ('check', 'retrieve'))}
        cells[split] = {'arms': arms, 'primary_pairs': pairs, 'exclusive_corrections': exclusive}
    confirmation = cells['confirmation']
    numeric = all(confirmation['arms']['check']['correct'] >= confirmation['arms'][v]['correct'] + 2
                  for v in ('review', 'resample')) and all(confirmation['exclusive_corrections'].values())
    result = {'status': 'paired_action_screen_NOT_game_training', 'cells': cells, 'per_case': per_case,
        'numeric_screen_pass': numeric, 'scientific_gate': 'PENDING_case_review' if numeric else 'FAILED_no_expansion',
        'never_automatic_training': True, 'requires_at_least_two_nonlexical_factual_check_corrections': True,
        'physical_calls': receipt['calls'], 'physical_tokens': receipt['tokens'], 'usd_proxy': receipt['usd_proxy']}
    freeze(r.OUT / 'analysis.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('cells', 'per_case')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
