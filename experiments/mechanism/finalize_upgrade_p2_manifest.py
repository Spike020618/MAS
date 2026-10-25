"""Add source-verified private-label commitments without resampling any case."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.core.mechanism.upgrade_game import digest
from experiments.mechanism.prepare_upgrade_p2 import pool
from experiments.mechanism.run_upgrade_pilot import freeze


def finalize(original, output):
    provenance = json.loads((original / 'provenance.json').read_text())
    paths = {s['domain']: Path(s['path']) for s in provenance['sources']}
    for source in provenance['sources']:
        if hashlib.sha256(paths[source['domain']].read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('source snapshot hash changed')
    rows, _ = pool(SimpleNamespace(strategyqa=paths.get('strategyqa'), truthfulqa=paths.get('truthfulqa'), mmlu=paths.get('mmlu')))
    by_id = {}
    for row in rows:
        # Multiple native option variants may share one original stem group.
        # Preparation uses stable source order before its input-only group dedup.
        by_id.setdefault(row['domain'] + ':' + row['source_id'], row)
    private_hashes = {}
    outputs = {}
    ids = set()
    for split in ('calibration', 'development'):
        cases = json.loads((original / (split + '_public.json')).read_text())
        labels = json.loads((original / (split + '_private.json')).read_text())
        if digest(cases) != provenance['public_hashes'][split] or len(cases) != provenance['counts'][split]:
            raise ValueError('original screened public manifest changed')
        label_map = {r['id']: r['label'] for r in labels}
        if len(label_map) != len(labels) or set(label_map) != {c['id'] for c in cases}:
            raise ValueError('private label set mismatch')
        for case in cases:
            if case['id'] in ids:
                raise ValueError('split overlap')
            ids.add(case['id'])
            row = by_id[case['id']]
            expected = {'id': case['id'], 'source_id': row['source_id'], 'question': row['question'], 'choices': row['choices']}
            if expected != case or label_map[case['id']] != row['label']:
                raise ValueError('public conversion or private label differs from original source')
        private_hashes[split] = digest(labels)
        outputs[split + '_public.json'] = cases
        outputs[split + '_private.json'] = labels
    augmented = {**provenance, 'private_hashes': private_hashes,
                 'parent_selection_provenance_sha256': hashlib.sha256((original / 'provenance.json').read_bytes()).hexdigest(),
                 'finalization': 'source-verified label commitment only; no case, label or split changed'}
    output.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        freeze(output / name, content)
    freeze(output / 'selection_provenance.json', provenance)
    freeze(output / 'provenance.json', augmented)
    return {'scope': augmented['scope'], 'counts': augmented['counts'], 'private_labels_verified_against_sources': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.original, args.output)))
