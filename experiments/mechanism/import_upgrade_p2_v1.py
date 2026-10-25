"""Import committed calls into engineering-v2 without issuing or rewriting calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from experiments.core.mechanism.upgrade_game import digest
from experiments.mechanism.run_upgrade_pilot import freeze, read_rows, lock


def migrate(source, target):
    with lock(target):
        responses = read_rows(source / 'responses.jsonl')
        attempts = read_rows(source / 'attempts.jsonl')
        if len({r['key'] for r in responses}) != len(responses) or len({r['key'] for r in attempts}) != len(attempts):
            raise ValueError('duplicate source job')
        attempted = {r['key']: r for r in attempts}
        if set(attempted) != {r['key'] for r in responses}:
            raise ValueError('unresolved transport attempt cannot be migrated')
        for row in responses:
            if row['request_hash'] != digest(row['request']) or row['response_hash'] != digest(row['response']) or attempted[row['key']]['request_hash'] != row['request_hash']:
                raise ValueError('source raw commitment mismatch')
        files = [source / name for name in ('attempts.jsonl', 'responses.jsonl', 'timeline.jsonl')]
        for suffix in ('.state.json', '.selection.json', '.events.json', '.sealed.json'):
            files.extend(source.glob('*' + suffix))
        copied = {}
        for path in files:
            destination = target / path.name
            data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != data_hash:
                    raise ValueError('refuse changed migration target; never reimport after continuation')
            else:
                shutil.copyfile(path, destination)
            copied[path.name] = data_hash
        manifest = {'source': str(source), 'source_protocol_sha256': hashlib.sha256((source / 'protocol.json').read_bytes()).hexdigest(),
                    'imported_responses': len(responses), 'imported_attempts': len(attempts), 'files': copied,
                    'billing': 'same original calls, not new calls or independent replication', 'api_calls_added_by_import': 0}
        freeze(target / 'v1_import.json', manifest)
        return {k: manifest[k] for k in ('imported_responses', 'imported_attempts', 'api_calls_added_by_import')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.target)))
