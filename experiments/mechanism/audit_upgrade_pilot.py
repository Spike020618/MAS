"""Read-only raw-response/event/ordering audit for a completed pilot."""
import argparse
import json
from pathlib import Path

from experiments.core.mechanism.upgrade_game import digest, freeze_state, upgrade_request
from experiments.core.mechanism.upgrade_execution import seal_selected
from experiments.mechanism.run_upgrade_pilot import (
    engineering_contract, initial_request, parse, public_cases, read_rows,
)


def audit(directory):
    cases = json.loads((directory / 'public_cases.json').read_text())
    public_cases(cases)
    protocol = json.loads((directory / 'protocol.json').read_text())
    if protocol['case_hash'] != digest(cases):
        raise ValueError('case manifest mismatch')
    contract = engineering_contract()
    if protocol['contract_hash'] != contract.commitment():
        raise ValueError('contract mismatch')
    rows = read_rows(directory / 'responses.jsonl')
    attempts = read_rows(directory / 'attempts.jsonl')
    log = read_rows(directory / 'timeline.jsonl')
    if [r['sequence'] for r in log] != list(range(len(log))):
        raise ValueError('timeline sequence invalid')
    positions = {}
    for r in log:
        key = (r['key'], r['event'])
        if key in positions:
            raise ValueError('duplicate timeline event')
        positions[key] = r

    def position(key, event, commitment):
        record = positions[(key, event)]
        if record['commitment'] != commitment:
            raise ValueError('timeline hash mismatch')
        return record['sequence']

    by_key = {r['key']: r for r in rows}
    by_attempt = {r['key']: r for r in attempts}
    if len(by_key) != len(rows) or len(by_attempt) != len(attempts):
        raise ValueError('duplicate call or attempt')
    if set(by_key) != set(by_attempt) or len(rows) != 6 * len(cases):
        raise ValueError('missing, extra or unresolved calls')
    expected_keys = set()
    for case in cases:
        prefix = digest(case)[:20]
        initial = []

        def verify(key, request):
            expected_keys.add(key)
            row = by_key[key]
            if row['request'] != request or row['request_hash'] != digest(request):
                raise ValueError('request drift')
            if by_attempt[key]['request_hash'] != row['request_hash']:
                raise ValueError('attempt drift')
            if row['response_hash'] != digest(row['response']):
                raise ValueError('raw response drift')
            start = position(key, 'attempt', row['request_hash'])
            end = position(key, 'response', row['response_hash'])
            if start >= end:
                raise ValueError('response before attempt')
            report, usage = parse(row, case['choices'])
            return row, report, usage, start, end

        ends = []
        for i in range(3):
            _, report, _, _, end = verify(prefix + '/initial/' + str(i), initial_request(case, i))
            initial.append(report)
            ends.append(end)
        frozen = freeze_state(case['question'], case['choices'], initial)
        if json.loads((directory / (prefix + '.state.json')).read_text()) != json.loads(frozen):
            raise ValueError('frozen initial report drift')
        state_pos = position(prefix, 'state', digest(json.loads(frozen)))
        events = json.loads((directory / (prefix + '.events.json')).read_text())
        selected = json.loads((directory / (prefix + '.selection.json')).read_text())
        if events[0] != selected:
            raise ValueError('selected event mismatch')
        selection_pos = position(prefix, 'selection', digest(selected))
        if max(ends) >= state_pos or state_pos >= selection_pos:
            raise ValueError('selection before initial state frozen')
        # JSON turns tuples from the independent enumeration into lists.
        # Reconstruct canonical in-memory selection for runtime seal validation.
        from experiments.core.mechanism.upgrade_execution import select
        canonical_selection = select(frozen, contract)
        if json.loads(json.dumps(canonical_selection)) != selected:
            raise ValueError('recorded selection not reproducible')
        canonical_events = [canonical_selection] + events[1:]
        seal = seal_selected(frozen, contract, canonical_events)
        saved = json.loads((directory / (prefix + '.sealed.json')).read_text())
        if seal != saved:
            raise ValueError('sealed output drift')
        seal_pos = position(prefix, 'seal', digest(saved))
        event_by_agent = {e['agent']: e for e in events[1:]}
        for i, action in enumerate(selected['actions']):
            phase = 'upgrade' if action else 'diagnostic'
            row, report, usage, start, end = verify(prefix + '/' + phase + '/' + str(i), upgrade_request(frozen, i))
            if action:
                e = event_by_agent[i]
                if e['response_hash'] != row['response_hash'] or e['report'] != report or e['usage'] != usage:
                    raise ValueError('execution event not backed by raw response')
                if not selection_pos < start < end < seal_pos:
                    raise ValueError('selected upgrade ordering invalid')
            elif start <= seal_pos:
                raise ValueError('diagnostic generated before deployed output sealed')
    if expected_keys != set(by_key):
        raise ValueError('unregistered calls present')
    return {'status': 'passed', 'cases': len(cases), 'raw_responses': len(rows),
            'checked': ['raw_hashes', 'request_replay', 'report_parse', 'usage', 'selection_replay',
                        'output_seal', 'selected_execution_order', 'diagnostics_after_seal'],
            'scope': 'engineering integrity only, not provider signature or efficacy'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory)))
