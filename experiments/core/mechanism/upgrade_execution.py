"""Append-only, offline-verifiable execution ledger for the two-stage protocol.

The caller persists each event before continuing. This module validates ordering
and commitments; it does not certify an API receipt's external authenticity.
"""
import json
from .upgrade_game import digest, freeze_state, improve, upgrade_request, validate_report, aggregate
from experiments.mechanism.audit_upgrade_game import audit


def select(frozen, contract):
    state = json.loads(frozen)
    if freeze_state(state['question'], state['choices'], state['reports']) != frozen:
        raise ValueError('invalid frozen state')
    if len(state['reports']) != contract.n:
        raise ValueError('registered agent count mismatch')
    result = improve(contract)
    checked = audit(contract, result['actions'])
    if checked['selected']['regret'] > result['tolerance']:
        raise ValueError('independent equilibrium check failed')
    return {'type': 'selection', 'state_hash': digest(state),
            'contract_hash': contract.commitment(), 'actions': result['actions'],
            'updates': result['trace'], 'audit': checked}


def seal_selected(frozen, contract, events):
    """Selection first, selected completions only, seal before diagnostics.

    Failed/truncated HTTP responses belong in the raw journal; they cannot become
    a valid report here. This routine never silently replaces invalid upgrades.
    """
    expected = select(frozen, contract)
    if not events or events[0] != expected:
        raise ValueError('selection must be committed before any upgrade completion')
    state = json.loads(frozen)
    reports = json.loads(frozen)['reports']
    required = {i for i, a in enumerate(expected['actions']) if a == 1}
    received = set()
    receipts = []
    for event in events[1:]:
        if set(event) != {'type', 'agent', 'request_hash', 'response_hash', 'report', 'finish_reason', 'usage'}:
            raise ValueError('unexpected execution event schema')
        i = event['agent']
        if event['type'] != 'selected_upgrade' or i not in required or i in received:
            raise ValueError('unselected, duplicate or diagnostic output before seal')
        request = upgrade_request(frozen, i)
        if event['request_hash'] != request['request_hash'] or event['finish_reason'] != 'stop':
            raise ValueError('request mismatch or incomplete generation')
        if not isinstance(event['response_hash'], str) or len(event['response_hash']) != 64:
            raise ValueError('raw response journal commitment required')
        usage = event['usage']
        if not isinstance(usage, dict) or not usage or any(type(v) is not int or v < 0 for v in usage.values()):
            raise ValueError('recorded nonnegative token usage required')
        validate_report(event['report'], state['choices'])
        reports[i] = event['report']
        received.add(i)
        receipts.append({'agent': i, 'request_hash': event['request_hash'], 'response_hash': event['response_hash']})
    if required != received:
        raise ValueError('selected upgrades incomplete; no seal permitted')
    output = {'type': 'sealed_output', 'state_hash': expected['state_hash'],
              'contract_hash': expected['contract_hash'], 'actions': expected['actions'],
              'reports': reports, 'decision': aggregate(reports, state['choices']), 'receipts': receipts}
    return {**output, 'output_hash': digest(output)}
