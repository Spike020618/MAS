"""Executable v3 plan and sealed receipts. No transport or answer labels."""
import json

from .upgrade_game import digest
from .upgrade_protocol_v3 import decode_row, majority, request, snapshot
from .upgrade_coordination import recommend


def prepare(question, choices, initial, contract):
    if contract.n != 3:
        raise ValueError('three registered agents required')
    frozen = snapshot(question, choices, initial)
    choice = recommend(contract)
    jobs = [{'agent': i, 'request': request(frozen, i)} for i, a in enumerate(choice['actions']) if a]
    plan = {'version': 'upgrade-execution-v3', 'snapshot': frozen,
            'recommendation': choice, 'jobs': jobs, 'contract_hash': contract.commitment()}
    return {**plan, 'plan_hash': digest(plan)}


def finish(plan, contract, receipts):
    original = {k: v for k, v in plan.items() if k != 'plan_hash'}
    if digest(original) != plan['plan_hash'] or contract.commitment() != plan['contract_hash']:
        raise ValueError('execution plan or contract drift')
    state = json.loads(plan['snapshot'])
    if digest(prepare(state['question'], state['choices'], state['reports'], contract)) != digest(plan):
        raise ValueError('recommendation cannot be independently reproduced')
    required = {j['agent']: j['request'] for j in plan['jobs']}
    seen, reports, used = set(), state['reports'], []
    for receipt in receipts:
        if set(receipt) != {'agent', 'row'}:
            raise ValueError('exact receipt schema required')
        i, row = receipt['agent'], receipt['row']
        if type(i) is not int or i not in required or i in seen:
            raise ValueError('unselected or duplicate receipt')
        if row['request'] != required[i] or row['request_hash'] != digest(required[i]) or row['response_hash'] != digest(row['response']):
            raise ValueError('receipt request/raw-response mismatch')
        reports[i], usage = decode_row(row, state['choices'])
        used.append({'agent': i, 'response_hash': row['response_hash'], 'usage': usage,
                     'conclusion_usable': reports[i]['conclusion_usable'], 'aeic_complete': reports[i]['aeic_complete']})
        seen.add(i)
    if seen != set(required):
        raise ValueError('selected executions incomplete; cannot seal')
    output = {'version': 'upgrade-output-v3', 'plan_hash': plan['plan_hash'],
              'actions': plan['recommendation']['actions'], 'reports': reports,
              'decision': majority(reports, state['choices']), 'executed_upgrades': used}
    return {**output, 'output_hash': digest(output)}
