"""Simple procurement control with SAME information, routes, wallets and costs.

No new route selection and no API. Receipt-verified exact cost reimbursement
makes any already-public target mask strictly implementable for epsilon>0.
This is a necessity test against a simpler contract, not a new contribution.
"""
from hashlib import sha256
import json

from experiments.mechanism.replay_budgeted_information import ROOT, OUT
from experiments.mechanism.audit_effort_generalization import AUDIT
from experiments.mechanism.run_upgrade_pilot import freeze


def menu_reply(target, actual_incremental_cost, fee=1):
    if target not in (0, 1) or type(actual_incremental_cost) is not int or actual_incremental_cost <= 0 or fee <= 0:
        raise ValueError('invalid trusted cost, target or positive fee')
    # Initial cost is the same sunk constant for both alternatives.
    keep_utility = 0
    upgrade_credit = actual_incremental_cost+fee if target else 0
    upgrade_utility = upgrade_credit-actual_incremental_cost
    chosen = int(upgrade_utility > keep_utility)
    return {'chosen': chosen, 'keep_incremental_utility': keep_utility,
            'upgrade_incremental_utility': upgrade_utility,
            'credit_if_upgrade': upgrade_credit}


def main():
    traces_path = OUT/'certified_effort_episode_traces.json'
    result_path = OUT/'certified_effort_result.json'
    traces = json.loads(traces_path.read_text())
    result = json.loads(result_path.read_text())
    # Same authenticated routes, not per-question labels or a new optimizer.
    rows_by_scale = {}
    for scale, rows in traces.items():
        target = result['scenarios_all_scales'][scale]['selected_profile']
        paid = 0
        comparison = []
        for row in rows:
            details = []
            for i, ledger in enumerate(row['ledgers']):
                actual_high_cost = (ledger['tokens']-ledger['initial_tokens']) if target[i] else 2000
                menu = menu_reply(target[i], actual_high_cost)
                if menu['chosen'] != target[i]: raise ValueError('desired authenticated path not implemented')
                payment = menu['credit_if_upgrade'] if menu['chosen'] else 0
                paid += payment
                details.append({'agent': i, 'target': target[i], 'choice': menu['chosen'],
                    'cost_reimbursement_plus_fee': payment,
                    'full_utility': payment-ledger['tokens']})
            if sum(d['cost_reimbursement_plus_fee'] for d in details) > 2*int(scale):
                raise ValueError('not within pivotal contract total payment ceiling in this comparison')
            comparison.append({'id': row['id'], 'same_certified_outputs_and_debits': True,
                'correct_unchanged': row['correct'], 'actors': details})
        s = result['scenarios_all_scales'][scale]['selected_summary']
        rows_by_scale[scale] = {'target_profile_unchanged': target,
            'correct_unchanged': s['correct'], 'tokens_unchanged': s['tokens_including_initial'],
            'high_calls_unchanged': s['high_calls_replayed'],
            'pivotal_virtual_payment': s['total_virtual_credit'],
            'cost_plus_one_virtual_payment': paid,
            'simple_contract_within_same_total_payment_ceiling': True,
            'rows': comparison}
    active = ROOT/'docs/research/ACTIVE_RESEARCH_PLAN.md'
    checkpoint = AUDIT/'active_plan_before_procurement_control.json'
    # Preserve the prior mutable-entry snapshot referenced by the earlier
    # acceptance, before advancing that entry to this new control result.
    if not checkpoint.exists():
        freeze(checkpoint, {'path': str(active.relative_to(ROOT)),
            'sha256': sha256(active.read_bytes()).hexdigest(), 'utf8_text': active.read_text()})
    audit = {'scope': 'same-path simple-contract necessity control on already opened traces',
        'new_model_calls': 0, 'fee': 1,
        'current_question_outcome_used_by_control': False,
        'route_mask_source': 'existing calibration plan, no refitting from evaluation',
        'scenarios': rows_by_scale,
        'inputs_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in [traces_path, result_path]},
        'conclusion': 'public fixed routes and verified costs permit simpler implementation of identical actions; pivotal payment has no established advantage',
        'assumptions': ['trusted receipts expose complete action-dependent cost',
            'fixed gateway requests; agents cannot inflate authorized cost or forge output',
            'existing funded wallets can front reserved cost before reimbursement',
            'same already-enrolled actors and mandatory initial-work scope',
            'no unmodeled private opportunity cost, hidden quality choice or strategic exit'],
        'limits': ['virtual payment reductions are NOT API, Token, accuracy or latency improvements',
            'no universal payment dominance asserted outside these sampled paths and assumptions',
            'one-unit fee is arbitrary positive epsilon, not outcome-tuned',
            'does not establish private-cost truthfulness or solve the participation stage',
            'no extra hidden asymmetry is invented to rescue the complex mechanism']}
    freeze(AUDIT/'procurement_control.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k != 'scenarios'} | {
        'scenarios': {k: {x: y for x, y in v.items() if x != 'rows'} for k, v in rows_by_scale.items()}}, ensure_ascii=False, indent=2))
    return audit


if __name__ == '__main__': main()
