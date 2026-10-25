"""Public-table equilibrium recommendation BEFORE resource commitments.

Explicitly a centralized finite-table selector, equal to the registered central
proxy optimum baseline. Per-agent checks give unilateral stability and weak
participation under this proxy contract, NOT strong Nash or true-value efficacy.
"""
import math

from .upgrade_game import digest
from experiments.mechanism.audit_upgrade_game import audit


def recommend(contract, tolerance=1e-12):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('finite nonnegative tolerance required')
    profiles = contract.profiles()
    actions = min(profiles, key=lambda p: (-contract.settlement(p)['potential'], sum(p), p))
    checked = audit(contract, actions)
    if checked['selected']['regret'] > tolerance or checked['global_objective_gap'] > tolerance:
        raise ValueError('independent proxy optimum/equilibrium check failed')
    settled = contract.settlement(actions)
    approvals = []
    for i, action in enumerate(actions):
        alternative = list(actions); alternative[i] = 1 - action
        gain = contract.settlement(alternative)['utilities'][i] - settled['utilities'][i]
        outside = contract.beta - contract.costs[i][0]
        # A K action always pays beta, independent of peers' choices.
        accepted = gain <= tolerance and settled['utilities'][i] >= outside - tolerance
        approvals.append({'agent': i, 'action': action, 'utility': settled['utilities'][i],
                          'keep_reference_utility': outside, 'best_unilateral_deviation_gain': gain,
                          'accept': accepted, 'participation': 'weak_not_necessarily_strict'})
    if not all(a['accept'] for a in approvals):
        raise ValueError('agent utility/participation check failed')
    decision = {'version': 'public-proxy-equilibrium-recommendation-v3',
                'coordination': 'trusted_central_enumeration_of_public_table', 'actions': list(actions),
                'contract_hash': contract.commitment(), 'agent_approvals': approvals,
                'audit': checked, 'settlement': settled,
                'limitations': ['proxy optimum only', 'not strong Nash or coalition proof',
                                'not a new distributed optimizer', 'LLM natural rationality not measured']}
    return {**decision, 'decision_hash': digest(decision)}
