"""Restricted, finite, verifiable-production action game.

Only K/U resource actions are strategic; the trusted gateway automatically seals
the selected original tool output. This is NOT a truthful-language-report game.
Routes are reserved ex ante for a batch, before observing its task or output.
"""
from fractions import Fraction as F
from itertools import product
from dataclasses import dataclass


PROFILES = tuple(product((0, 1), repeat=3))


@dataclass(frozen=True)
class FollowerMenu:
    agent: int
    recommended: int
    expected_keep_utility: F
    expected_upgrade_utility: F
    wallet: int = 3000
    initial_reservation: int = 1000
    upgrade_reservation: int = 2000


def follower_reply(menu):
    """Own two utilities and own wallet ONLY; cannot read question/gold/peers.

    Recommendation is accepted on utility ties. Rejecting a non-best-response
    recommendation is allowed; the synchronizer must not force it through.
    """
    if menu.agent not in (0, 1, 2) or menu.recommended not in (0, 1):
        raise ValueError('invalid actor or recommendation')
    if min(menu.wallet, menu.initial_reservation, menu.upgrade_reservation) < 0:
        raise ValueError('invalid private reservation')
    if menu.wallet < menu.initial_reservation:
        raise ValueError('cannot fund mandatory initial action')
    feasible_upgrade = menu.wallet >= menu.initial_reservation+menu.upgrade_reservation
    if not feasible_upgrade: return 0
    keep, upgrade = menu.expected_keep_utility, menu.expected_upgrade_utility
    if upgrade > keep: return 1
    if keep > upgrade: return 0
    return menu.recommended


def follower_menus(table, recommendation):
    result = []
    for i in range(3):
        values = []
        for own_action in (0, 1):
            profile = tuple(own_action if j == i else v for j, v in enumerate(recommendation))
            values.append(table[profile]['utilities'][i])
        result.append(FollowerMenu(i, recommendation[i], values[0], values[1]))
    return result


def commit_recommendation(table, recommendation):
    menus = follower_menus(table, recommendation)
    replies = tuple(follower_reply(menu) for menu in menus)
    if replies != recommendation:
        raise ValueError('independent follower rejected recommendation; do not execute')
    return menus, replies


def majority(votes, choices):
    return next((choice for choice in choices if votes.count(choice) >= 2), 'abstain')


def pivotal_credits(votes, choices, outcome, scale):
    """Correct pivotal-vote payment; outcome appears ONLY at settlement.

    For fixed quorum 2 of 3, adding one authenticated vote cannot overturn two
    agreeing peers. Thus V(full)-V(without_i) is 0 or 1, and at most two agents
    are pivotal. Total payment is <=2*scale without a constant subsidy.
    """
    if len(votes) != 3 or outcome not in choices or scale < 0:
        raise ValueError('invalid registry, scale or panel')
    full = int(majority(votes, choices) == outcome)
    credits = []
    for i in range(3):
        reference = int(majority([v for j, v in enumerate(votes) if j != i], choices) == outcome)
        difference = full-reference
        if difference not in (0, 1): raise ValueError('monotonic quorum assumption failed')
        credits.append(scale*difference)
    if sum(credits) > 2*scale: raise ValueError('payment cap failed')
    return credits


def empirical_game(cases, outcomes, scale):
    """Exact empirical expectation; caller supplies CALIBRATION cases only.

    Cases contain both authenticated original paths, measured incremental costs
    and initial costs. The gold dataset is an offline estimator, not an agent's
    current-question action input. No conditional per-question routing here.
    """
    if not cases or len({r['id'] for r in cases}) != len(cases):
        raise ValueError('nonempty unique calibration cases required')
    scale = F(scale)
    if scale < 0: raise ValueError('negative credit scale')
    for row in cases:
        if len(row['choices']) < 2 or len(set(row['choices'])) != len(row['choices']):
            raise ValueError('invalid choice registry')
        if any(len(row[k]) != 3 for k in ('low', 'high', 'low_tokens', 'high_tokens')):
            raise ValueError('exactly three certified slots required')
        if any(v not in [*row['choices'], 'abstain'] for k in ('low', 'high') for v in row[k]):
            raise ValueError('unregistered conclusion')
        if any(type(t) is not int or t <= 0 for k in ('low_tokens', 'high_tokens') for t in row[k]):
            raise ValueError('measured positive integer usage required')
    n = len(cases)
    table = {}
    for actions in PROFILES:
        value = F(0)
        payments = [F(0)]*3
        costs = [F(0)]*3
        initial = [F(0)]*3
        individual_payments = [F(0)]*3
        for case in cases:
            votes = [case['high'][i] if a else case['low'][i] for i, a in enumerate(actions)]
            outcome = outcomes[case['id']]
            value += int(majority(votes, case['choices']) == outcome)
            credit = pivotal_credits(votes, case['choices'], outcome, scale)
            for i in range(3):
                payments[i] += credit[i]
                costs[i] += actions[i]*case['high_tokens'][i]
                initial[i] += case['low_tokens'][i]
                # Same aggregate ceiling 2*scale as pivotal payment, NOT equal
                # realized expenditure or the same quality-sensitive incentives.
                individual_payments[i] += F(2, 3)*scale*int(votes[i] == outcome)
        value /= n
        payments = [x/n for x in payments]
        costs = [x/n for x in costs]
        initial = [x/n for x in initial]
        individual_payments = [x/n for x in individual_payments]
        table[actions] = {'accuracy': value, 'payments': payments,
            'incremental_costs': costs, 'initial_costs': initial,
            'utilities': [payments[i]-costs[i]-initial[i] for i in range(3)],
            'individual_payments_same_cap': individual_payments,
            'individual_utilities': [individual_payments[i]-costs[i]-initial[i] for i in range(3)],
            'potential': scale*value-sum(costs)-sum(initial)}
    return table


def regret(table, profile, key='utilities'):
    return max([F(0)]+[table[tuple(1-v if j == i else v for j, v in enumerate(profile))][key][i]
                         -table[profile][key][i] for i in range(3)])


def equilibria(table, key='utilities'):
    return [a for a in PROFILES if regret(table, a, key) == 0]


def select_equilibrium(table):
    """Public ex-ante recommendation: max CALIBRATION potential, cost, lex tie.

    Central finite enumeration is explicit; not a decentralized solver, not an
    optimal contract search and not a population-optimal allocation guarantee.
    """
    ne = equilibria(table)
    if not ne: raise ValueError('no pure NE')
    return min(ne, key=lambda a: (-table[a]['potential'], sum(table[a]['incremental_costs']), a))


def strict_reply_path(table, start=(0, 0, 0), order=(0, 1, 2)):
    state = start
    path = [state]
    for _ in range(len(PROFILES)):
        changed = False
        for i in order:
            candidate = tuple(1-v if j == i else v for j, v in enumerate(state))
            if table[candidate]['utilities'][i] > table[state]['utilities'][i]:
                if table[candidate]['potential'] <= table[state]['potential']:
                    raise ValueError('payoff/potential mismatch')
                state = candidate
                if state in path: raise ValueError('cycle')
                path.append(state)
                changed = True
        if not changed: return path
    raise ValueError('finite improvement bound failed')
