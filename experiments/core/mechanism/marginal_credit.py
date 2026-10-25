"""Bounded delayed-outcome difference credit for sealed independent AEIC reports.

Classical difference-reward construction, not a truthfulness theorem. Requires
a trusted outcome AFTER reports are sealed. No outcome => pending, not pseudo
truth inferred from agreement. Does not model downstream peer reactions.
"""
from dataclasses import dataclass
import hashlib
import json
import math


def nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative number required')
    return float(value)


def decision(stances):
    if any(s not in ('yes', 'no', 'abstain', None) for s in stances):
        raise ValueError('invalid stance')
    # Explicit abstentions count toward quorum; None is only the counterfactual
    # removed input. Three registered reports therefore require two equal votes.
    quorum = sum(s is not None for s in stances)//2 + 1
    return next((s for s in ('yes', 'no') if stances.count(s) >= quorum), None)


def team_score(stances, label, score_mode='accuracy'):
    if label not in ('yes', 'no'): raise ValueError('binary verified label required')
    if score_mode == 'accuracy':
        return float(decision(stances) == label)
    if score_mode == 'brier':
        if not stances: raise ValueError('fixed panel required')
        if any(s not in ('yes', 'no', 'abstain', None) for s in stances):
            raise ValueError('invalid stance')
        # Fixed registered denominator. Removed/abstaining inputs are neutral.
        # This is a team score proxy, NOT a model's calibrated probability.
        p = sum(1 if s == 'yes' else 0 if s == 'no' else .5 for s in stances)/len(stances)
        return 1-(p-int(label == 'yes'))**2
    raise ValueError('unknown score mode')


def seal(reports):
    if not reports: raise ValueError('registered reports required')
    for agent, report in reports.items():
        if not isinstance(agent, str) or not agent: raise ValueError('registered string ID required')
        if not isinstance(report, dict) or set(report) != {'A', 'E', 'I', 'C'}:
            raise ValueError('complete AEIC schema required before settlement')
        if report['C'] not in ('yes', 'no', 'abstain'):
            raise ValueError('explicit conclusion required; parse failure is not abstain')
        if not isinstance(report['A'], str) or not isinstance(report['I'], str):
            raise ValueError('A/I strings required')
        if not isinstance(report['E'], list) or not all(isinstance(e, str) for e in report['E']):
            raise ValueError('E string list required')
    raw = json.dumps(reports, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class VerifiedOutcome:
    label: str
    provenance: str

    def __post_init__(self):
        if self.label not in ('yes', 'no') or not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError('verified binary outcome and provenance required')


def settle(reports, costs, budget, commitment, outcome=None, score_mode='accuracy'):
    """Registered identities fixed before execution; payment not real money.

    Outcome provenance is a trusted-caller assumption, not authenticated here.
    `costs` must reflect own action only for the exact-potential theorem.
    """
    if seal(reports) != commitment: raise ValueError('sealed reports changed')
    if set(costs) != set(reports): raise ValueError('one own-action cost per agent required')
    costs = {a: nonnegative(c) for a, c in costs.items()}
    budget = nonnegative(budget)
    if score_mode not in ('accuracy', 'brier'): raise ValueError('unknown score mode')
    if outcome is None:
        return {'status': 'pending_verified_outcome', 'credits': None, 'utilities': None,
                'paid': None, 'costs': costs, 'commitment': commitment}
    if not isinstance(outcome, VerifiedOutcome): raise ValueError('trusted outcome object required')
    agents = sorted(reports)
    votes = [reports[a]['C'] for a in agents]
    full = team_score(votes, outcome.label, score_mode)
    beta = budget / (2 * len(agents))
    credits, components = {}, {}
    for i, agent in enumerate(agents):
        without = votes.copy(); without[i] = None
        reference = team_score(without, outcome.label, score_mode)
        delta = full - reference
        credits[agent] = beta * (1 + delta)
        components[agent] = {'participation': beta, 'marginal_performance': beta * delta,
                             'score_without_agent': reference, 'delta': delta}
    return {'status': 'settled', 'score': full, 'score_mode': score_mode, 'decision': decision(votes),
            'beta': beta, 'budget': budget, 'credits': credits, 'components': components,
            'utilities': {a: credits[a]-costs[a] for a in agents}, 'costs': costs,
            'paid': sum(credits.values()),
            'potential': beta * full - sum(costs.values()),
            'commitment': commitment, 'outcome_provenance': outcome.provenance,
            'scope': 'sealed-report direct aggregation contribution; not causal peer influence, report truthfulness, or autonomous LLM equilibrium'}
