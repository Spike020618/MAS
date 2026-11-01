"""Offline decision-agent environment, not a claim about natural LLM rationality.

No filesystem/network/model imports. The agent sees its own initial observation
and a calibration estimate, never the cached upgrade or the evaluation outcome.
Private Python objects are an API boundary, NOT an adversarial process sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from hashlib import sha256
import json
import math


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False).encode()).hexdigest()


def valid_probability(q, size):
    return (q is not None and len(q) == size and size >= 2
            and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in q)
            and sum(Fraction(str(x)) for x in q) == 1)


@dataclass(frozen=True)
class Report:
    answer: str
    probabilities: tuple | None

    def valid(self, choices):
        return (valid_probability(self.probabilities, len(choices))
                and self.answer in choices
                and self.probabilities[choices.index(self.answer)] >= max(self.probabilities)-1e-12)

    def score(self, outcome, choices):
        if outcome not in choices:
            raise ValueError('unregistered outcome')
        if not self.valid(choices):
            return 0.0  # Declared interface failure rule, never impute a distribution.
        return 1 - .5 * sum((v - int(choices[j] == outcome)) ** 2
                            for j, v in enumerate(self.probabilities))


@dataclass(frozen=True)
class Observation:
    case_id: str
    agent: int
    choices: tuple
    initial: Report
    initial_tokens: int


@dataclass(frozen=True)
class Estimate:
    mean_score_gain: float
    mean_upgrade_tokens: float
    n: int
    lower_score_gain: float
    cell: str


@dataclass(frozen=True)
class Contract:
    score_weight: float
    wallet: int = 3000
    upgrade_reservation: int = 2000

    def __post_init__(self):
        if not math.isfinite(self.score_weight) or self.score_weight < 0:
            raise ValueError('nonnegative finite score weight required')
        if type(self.wallet) is not int or type(self.upgrade_reservation) is not int:
            raise ValueError('integer token budgets required')
        if self.wallet < 0 or self.upgrade_reservation <= 0:
            raise ValueError('invalid budget')


@dataclass(frozen=True)
class Decision:
    action: str
    expected_upgrade_advantage: float
    reason: str


def choose(observation, estimate, contract, conservative=False):
    """Best reply to a DECLARED estimated payoff, not to realized gold payoffs."""
    if observation.initial_tokens < 0 or observation.initial_tokens > contract.wallet:
        raise ValueError('initial resource charge exceeds wallet')
    if not observation.initial.valid(observation.choices):
        return Decision('K', 0., 'invalid_initial_probability_no_imputation')
    if estimate.n <= 0 or estimate.mean_upgrade_tokens <= 0:
        return Decision('K', 0., 'no_calibration_support')
    gain = estimate.lower_score_gain if conservative else estimate.mean_score_gain
    advantage = contract.score_weight * gain - estimate.mean_upgrade_tokens
    if contract.wallet - observation.initial_tokens < contract.upgrade_reservation:
        return Decision('K', advantage, 'insufficient_private_budget')
    return Decision('U' if advantage > 1e-10 else 'K', advantage,
                    'positive_estimated_net_value' if advantage > 1e-10 else 'keep_on_nonpositive_value')


class ReplayResource:
    """Trusted replay broker; recorded high output is released only on purchase.

    The broker is NOT an agent. Cached alternatives were already collected in
    the source studies. Receipts represent replayed usage, not new API charges.
    """
    def __init__(self, upgrades):
        self.__upgrades = dict(upgrades)
        self.__used = set()

    def purchase(self, case_id, agent, reservation):
        key = (case_id, agent)
        if key in self.__used:
            raise ValueError('duplicate resource purchase')
        report, tokens, source_hash = self.__upgrades[key]
        if type(tokens) is not int or not 0 < tokens <= reservation:
            raise ValueError('recorded usage outside reservation; cannot silently overspend')
        self.__used.add(key)
        return report, {'case_id': case_id, 'agent': agent, 'tokens': tokens,
                        'source_response_hash': source_hash, 'replayed_not_new_call': True}


class Episode:
    """Fixed 3-agent sealed submission, aggregate, then delayed outcome settlement."""
    def __init__(self, observations, contract):
        if len(observations) != 3 or {o.agent for o in observations} != {0, 1, 2}:
            raise ValueError('three distinct agents required')
        if len({(o.case_id, o.choices) for o in observations}) != 1:
            raise ValueError('one question and choice registry required')
        if any(not 0 <= o.initial_tokens <= contract.wallet for o in observations):
            raise ValueError('initial budget exceeded')
        self.__observations = {o.agent: o for o in observations}
        self.contract = contract
        self.__sealed = {}
        self.__aggregated = False
        self.__settled = False
        self.events = []

    def submit(self, agent, decision, broker):
        if agent in self.__sealed or self.__aggregated:
            raise ValueError('submission already sealed')
        o = self.__observations[agent]
        if decision.action not in ('K', 'U'):
            raise ValueError('unknown action')
        report, receipt = o.initial, None
        spent = o.initial_tokens
        if decision.action == 'U':
            if self.contract.wallet - spent < self.contract.upgrade_reservation:
                raise ValueError('private budget cannot cover reservation')
            report, receipt = broker.purchase(o.case_id, agent, self.contract.upgrade_reservation)
            spent += receipt['tokens']
        record = {'agent': agent, 'action': decision.action, 'report': asdict(report),
                  'tokens': spent, 'initial_tokens': o.initial_tokens, 'receipt': receipt,
                  'accepted_answer': ('abstain' if valid_probability(report.probabilities, len(o.choices))
                                      and not report.valid(o.choices) else report.answer),
                  'rejected_inconsistent_conclusion': bool(valid_probability(report.probabilities, len(o.choices))
                                                          and not report.valid(o.choices))}
        self.__sealed[agent] = record
        commitment = digest(record)
        self.events.append({'event': 'sealed', 'agent': agent, 'commitment': commitment})
        return commitment

    def aggregate(self):
        if len(self.__sealed) != 3 or self.__aggregated:
            raise ValueError('exactly one aggregation after all seals required')
        choices = self.__observations[0].choices
        votes = [self.__sealed[i]['accepted_answer'] for i in range(3)]
        answer = next((c for c in choices if votes.count(c) >= 2), 'abstain')
        self.__aggregated = True
        self.events.append({'event': 'aggregate', 'answer': answer})
        return answer

    def settle(self, outcome):
        if not self.__aggregated or self.__settled:
            raise ValueError('settlement only once after sealed aggregation')
        choices = self.__observations[0].choices
        if outcome not in choices:
            raise ValueError('unregistered outcome')
        records = []
        for agent in range(3):
            record = self.__sealed[agent]
            report = Report(record['report']['answer'],
                            tuple(record['report']['probabilities']) if record['report']['probabilities'] is not None else None)
            value = report.score(outcome, choices)
            credit = self.contract.score_weight * value
            records.append(dict(record, score=value, credit=credit,
                                utility=credit-record['tokens'],
                                wallet_remaining=self.contract.wallet-record['tokens']))
        self.__settled = True
        self.events.append({'event': 'delayed_settlement', 'outcome': outcome})
        return {'records': records, 'events': list(self.events),
                'aggregate_commitment': digest(self.events[:-1]),
                'total_credit': sum(r['credit'] for r in records),
                'max_credit': 3*self.contract.score_weight}
