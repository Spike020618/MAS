"""Exact known-channel UNIT FIXTURE, not an empirical LLM method.

Three agents choose acquisition BEFORE receiving private signals. State is a
fair binary prior; purchased channels are conditionally independent given it.
Receipts certify acquisition, not the truth of a report. Reports are private
signal symbols, not AEIC explanations or arbitrary uncalibrated probabilities.
"""
from fractions import Fraction as F
from itertools import product


def score(p, y):
    return 1-(p-y)**2


def posterior(qualities, reports):
    likelihoods = []
    for y in (0, 1):
        value = F(1, 2)
        for quality, signal in zip(qualities, reports):
            if signal:
                value *= quality if int(signal == 1) == y else 1-quality
        likelihoods.append(value)
    if sum(likelihoods) == 0:
        raise ValueError('impossible signal combination')
    return likelihoods[1]/sum(likelihoods)


def outcomes(qualities, actions):
    for y in (0, 1):
        for signals in product(*[(-1, 1) if a else (0,) for a in actions]):
            probability = F(1, 2)
            for q, a, signal in zip(qualities, actions, signals):
                if a: probability *= q if int(signal == 1) == y else 1-q
            yield y, signals, probability


def expected(qualities, costs, actions, scale=F(1), deviation=None, individual=False):
    """Integrate generative prior, never select an action from realized gold.

    deviation=(agent, mapping of received -/+ signals to reported -/0/+).
    Every strategy is chosen before the world draw; peers keep truthful maps.
    """
    if len(qualities) != 3 or len(costs) != 3 or len(actions) != 3:
        raise ValueError('exactly three agents required')
    if any(not F(1, 2) < q < 1 for q in qualities) or any(c < 0 for c in costs):
        raise ValueError('invalid known channel or cost')
    if any(a not in (0, 1) for a in actions) or scale < 0:
        raise ValueError('invalid actions or scale')
    payments = [F(0)]*3
    value = F(0)
    accuracy = F(0)
    for y, signals, probability in outcomes(qualities, actions):
        reports = list(signals)
        if deviation:
            agent, mapping = deviation
            if actions[agent]: reports[agent] = mapping[int(signals[agent] == 1)]
        full = score(posterior(qualities, reports), y)
        value += probability*full
        accuracy += probability*int(int(posterior(qualities, reports) >= F(1, 2)) == y)
        for i in range(3):
            without = list(reports)
            without[i] = 0
            reference = score(posterior(qualities, without), y)
            own = [0]*3
            own[i] = reports[i]
            credit = scale/F(2)*(1+(score(posterior(qualities, own), y)-score(F(1, 2), y) if individual else full-reference))
            if not 0 <= credit <= scale:
                raise ValueError('bounded payment violated')
            payments[i] += probability*credit
    utility = [payments[i]-costs[i]*actions[i] for i in range(3)]
    return {'expected_collective_score': value, 'expected_accuracy': accuracy,
            'expected_payments': payments, 'expected_utilities': utility,
            'cost': sum(c*a for c, a in zip(costs, actions)),
            'potential': scale/F(2)*value-sum(c*a for c, a in zip(costs, actions))}


def normal_form(qualities, costs, scale=F(1), individual=False):
    return {a: expected(qualities, costs, a, scale, individual=individual)
            for a in product((0, 1), repeat=3)}


def equilibria(table):
    result = []
    for action, row in table.items():
        if all(table[tuple(1-a if j == i else a for j, a in enumerate(action))]['expected_utilities'][i]
               <= row['expected_utilities'][i] for i in range(3)):
            result.append(action)
    return result


def full_strategy_regret(qualities, costs, actions, scale=F(1)):
    """All unilateral acquisition + finite report-mapping deviations, exact."""
    truthful = expected(qualities, costs, actions, scale)
    regret = F(0)
    for i in range(3):
        for new_action in (0, 1):
            changed = list(actions)
            changed[i] = new_action
            for mapping in product((-1, 0, 1), repeat=2) if new_action else ((0, 0),):
                payoff = expected(qualities, costs, tuple(changed), scale, (i, mapping))['expected_utilities'][i]
                regret = max(regret, payoff-truthful['expected_utilities'][i])
    return regret


def correlated_duplicate_counterexample(q=F(4, 5)):
    """One common signal copied 3 times; independent-channel fusion is WRONG."""
    correct_p = q
    wrongly_fused_p = posterior((q,)*3, (1,)*3)
    truthful_value = q*score(correct_p, 1)+(1-q)*score(correct_p, 0)
    duplicate_value = q*score(wrongly_fused_p, 1)+(1-q)*score(wrongly_fused_p, 0)
    return {'true_posterior': correct_p, 'wrong_independence_posterior': wrongly_fused_p,
            'true_expected_score': truthful_value, 'duplicate_expected_score': duplicate_value,
            'loss': truthful_value-duplicate_value,
            'accuracy_unchanged': q}
