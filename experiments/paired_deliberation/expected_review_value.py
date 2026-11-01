"""Risk-aware acquisition proxy, with an explicit non-review counterfactual.

Assume an uncertain candidate-swap quality difference D ~ Normal(mu,sigma^2).
Assume a review picks the better candidate with symmetric accuracy q independent
of |D|. Neither assumption is asserted true for an LM; estimate/check separately.
This is not a semantic verifier or a calibrated posterior by itself.
"""
import math


def positive_part_normal(mu,sigma):
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma<0:raise ValueError('finite mean and nonnegative standard deviation required')
    if sigma==0:return max(mu,0.)
    z=mu/sigma
    cdf=.5*math.erfc(-z/math.sqrt(2.))
    pdf=math.exp(-.5*z*z)/math.sqrt(2.*math.pi)
    return max(0.,sigma*pdf+mu*cdf)


def expected_review_value(mu,sigma,accuracy,cost=0.,cost_weight=1.,baseline='best_mean'):
    if not math.isfinite(accuracy) or not 0<=accuracy<=1:raise ValueError('review accuracy must be within [0,1]')
    if not math.isfinite(cost) or cost<0 or not math.isfinite(cost_weight) or cost_weight<0:raise ValueError('nonnegative finite cost/weight')
    if baseline not in ('retain','best_mean'):raise ValueError('explicit baseline required')
    positive=positive_part_normal(mu,sigma)
    negative=max(0.,positive-mu)
    expected=accuracy*positive-(1-accuracy)*negative
    direct=max(0.,mu) if baseline=='best_mean' else 0.
    incremental=expected-direct
    net=incremental-cost_weight*cost
    return {'mean_delta':mu,'std_delta':sigma,'assumed_symmetric_accuracy':accuracy,
        'perfect_information_gain_over_retain':positive,'expected_damage_if_wrong':negative,
        'review_gain_over_retain':expected,'non_review_baseline':baseline,'non_review_gain':direct,
        'incremental_review_gain':incremental,'cost':cost,'cost_weight':cost_weight,'net_value':net,
        'participation_candidate':net>1e-12,
        'scope':'analytic under explicit Normal/symmetric-q assumptions; q=1 is oracle acquisition upper bound, not LM effect',
        'count_units':'mu may represent expected repeated-target count difference; no serving-time future target-count weight'}


def asymmetric_review_value(mu,sigma,recover_b,retain_a,cost=0.,cost_weight=1.,baseline='best_mean'):
    """Effective actions may include rejected/no-op reviews as safe retention.

    recover_b=P(effective swap | b better); retain_a=P(no harmful swap | a better).
    These conditional rates are NOT unconditional semantic verifier accuracy.
    """
    for q in (recover_b,retain_a):
        if not math.isfinite(q) or not 0<=q<=1:raise ValueError('conditional action rates must be within [0,1]')
    if baseline not in ('retain','best_mean'):raise ValueError('explicit baseline required')
    if not math.isfinite(cost) or cost<0 or not math.isfinite(cost_weight) or cost_weight<0:raise ValueError('nonnegative finite cost/weight')
    positive=positive_part_normal(mu,sigma);negative=max(0.,positive-mu)
    expected=recover_b*positive-(1-retain_a)*negative
    direct=max(0.,mu) if baseline=='best_mean' else 0.
    net=expected-direct-cost_weight*cost
    return {'mean_delta':mu,'std_delta':sigma,'assumed_recover_b':recover_b,'assumed_retain_a':retain_a,
        'perfect_information_gain_over_retain':positive,'negative_part_expected_count':negative,
        'review_gain_over_retain':expected,'non_review_baseline':baseline,'non_review_gain':direct,
        'incremental_review_gain':expected-direct,'cost':cost,'cost_weight':cost_weight,'net_value':net,
        'participation_candidate':net>1e-12,
        'scope':'Normal difference and conditional action-rate proxy only; sampled training-stratum rates do not certify deployment utility'}
