"""No-call scope guard: a fitted NE is not a population NE or peer-review NE."""
from hashlib import sha256
import json
import math
from pathlib import Path

from experiments.core.mechanism.certified_effort_game import majority, pivotal_credits
from experiments.mechanism.replay_budgeted_information import ROOT, OUT
from experiments.mechanism.run_upgrade_pilot import freeze

AUDIT = ROOT/'experiments/results/audit/certified_effort_generalization_2026-10-23'


def prospective_hoeffding_radius(scale, n, delta=.05, family=96, high_cost_cap=2000):
    """Hypothetical future iid bounded-cost design calculation, NOT current CI.

    A deviation advantage is lambda*Delta(correct)-Delta(own cost).
    Each oriented deviation has range width 2*lambda+high_cost_cap.
    All 4 coefficients x 8 profiles x 3 unilateral choices = 96 comparisons.
    """
    if scale < 0 or n <= 0 or not 0 < delta < 1 or family < 1 or high_cost_cap <= 0:
        raise ValueError('invalid prospective bound assumptions')
    return (2*scale+high_cost_cap)*math.sqrt(math.log(family/delta)/(2*n))


def peer_influence_counterexample(scale=1000, cost=10):
    """Constructed causal feedback, NOT observed data and NOT valid game input.

    Agent 0's upgrade corrects two peers while its own final vote stays wrong.
    Removing only its final vote leaves the downstream influence in place.
    """
    choices = ['a', 'b']
    before, after = ['a', 'a', 'a'], ['a', 'b', 'b']
    p0 = pivotal_credits(before, choices, 'b', scale)[0]
    p1 = pivotal_credits(after, choices, 'b', scale)[0]
    dv = int(majority(after, choices) == 'b')-int(majority(before, choices) == 'b')
    own_delta = p1-p0-cost
    potential_delta = scale*dv-cost
    return {'constructed_not_model_behavior': True,
            'before_votes': before, 'after_votes': after,
            'agent_zero_credit_before': p0, 'agent_zero_credit_after': p1,
            'own_utility_change': own_delta, 'collective_potential_change': potential_delta,
            'identity_error': potential_delta-own_delta,
            'violated_assumption': 'others final outputs depend on agent zero resource action',
            'conclusion': 'final-vote deletion is not a causal deletion of shared evidence'}


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    source = OUT/'certified_effort_result.json'
    result = json.loads(source.read_text())
    bounds = {}
    for scale, row in result['scenarios_all_scales'].items():
        profile = tuple(row['selected_profile'])
        table = row['evaluation_table_offline_only']
        own = table[str(profile)]['utilities']
        margins = []
        for i in range(3):
            alternative = tuple(1-v if j == i else v for j, v in enumerate(profile))
            margins.append(table[str(alternative)]['utilities'][i]-own[i])
        radius = prospective_hoeffding_radius(int(scale), 24)
        bounds[scale] = {'opened_replay_deviation_means': margins,
            'opened_replay_empirical_regret': max(0., *margins),
            'hypothetical_future_iid_uniform_radius_at_n24': radius,
            'hypothetical_upper_if_data_were_prospectively_eligible': max(0., max(margins)+radius),
            'this_is_NOT_a_confidence_bound_for_current_opened_data': True}
    audit = {'scope': 'no-call mathematical scope audit; no changed pay rule or route selection',
        'new_model_calls': 0, 'source_sha256': sha256(source.read_bytes()).hexdigest(),
        'counterexample_peer_influence': peer_influence_counterexample(),
        'prospective_bound_template': bounds,
        'requirements_for_template': ['fixed action space, aggregator, coefficients and cost cap',
            'fresh independent iid task/output samples from the claimed target distribution',
            'no current-question label in action selection; trusted delayed scoring',
            'same authenticated-output production subgame; no downstream peer feedback',
            'uniform family of 96 comparisons at total delta=.05'],
        'warnings': ['the current samples are already opened; no 95% population-NE certificate is produced',
            'stratified/native-source sampling and changing backends require an appropriate design, not an automatic iid assumption',
            'Hoeffding is conservative; wide radius is not a lower bound on all possible sample requirements',
            'zero empirical regret is not proof of zero true regret',
            'adding peer-guided report updates requires a new counterfactual/reference proof',
            'delayed true outcome is an applicability condition, not an online oracle']}
    freeze(AUDIT/'result.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == '__main__': main()
