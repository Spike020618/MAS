"""Analysis rules frozen before P1 response collection; no outcome-based subset."""
from .schema import digest
PROTOCOL={
 'version':'coordination-p1-analysis-v1','tasks':12,'repeats':2,'branches_per_repeat':16,
 'engineering':{'minimum_valid_fraction':.95,'maximum_budget_violations':0,'maximum_leakage_events':0},
 'unit':'question; average repeats before paired bootstrap','bootstrap_seed':20261030,'bootstrap_draws':4000,
 'metrics':['answer_f1','answer_em','corrected','harmed','invalid'],
 'science_gate':'review_required: report replicated state/combination effects across all questions, no automatic cherry-picked pass',
 'primary_scope':'opportunity diagnosis; estimator gain and solver gain not measured',
 'comparisons':['single_vs_empty','same_member_AB_vs_empty','cross_member_AB_vs_empty','cross_member_AB_vs_same_member_AB'],
 'invalid_policy':'score recorded end-to-end fallback answer; validity reported separately; do not replace failed output with gold',
}
def frozen_protocol():return {'protocol':PROTOCOL,'sha256':digest(PROTOCOL)}
