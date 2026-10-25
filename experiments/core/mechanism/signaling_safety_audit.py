"""Audit, NOT an improved consensus algorithm.

The rank-lock bound is a consequence of G's analytical best response:
 log(g(a)/g(b)) = log(g0(a)/g0(b)) + (d(a,+)-d(b,+))/lambda.
Since d belongs to [0,1], an anchor log-margin >1/lambda cannot flip at
an exact response. A numerical epsilon equilibrium is not automatically an
exact-response certificate; the bound is reported with its assumptions.

Quarantine prevents a research wrapper from advertising a changed checker
verdict as verified truth. Its baseline fallback equals the original checker
by construction. It is not a new game-theoretic effectiveness contribution.
"""
from math import isfinite,log
from experiments.core.mechanism.signaling_game import distribution


def rank_lock(choices,reference,strength):
    choices=tuple(choices);p=distribution(reference,len(choices),'generator anchor')
    if len(choices)<2 or len(set(choices))!=len(choices):raise ValueError('invalid choices')
    if type(strength) not in (int,float) or not isfinite(strength) or strength<=0:
        raise ValueError('positive finite strength required')
    order=sorted(range(len(p)),key=lambda i:-p[i]);a,b=order[:2]
    margin=log(p[a])-log(p[b]);bound=1/strength
    return {'anchor_top':choices[a],'anchor_runner_up':choices[b],
            'log_margin':margin,'maximum_opponent_log_odds_shift':bound,
            'exact_response_rank_locked':margin>bound,
            'scope':'any exact G best response to a legal D; does not certify true answer or arbitrary epsilon-policy ranking'}


def quarantine(game_answer,checker_answer,*,inputs_complete,converged):
    if not inputs_complete:reason='incomplete_game_inputs'
    elif not converged:reason='uncertified_numeric_residual'
    elif game_answer!=checker_answer:reason='unresolved_independent_checker_disagreement'
    else:reason='same_as_checker_no_independent_effectiveness_evidence'
    usable=inputs_complete and converged and game_answer==checker_answer
    # Explicit provenance: no silent substitution reported as a successful game.
    return {'game_admissible_as_unchanged_checker_readout':usable,
            'reason':reason,'reported_game_answer':game_answer if usable else None,
            'baseline_answer':checker_answer,
            'operational_fallback_answer':checker_answer,
            'contribution':'baseline fallback, not an effective novel mechanism'}
