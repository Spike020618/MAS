"""Structural repair CONTROLS, not novel algorithms or solved leader games.

Frozen D removes endogenous checker corruption but is no longer a two-player
joint equilibrium. Uniform G anchor removes rank lock, but then its best
response ranks exactly like D: exp(D/tau) preserves order. This equivalence
is a stopping reason, not an accuracy contribution to claim for game theory.
"""
from experiments.core.mechanism.signaling_game import SignalingGame
from experiments.core.mechanism.scored_signaling import rank


def frozen_checker_controls(choices,generator_reference,discriminator_reference):
    choices=tuple(choices);n=len(choices);answers={};policies={}
    for mode in ('freeze_checker','uniform_anchor_freeze_checker'):
        g=generator_reference if mode=='freeze_checker' else ((1/n,)*n,)*2
        game=SignalingGame(choices,tuple(g),tuple(discriminator_reference))
        response=game.best_generator(discriminator_reference)
        answers[mode]=rank(choices,response[0]);policies[mode]=response
    direct=rank(choices,[r[0] for r in discriminator_reference])
    return {'answers':answers,'generator_responses':policies,'direct_checker':direct,
            'uniform_readout_equals_checker':answers['uniform_anchor_freeze_checker']==direct,
            'scope':'best responses to a fixed checker, not learned Stackelberg equilibrium; uniform variant is a baseline-equivalent control'}
