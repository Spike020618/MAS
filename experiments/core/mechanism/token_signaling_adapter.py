"""Explicit token-coded adapter; NOT original full-sequence likelihood scoring.

All four input distributions must be observed, full support, and over exact
codes A/B. Optional cross-normalization follows the baseline's contrastive
normalization idea on this restricted space. No inferred missing probabilities.
"""
from experiments.core.mechanism.signaling_game import SignalingGame, distribution


SLOTS=('generate_positive','generate_negative','verify_A','verify_B')


def game_from_probe(slots, strength=.5, normalization='conditional'):
    if set(slots)!=set(SLOTS):raise ValueError('all four observed slots required')
    probabilities=[]
    for slot in SLOTS:
        observed=slots[slot]
        if not observed.get('usable'):raise ValueError('unusable observed slot')
        masses=observed.get('conditional_probabilities',{})
        if set(masses)!=set(('A','B')):raise ValueError('exact A/B codes required')
        probabilities.append(distribution((masses['A'],masses['B']),2,slot))
    g=probabilities[:2];d=probabilities[2:]
    if normalization=='contrastive':
        # G cross-normalizes across abstract types, then normalizes each row.
        gtmp=[[g[v][y]/sum(g[u][y] for u in range(2)) for y in range(2)] for v in range(2)]
        g=[tuple(x/sum(row) for x in row) for row in gtmp]
        # D cross-normalizes across candidates, then normalizes each row.
        dtmp=[[d[y][v]/sum(d[z][v] for z in range(2)) for v in range(2)] for y in range(2)]
        d=[tuple(x/sum(row) for x in row) for row in dtmp]
    elif normalization!='conditional':raise ValueError('unknown normalization')
    return SignalingGame(('A','B'),tuple(g),tuple(d),strength,strength)
