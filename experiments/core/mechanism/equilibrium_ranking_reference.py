"""Equation-level Consensus Game reference (ICLR 2024, Sec. 2.2).

Consumes full candidate-sequence log likelihoods, not sampled token codes.
No benchmark labels, network calls, parameter fitting, or Nash guarantee.
Implements simultaneous piKL equations (1)-(2), not block best responses.
"""
from math import exp, fsum, isfinite, log


def softmax(values):
    values = tuple(values)
    if not values or any(not isfinite(v) for v in values):
        raise ValueError('finite complete log likelihoods required')
    peak = max(values)
    masses = tuple(exp(v - peak) for v in values)
    if min(masses) == 0:
        raise ValueError('floating point underflow; do not clip missing mass')
    total = fsum(masses)
    return tuple(v / total for v in masses)


def anchors(generator_logs, discriminator_logs):
    """Cross-normalize raw likelihoods, then normalize information sets.

    G shape [2][n]; D shape [n][2]. Any task-specific unconditional
    likelihood correction must already be applied and recorded by caller.
    """
    g = tuple(tuple(row) for row in generator_logs)
    d = tuple(tuple(row) for row in discriminator_logs)
    if len(g) != 2 or len(g[0]) < 2 or len(g[1]) != len(g[0]):
        raise ValueError('G must have shape 2 by n, n >= 2')
    n = len(g[0])
    if len(d) != n or any(len(row) != 2 for row in d):
        raise ValueError('D must have shape n by 2')
    gcross = [softmax((g[0][y], g[1][y])) for y in range(n)]
    dcross = [softmax(tuple(d[y][v] for y in range(n))) for v in range(2)]
    return (tuple(softmax(tuple(log(gcross[y][v]) for y in range(n))) for v in range(2)),
            tuple(softmax((log(dcross[0][y]), log(dcross[1][y]))) for y in range(n)))


def solve(generator_logs, discriminator_logs, iterations=5000, eta=.1, strength=.1):
    """Paper defaults; iterations counts simultaneous update steps.

    Return last policies and average policies separately, never confuse
    empirical last-iterate stability with the paper's correlated-play result.
    """
    if type(iterations) is not int or iterations < 1:
        raise ValueError('positive integer update count required')
    if any(type(x) not in (int, float) or not isfinite(x) or x <= 0
           for x in (eta, strength)):
        raise ValueError('positive finite parameters required')
    g0, d0 = anchors(generator_logs, discriminator_logs)
    g, d = g0, d0
    n = len(d)
    sg = [[0.] * n for _ in range(2)]
    sd = [[0.] * 2 for _ in range(n)]
    ag = [[0.] * n for _ in range(2)]
    ad = [[0.] * 2 for _ in range(n)]
    for t in range(1, iterations + 1):
        for v in range(2):
            for y in range(n):
                sg[v][y] += .5 * d[y][v]
                sd[y][v] += .5 * g[v][y]
                ag[v][y] += g[v][y]
                ad[y][v] += d[y][v]
        denominator = 1 / (eta * t) + strength
        # Both updates use the same history through t, not the fresh G.
        ng = tuple(softmax(tuple((sg[v][y] / t + strength * log(g0[v][y]))
                                / denominator for y in range(n))) for v in range(2))
        nd = tuple(softmax(tuple((sd[y][v] / t + strength * log(d0[y][v]))
                                / denominator for v in range(2))) for y in range(n))
        g, d = ng, nd
    return {'generator': g, 'discriminator': d,
            'average_generator': tuple(tuple(x / iterations for x in row) for row in ag),
            'average_discriminator': tuple(tuple(x / iterations for x in row) for row in ad),
            'updates': iterations, 'eta': eta, 'lambda': strength,
            'scope': 'equation-level reference; not full experimental reproduction or correctness proof'}
