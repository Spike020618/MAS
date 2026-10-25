"""Explicit regularized signaling-game BASELINE, not a new truth mechanism.

G selects a candidate conditional on an abstract message type v; D infers v.
Neither v nor the common payoff is the benchmark answer key. Full-support
reference policies anchor the semantics but can themselves be wrong.

Weights rho(v), mu(y) are explicit. This is a documented weighted variant of
the Consensus Game, not a claim to reproduce its exact scoring/normalization.
Alternating exact block responses ascend a common potential. We measure each
returned policy's numerical best-response gap; no last-iterate convergence,
global optimum, truthfulness or accuracy guarantee is asserted.
"""
from dataclasses import dataclass
from math import exp, fsum, isfinite, log


def distribution(values, size, name):
    values = tuple(values)
    if len(values) != size or any(type(x) not in (int, float) or not isfinite(x)
                                   or x <= 0 for x in values):
        raise ValueError(name + ': finite, strictly positive probabilities required')
    if abs(fsum(values) - 1) > 1e-10:
        raise ValueError(name + ': probabilities must sum to one; no silent normalization')
    return values


def _matrix(rows, nrows, ncols, name):
    rows = tuple(rows)
    if len(rows) != nrows:
        raise ValueError(name + ': wrong number of information sets')
    return tuple(distribution(row, ncols, name) for row in rows)


def _kl(p, q):
    return fsum(x * (log(x) - log(y)) for x, y in zip(p, q))


def _response(anchor, values, strength):
    logits = [log(p) + v / strength for p, v in zip(anchor, values)]
    peak = max(logits)
    masses = [exp(v - peak) for v in logits]
    if any(m == 0 or not isfinite(m) for m in masses):
        raise ArithmeticError('best response exceeds floating-point range; no implicit clipping')
    denominator = fsum(masses)
    return tuple(m / denominator for m in masses)


@dataclass(frozen=True)
class SignalingGame:
    candidates: tuple
    generator_reference: tuple  # [v][y], v=0 affirmative, v=1 negative
    discriminator_reference: tuple  # [y][v]
    lambda_generator: float = 0.5
    lambda_discriminator: float = 0.5
    type_prior: tuple = (0.5, 0.5)
    candidate_measure: tuple = ()

    def __post_init__(self):
        candidates = tuple(self.candidates)
        if (len(candidates) < 2 or any(not isinstance(c, str) or not c for c in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError('at least two distinct nonempty candidate IDs required')
        n = len(candidates)
        object.__setattr__(self, 'candidates', candidates)
        object.__setattr__(self, 'generator_reference',
                           _matrix(self.generator_reference, 2, n, 'G reference'))
        object.__setattr__(self, 'discriminator_reference',
                           _matrix(self.discriminator_reference, n, 2, 'D reference'))
        object.__setattr__(self, 'type_prior', distribution(self.type_prior, 2, 'type prior'))
        measure = self.candidate_measure or (1 / n,) * n
        object.__setattr__(self, 'candidate_measure', distribution(measure, n, 'candidate measure'))
        for value in (self.lambda_generator, self.lambda_discriminator):
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError('strictly positive finite regularization required')

    def policies(self, generator, discriminator):
        n = len(self.candidates)
        return (_matrix(generator, 2, n, 'G policy'),
                _matrix(discriminator, n, 2, 'D policy'))

    def quantities(self, generator, discriminator):
        g, d = self.policies(generator, discriminator)
        common = fsum(self.type_prior[v] * g[v][y] * d[y][v]
                      for v in range(2) for y in range(len(self.candidates)))
        kg = self.lambda_generator * fsum(self.type_prior[v] * _kl(g[v], self.generator_reference[v])
                                           for v in range(2))
        kd = self.lambda_discriminator * fsum(self.candidate_measure[y] * _kl(d[y], self.discriminator_reference[y])
                                               for y in range(len(self.candidates)))
        return {'common_payoff': common, 'generator_utility': common - kg,
                'discriminator_utility': common - kd, 'potential': common - kg - kd}

    def best_generator(self, discriminator):
        d = _matrix(discriminator, len(self.candidates), 2, 'D policy')
        return tuple(_response(self.generator_reference[v], [row[v] for row in d],
                               self.lambda_generator) for v in range(2))

    def best_discriminator(self, generator):
        g = _matrix(generator, 2, len(self.candidates), 'G policy')
        return tuple(_response(self.discriminator_reference[y],
                               [self.type_prior[v] * g[v][y] / self.candidate_measure[y]
                                for v in range(2)], self.lambda_discriminator)
                     for y in range(len(self.candidates)))

    def residual(self, generator, discriminator):
        """Analytic full-simplex best responses, evaluated in floating point.

        Gibbs identity: response gain = lambda * weighted KL(current || BR).
        This is a numerical residual, NOT interval-arithmetic certification.
        """
        g, d = self.policies(generator, discriminator)
        bg, bd = self.best_generator(d), self.best_discriminator(g)
        rg = self.lambda_generator * fsum(self.type_prior[v] * _kl(g[v], bg[v]) for v in range(2))
        rd = self.lambda_discriminator * fsum(self.candidate_measure[y] * _kl(d[y], bd[y])
                                               for y in range(len(self.candidates)))
        return {'generator_gap': max(0., rg), 'discriminator_gap': max(0., rd),
                'max_gap': max(0., rg, rd),
                'scope': 'numerical epsilon-Nash residual in this fixed regularized game; not truth or global optimality'}

    def solve(self, tolerance=1e-9, max_sweeps=2000, first='generator'):
        if type(tolerance) not in (int, float) or not isfinite(tolerance) or tolerance <= 0:
            raise ValueError('positive finite tolerance required')
        if type(max_sweeps) is not int or max_sweeps < 0 or first not in ('generator', 'discriminator'):
            raise ValueError('invalid sweep limit or update order')
        g, d = self.generator_reference, self.discriminator_reference
        trace = []
        for step in range(max_sweeps + 1):
            values = self.quantities(g, d)
            residual = self.residual(g, d)
            trace.append({'sweep': step, **values, **residual})
            if residual['max_gap'] <= tolerance or step == max_sweeps:
                break
            if first == 'generator':
                g = self.best_generator(d)
                d = self.best_discriminator(g)
            else:
                d = self.best_discriminator(g)
                g = self.best_generator(d)
        return {'generator': g, 'discriminator': d, 'trace': trace,
                'converged_to_tolerance': residual['max_gap'] <= tolerance,
                'residual': residual, 'sweeps': step, 'update_order': first,
                'readout': self.readout(g, d)}

    def readout(self, generator, discriminator):
        """Affirmative-type G ranking; other simple rankings exposed as controls.

        No label access; exact/near numerical ties abstain rather than use ID order.
        """
        g, d = self.policies(generator, discriminator)
        def rank(scores):
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            answer = self.candidates[order[0]] if scores[order[0]] - scores[order[1]] > 1e-10 else None
            return {'answer': answer, 'scores': dict(zip(self.candidates, scores))}
        return {'generator': rank(g[0]), 'discriminator': rank([r[0] for r in d]),
                'product': rank([g[0][i] * d[i][0] for i in range(len(g[0]))])}


def from_spec(spec):
    """Closed inference schema deliberately has no gold, domain or task text."""
    allowed = {'candidates', 'generator_reference', 'discriminator_reference',
               'lambda_generator', 'lambda_discriminator', 'type_prior', 'candidate_measure'}
    if set(spec) - allowed:
        raise ValueError('unknown game fields: ' + ','.join(sorted(set(spec) - allowed)))
    return SignalingGame(**spec)
