import math
import random
import unittest

from experiments.core.mechanism.signaling_game import SignalingGame, from_spec


def example(**kwargs):
    return SignalingGame(('a', 'b'), ((.7, .3), (.2, .8)), ((.75, .25), (.3, .7)), **kwargs)


class SignalingGameTests(unittest.TestCase):
    def test_closed_schema_rejects_label(self):
        with self.assertRaises(ValueError):
            from_spec({'gold': 'a'})

    def test_invalid_inputs(self):
        for kw in ({'lambda_generator': 0}, {'lambda_discriminator': float('nan')},
                   {'type_prior': (.9, .9)}, {'candidate_measure': (0., 1.)}):
            with self.assertRaises(ValueError): example(**kw)
        for rows in (((0., 1.), (.5, .5)), ((.3, .4), (.5, .5)), ((True, False), (.5, .5))):
            with self.assertRaises(ValueError):
                SignalingGame(('a','b'), rows, ((.5,.5),(.5,.5)))

    def test_residual_matches_direct_utility_improvement(self):
        game = example(type_prior=(.3, .7), candidate_measure=(.4,.6))
        g, d = ((.6,.4),(.45,.55)), ((.2,.8),(.65,.35))
        original = game.quantities(g,d)
        residual = game.residual(g,d)
        self.assertAlmostEqual(residual['generator_gap'],
            game.quantities(game.best_generator(d),d)['generator_utility'] - original['generator_utility'])
        self.assertAlmostEqual(residual['discriminator_gap'],
            game.quantities(g,game.best_discriminator(g))['discriminator_utility'] - original['discriminator_utility'])

    def test_exact_potential_under_unilateral_changes(self):
        game = example()
        g,d = game.generator_reference,game.discriminator_reference
        base = game.quantities(g,d)
        for next_g,next_d,key in [(((.4,.6),(.6,.4)),d,'generator_utility'),
                                 (g,((.4,.6),(.8,.2)),'discriminator_utility')]:
            changed = game.quantities(next_g,next_d)
            self.assertAlmostEqual(changed[key]-base[key],changed['potential']-base['potential'])

    def test_best_response_dominates_independent_grid(self):
        game=example()
        g,d=game.generator_reference,game.discriminator_reference
        best_g=game.quantities(game.best_generator(d),d)['generator_utility']
        best_d=game.quantities(g,game.best_discriminator(g))['discriminator_utility']
        for i in range(1,20):
            for j in range(1,20):
                candidate=((i/20,1-i/20),(j/20,1-j/20))
                self.assertLessEqual(game.quantities(candidate,d)['generator_utility'],best_g+1e-12)
                self.assertLessEqual(game.quantities(g,candidate)['discriminator_utility'],best_d+1e-12)

    def test_alternating_response_and_explicit_nonconvergence(self):
        game=example()
        for first in ('generator','discriminator'):
            result=game.solve(first=first)
            self.assertTrue(result['converged_to_tolerance'])
            self.assertLessEqual(result['residual']['max_gap'],1e-9)
            self.assertTrue(all(b['potential']>=a['potential']-1e-12
                                for a,b in zip(result['trace'],result['trace'][1:])))
        result=game.solve(max_sweeps=0)
        self.assertFalse(result['converged_to_tolerance'])

    def test_symmetry_abstains_and_can_be_stably_wrong(self):
        uniform=SignalingGame(('a','b'),((.5,.5),(.5,.5)),((.5,.5),(.5,.5)))
        self.assertIsNone(uniform.solve()['readout']['generator']['answer'])
        wrong=SignalingGame(('wrong','right'),((.99,.01),(.01,.99)),((.99,.01),(.01,.99)))
        result=wrong.solve()
        self.assertTrue(result['converged_to_tolerance'])
        self.assertEqual(result['readout']['generator']['answer'],'wrong')

    def test_candidate_permutation_equivariance(self):
        game=example()
        flipped=SignalingGame(('b','a'),tuple(tuple(reversed(r)) for r in game.generator_reference),
                              tuple(reversed(game.discriminator_reference)))
        a,b=game.solve(),flipped.solve()
        self.assertEqual(a['readout']['generator']['answer'],b['readout']['generator']['answer'])
        for i in range(2):
            self.assertAlmostEqual(a['generator'][i][0],b['generator'][i][1])

    def test_random_simplex_inputs_and_best_response(self):
        rng=random.Random(20261024)
        def row(n):
            v=[rng.random()+.05 for _ in range(n)];s=sum(v)
            return tuple(x/s for x in v)
        for n in range(2,7):
            game=SignalingGame(tuple(map(str,range(n))),tuple(row(n) for _ in range(2)),
                               tuple(row(2) for _ in range(n)),candidate_measure=row(n))
            result=game.solve()
            self.assertTrue(result['converged_to_tolerance'])
            self.assertTrue(math.isfinite(result['residual']['max_gap']))

    def test_numeric_underflow_is_not_silently_clipped(self):
        with self.assertRaises(ArithmeticError):
            example(lambda_generator=1e-10).solve()


if __name__=='__main__':unittest.main()
