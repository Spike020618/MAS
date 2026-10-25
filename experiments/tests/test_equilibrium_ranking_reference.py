import math
import unittest

from experiments.core.mechanism.equilibrium_ranking_reference import anchors, solve, softmax


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.g = ((math.log(.8), math.log(.2)), (math.log(.3), math.log(.7)))
        self.d = ((math.log(.7), math.log(.3)), (math.log(.4), math.log(.6)))

    def test_cross_normalization(self):
        g, d = anchors(self.g, self.d)
        self.assertAlmostEqual(g[0][0], (.8/1.1)/(.8/1.1+.2/.9))
        self.assertAlmostEqual(d[0][0], (.7/1.1)/(.7/1.1+.3/.9))

    def test_first_simultaneous_update_matches_equations(self):
        g, d = anchors(self.g, self.d)
        r = solve(self.g, self.d, iterations=1)
        for v in range(2):
            expected = softmax(tuple((.5*d[y][v]+.1*math.log(g[v][y]))/10.1 for y in range(2)))
            self.assertEqual(r['generator'][v], expected)
        for y in range(2):
            expected = softmax(tuple((.5*g[v][y]+.1*math.log(d[y][v]))/10.1 for v in range(2)))
            self.assertEqual(r['discriminator'][y], expected)

    def test_uniform_symmetry(self):
        r = solve(((0.,0.,0.),)*2, ((0.,0.),)*3)
        self.assertEqual(r['generator'], ((1/3,)*3,)*2)
        self.assertEqual(r['discriminator'], ((.5,.5),)*3)

    def test_permutation(self):
        a = solve(self.g, self.d)
        b = solve(tuple(tuple(reversed(row)) for row in self.g), tuple(reversed(self.d)))
        for v in range(2):
            self.assertAlmostEqual(a['generator'][v][0], b['generator'][v][1])

    def test_invalid_inputs(self):
        for values in ((), (float('nan'), 0.), (-10000., 0.)):
            with self.assertRaises(ValueError): softmax(values)
        with self.assertRaises(ValueError): solve(self.g, self.d, iterations=0)
        with self.assertRaises(ValueError): anchors(self.g, self.d[:1])


if __name__ == '__main__':
    unittest.main()
