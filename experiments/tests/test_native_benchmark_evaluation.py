import ast
from itertools import permutations
import math
import unittest

from experiments.consensus_recommendation.common import ROOT
from experiments.paired_deliberation.benchmark_evaluation import native_item_metrics


class NativeBenchmarkEvaluationTests(unittest.TestCase):
    def test_parity_with_pinned_authors_three_metric_functions(self):
        # Extract only the three inspected arithmetic functions, without importing torch or training code.
        path = ROOT/'docs/research/benchmark_protocol_sources_2026-10-27/UniCRS/src/evaluate_rec.py'
        tree = ast.parse(path.read_text())
        names = {'compute_recall', 'compute_mrr', 'compute_ndcg'}
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RecEvaluator')
        functions = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({f.name for f in functions}, names)
        scope = {'math': math, '__builtins__': {'int': int}}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
        for ranking_tuple in permutations(range(5)):
            ranking = list(ranking_tuple)
            for target in range(6):
                ours = native_item_metrics(ranking, target, ks=(1,3,5))
                for k in (1,3,5):
                    for name in ('recall','mrr','ndcg'):
                        self.assertAlmostEqual(ours[name+'@'+str(k)], scope['compute_'+name](None, ranking, target, k))

    def test_missing_target_is_zero_not_removed_and_duplicate_ranking_is_invalid(self):
        result = native_item_metrics(['a','b'], 'c', ks=(1,5))
        self.assertTrue(all(v == 0 for v in result.values()))
        with self.assertRaises(ValueError): native_item_metrics(['a','a'], 'a')
        with self.assertRaises(ValueError): native_item_metrics(['a'], None)


if __name__ == '__main__': unittest.main()
