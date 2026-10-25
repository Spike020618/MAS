import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.mechanism.prepare_upgrade_pilot import canonical, prepare, strings
from experiments.mechanism.build_upgrade_history import build


class SamplingTests(unittest.TestCase):
    def test_embedded_question_and_normalization(self):
        value = {'content': json.dumps({'question': 'Ａ  test?'})}
        self.assertIn('Ａ  test?', list(strings(value)))
        self.assertEqual(canonical('Ａ  test?'), 'atest')

    def test_label_independent_selection_and_private_separation(self):
        # Synthetic source only for exercising schemas; never a research sample.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = [{'qid': 'mock-' + str(i), 'question': 'Mock engineering question number ' + str(i) + '?',
                       'answer': bool(i % 2)} for i in range(210)]
            raw = root / 'raw.json'; raw.write_text(json.dumps(source))
            history = root / 'history.json'
            history.write_text(json.dumps({'errors': [], 'texts': {}, 'files': []}))
            first = root / 'one'
            prepare(raw, history, first, n=2, source_split='dev')
            source = [{**r, 'answer': not r['answer']} for r in source]
            raw.write_text(json.dumps(source))
            second = root / 'two'
            prepare(raw, history, second, n=2, source_split='dev')
            public = json.loads((first / 'public_cases.json').read_text())
            self.assertEqual(public, json.loads((second / 'public_cases.json').read_text()))
            self.assertTrue(all(set(r) == {'id', 'source_id', 'question', 'choices'} for r in public))
            a = json.loads((first / 'private_labels.json').read_text())
            b = json.loads((second / 'private_labels.json').read_text())
            self.assertTrue(all(x['label'] != y['label'] for x, y in zip(a, b)))

    def test_incomplete_source_and_index_errors_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw.json'; raw.write_text('[]')
            history = root / 'history.json'; history.write_text('{}')
            with self.assertRaises(ValueError): prepare(raw, history, root / 'out')

    def test_verified_raw_pool_is_not_confused_with_executed_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'experiments/results/mechanism/stackelberg/locked_source_game_test_v1'
            directory.mkdir(parents=True)
            pool = [{'qid': 'raw-1', 'question': 'Raw pool only?', 'answer': True},
                    {'qid': 'used-2', 'question': 'Executed question?', 'answer': False}]
            raw = directory / 'official_train.json'; raw.write_text(json.dumps(pool))
            (directory / 'dataset_provenance.json').write_text(json.dumps({
                'url': 'https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json',
                'sha256': hashlib.sha256(raw.read_bytes()).hexdigest(), 'n_examples': 2}))
            (directory / 'cases.json').write_text(json.dumps([{'question': 'Executed question?'}]))
            result = build(root)
            texts = {v['text'] for v in result['texts'].values()}
            self.assertNotIn('raw pool only?', texts)
            self.assertIn('executed question?', texts)
            # A renamed/serialized copy must also remain a raw pool, not execution.
            (directory / 'source_snapshot.json').write_text(json.dumps(pool, indent=2))
            result = build(root)
            self.assertEqual(sum(f.get('role') == 'verified_official_raw_pool_not_model_execution' for f in result['files']), 2)


if __name__ == '__main__': unittest.main()
