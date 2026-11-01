from hashlib import sha256
import unittest
from experiments.mechanism.probe_acquired_evidence_pool import combine


class PoolTests(unittest.TestCase):
    def doc(self, key, text):
        return {'docid': key, 'title': key, 'text': text, 'text_sha256': sha256(text.encode()).hexdigest()}

    def test_deduplicates_raw_text_not_titles_and_keeps_origins(self):
        docs, aliases = combine([self.doc('z', 'same'), self.doc('a', 'same'), self.doc('b', 'different')])
        self.assertEqual([d['docid'] for d in docs], ['a', 'b'])
        self.assertEqual(len(aliases[sha256(b'same').hexdigest()]), 2)

    def test_size_and_integrity_fail_closed(self):
        with self.assertRaises(ValueError): combine([self.doc('x', 'a' * 40001)])
        d = self.doc('x', 'valid'); d['text'] = 'forged'
        with self.assertRaises(ValueError): combine([d])


if __name__ == '__main__': unittest.main()
