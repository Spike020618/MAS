from hashlib import sha256
from pathlib import Path
import sqlite3
import tempfile
import unittest
from experiments.core.mechanism.evidence_acquisition import Corpus, Ledger, query_terms, answer_score, counterfactual_advantage


class EvidenceTests(unittest.TestCase):
    def corpus(self, folder):
        p = Path(folder) / 'test.sqlite'; con = sqlite3.connect(p)
        con.execute('CREATE TABLE docs(rowid INTEGER PRIMARY KEY,docid TEXT,title TEXT,text TEXT)')
        con.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(title,text,content='docs',content_rowid='rowid')")
        con.execute('CREATE TABLE state(k TEXT,v TEXT)'); con.execute("INSERT INTO state VALUES('count','2')")
        for row in [(1, '1', 'Ada', 'Ada wrote a program.'), (2, '2', 'Bob', 'Bob climbed a mountain.')]:
            con.execute('INSERT INTO docs VALUES(?,?,?,?)', row)
            con.execute('INSERT INTO docs_fts(rowid,title,text) VALUES(?,?,?)', (row[0], row[2], row[3]))
        con.commit(); con.close(); return Corpus(p)

    def test_real_fts_and_no_query_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.corpus(folder)
            self.assertEqual(c.search('Ada', 1)[0]['docid'], '1')
            c.search('"; DROP TABLE docs; --', 1)
            self.assertEqual(c.search('Bob', 1)[0]['docid'], '2'); c.close()

    def test_shared_dedup_and_search_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.corpus(folder); ledger = Ledger()
            ledger.acquire(c, 'Ada', 1); e = ledger.acquire(c, 'Ada', 1)
            self.assertEqual(e['duplicate_ids'], ['1'])
            self.assertEqual(len(ledger.public_documents()), 1)
            with self.assertRaises(RuntimeError): ledger.acquire(c, 'Bob', 1)
            self.assertEqual(set(ledger.public_documents()[0]), {'docid', 'title', 'text', 'text_sha256'}); c.close()

    def test_no_silent_text_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.corpus(folder); ledger = Ledger(max_unique_bytes=2)
            event = ledger.acquire(c, 'Ada', 1)
            self.assertEqual(event['skipped_budget_ids'], ['1']); self.assertFalse(ledger.public_documents()); c.close()

    def test_hotpot_answer_metric(self):
        self.assertEqual(answer_score('The United States.', 'United States')['em'], 1)
        self.assertEqual(answer_score('yes', 'no')['f1'], 0)
        self.assertEqual(answer_score('New York City', 'New York')['f1'], .8)

    def test_counterfactual_gradient_expectation_preserved(self):
        table = {(0, 0): .2, (0, 1): .8, (1, 0): .9, (1, 1): .4}
        policies = [[.3, .7], [.6, .4]]
        for i in range(2):
            plain = [0., 0.]; credit = [0., 0.]
            for profile, reward in table.items():
                probability = policies[0][profile[0]] * policies[1][profile[1]]
                advantage = counterfactual_advantage(table, policies, profile, i)
                for action in range(2):
                    score = int(profile[i] == action) - policies[i][action]
                    plain[action] += probability * reward * score
                    credit[action] += probability * advantage * score
            for a, b in zip(plain, credit): self.assertAlmostEqual(a, b, places=12)

    def test_missing_table_not_fabricated(self):
        with self.assertRaises(ValueError): counterfactual_advantage({(0, 0): 1}, [[.5, .5], [.5, .5]], (0, 0), 0)


if __name__ == '__main__': unittest.main()
