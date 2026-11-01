from pathlib import Path
import sqlite3
import tempfile
import unittest
from experiments.core.mechanism.title_anchor_retrieval import TitleAnchorCorpus, title_phrases


class AnchorTests(unittest.TestCase):
    def test_phrases_generic_and_bounded(self):
        self.assertEqual(title_phrases('Who wrote Green Valley after moving to North Island?'), ['Green Valley', 'North Island'])
        self.assertEqual(title_phrases('how many stars?'), [])
        self.assertEqual(title_phrases('What Is Azure Mountain?'), ['Azure Mountain'])

    def test_title_first_fallback_and_readonly(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'corpus.sqlite'; con = sqlite3.connect(path)
            con.executescript("CREATE TABLE state(k TEXT,v TEXT); INSERT INTO state VALUES('count','3');"
                'CREATE TABLE docs(rowid INTEGER PRIMARY KEY,docid TEXT,title TEXT,text TEXT);'
                "CREATE VIRTUAL TABLE docs_fts USING fts5(title,text,content='docs',content_rowid='rowid');")
            con.executemany('INSERT INTO docs VALUES(?,?,?,?)', [(1,'11','Green Valley','a secluded village'),
                (2,'12','Year','Green and Valley unrelated'), (3,'13','Astronomy','stars and moons')])
            con.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')"); con.commit(); con.close()
            engine = TitleAnchorCorpus(path)
            try:
                self.assertEqual(engine.search('Where is Green Valley?', 2)[0]['docid'], '11')
                self.assertEqual(engine.search('stars', 2)[0]['docid'], '13')
                self.assertLessEqual(len(engine.search('Green Valley OR " DROP TABLE docs;--', 2)), 2)
                with self.assertRaises(sqlite3.OperationalError): engine.con.execute('DELETE FROM docs')
                with self.assertRaises(ValueError): engine.search('x', 0)
            finally: engine.close()


if __name__ == '__main__': unittest.main()
