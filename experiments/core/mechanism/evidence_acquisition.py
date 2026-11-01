"""Label-free corpus access, budget ledger, and exact finite credit helpers."""
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
import re
import sqlite3
import string
import time


STOP = frozenset('a an the is are was were of in on at to from and or for with by which what who when where how did does do has have had it its that this as than both same'.split())


def query_terms(text, limit=12):
    if not isinstance(text, str) or not text.strip(): raise ValueError('nonempty query required')
    # No task/domain-specific rules, answer keys, gold entity markers or SQL.
    terms = [t.casefold() for t in re.findall(r"[A-Za-z0-9]+", text) if len(t) > 1 and t.casefold() not in STOP]
    return list(dict.fromkeys(terms))[:limit]


class Corpus:
    def __init__(self, path):
        self.path = path
        self.con = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
        row = self.con.execute("SELECT v FROM state WHERE k='count'").fetchone()
        if not row or int(row[0]) < 1: raise ValueError('incomplete corpus')

    def search(self, query, k=2):
        if type(k) is not int or not 1 <= k <= 10: raise ValueError('bounded top-k')
        terms = query_terms(query)
        if not terms: return []
        expression = ' OR '.join('"' + t + '"' for t in terms)
        rows = self.con.execute('SELECT d.docid,d.title,d.text,bm25(docs_fts,5.0,1.0) '
            'FROM docs_fts JOIN docs d ON d.rowid=docs_fts.rowid '
            'WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts,5.0,1.0),d.rowid LIMIT ?', (expression, k)).fetchall()
        return [{'docid': str(i), 'title': title, 'text': text, 'rank_score': float(score),
                 'text_sha256': sha256(text.encode()).hexdigest()} for i, title, text, score in rows]

    def close(self): self.con.close()


@dataclass
class Ledger:
    max_searches: int = 2
    max_unique_bytes: int = 16000
    searches: list = field(default_factory=list)
    documents: dict = field(default_factory=dict)

    def acquire(self, corpus, query, k=2):
        if len(self.searches) >= self.max_searches: raise RuntimeError('search budget exhausted')
        started = time.perf_counter(); docs = corpus.search(query, k)
        accepted = []; duplicates = []; skipped = []
        used = sum(len(d['text'].encode()) for d in self.documents.values())
        for d in docs:
            if d['text_sha256'] != sha256(d['text'].encode()).hexdigest(): raise ValueError('source hash mismatch')
            key = d['docid']
            if key in self.documents:
                if self.documents[key]['text_sha256'] != d['text_sha256']: raise ValueError('unstable source')
                duplicates.append(key); continue
            size = len(d['text'].encode())
            if used + size > self.max_unique_bytes:
                skipped.append(key); continue
            self.documents[key] = dict(d); used += size; accepted.append(key)
        event = {'query': query, 'k': k, 'returned_ids': [d['docid'] for d in docs],
                 'new_ids': accepted, 'duplicate_ids': duplicates, 'skipped_budget_ids': skipped,
                 'wall_seconds': time.perf_counter() - started}
        self.searches.append(event)
        return event

    def public_documents(self):
        return [{k: d[k] for k in ('docid', 'title', 'text', 'text_sha256')} for d in self.documents.values()]


def normalized_answer(text):
    value = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', value).split())


def answer_score(prediction, gold):
    """HotpotQA answer EM/F1 rules; gold is offline only, never corpus input."""
    p, g = normalized_answer(prediction), normalized_answer(gold)
    em = float(p == g)
    if (p in ('yes', 'no', 'noanswer') or g in ('yes', 'no', 'noanswer')) and p != g:
        return {'em': em, 'f1': 0.}
    common = sum((Counter(p.split()) & Counter(g.split())).values())
    if not common: return {'em': em, 'f1': 0.}
    precision, recall = common / len(p.split()), common / len(g.split())
    return {'em': em, 'f1': 2 * precision * recall / (precision + recall)}


def counterfactual_advantage(table, probabilities, profile, player):
    """Exact on a COMPLETE observed finite table; not a learned true critic."""
    profiles = set(product(*(range(len(p)) for p in probabilities)))
    if set(table) != profiles: raise ValueError('missing payoff cells, cannot impute')
    if tuple(profile) not in profiles or player not in range(len(probabilities)): raise ValueError('unregistered profile/player')
    baseline = 0.
    for action, probability in enumerate(probabilities[player]):
        alternate = list(profile); alternate[player] = action
        baseline += probability * table[tuple(alternate)]
    return table[tuple(profile)] - baseline
