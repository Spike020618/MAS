"""Simple deterministic title-anchor engineering baseline, not a learned method."""
from hashlib import sha256
import re
from experiments.core.mechanism.evidence_acquisition import Corpus

QUESTION_LEADS = frozenset('Who What Which Where When Why How Is Are Was Were Did Does Do Can Could The A An In At On From Both'.casefold().split())


def title_phrases(question):
    # Lexical capitalization heuristic only; no benchmark answers, dictionaries,
    # topic classifiers or learned parameters. Require >=2 consecutive words.
    spans = re.findall(r"\b[A-Z][A-Za-z0-9'-]*(?:[ \t]+[A-Z][A-Za-z0-9'-]*){1,5}\b", question)
    phrases = []
    for span in spans:
        words = span.split()
        while words and words[0].casefold() in QUESTION_LEADS: words.pop(0)
        if len(words) >= 2:
            value = ' '.join(words)
            if value not in phrases: phrases.append(value)
    return phrases[:8]


class TitleAnchorCorpus(Corpus):
    def search(self, query, k=2):
        if type(k) is not int or not 1 <= k <= 10: raise ValueError('bounded top-k')
        phrases = title_phrases(query)
        if not phrases: return super().search(query, k)
        expr = ' OR '.join('title : "' + p.replace('"', '""') + '"' for p in phrases)
        rows = self.con.execute('SELECT d.docid,d.title,d.text,bm25(docs_fts,5.0,1.0) '
            'FROM docs_fts JOIN docs d ON d.rowid=docs_fts.rowid '
            'WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts,5.0,1.0),d.rowid LIMIT ?', (expr, k)).fetchall()
        out = [{'docid': str(i), 'title': t, 'text': text, 'rank_score': float(score),
            'text_sha256': sha256(text.encode()).hexdigest()} for i, t, text, score in rows]
        if len(out) < k:
            seen = {d['docid'] for d in out}
            out.extend(d for d in super().search(query, k) if d['docid'] not in seen)
        return out[:k]
