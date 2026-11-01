"""Full row-by-row index parity with pinned corpus; read-only, bounded memory."""
from hashlib import sha256
import json
import sqlite3
import time
from experiments.mechanism import redesign_hotpot_data as data
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    started = time.perf_counter()
    provenance = p.base.read(data.RAW / 'corpus.parquet.provenance.json')
    con = sqlite3.connect('file:' + str(data.ENV / 'wiki_fts.sqlite') + '?mode=ro', uri=True)
    digest = sha256(); count = 0
    try:
        assert con.execute('PRAGMA quick_check').fetchall() == [('ok',)]
        for batch in data.parquet_module().ParquetFile(data.RAW / 'corpus.parquet').iter_batches(batch_size=10000):
            originals = batch.to_pylist()
            indexed = con.execute('SELECT docid,title,text FROM docs WHERE rowid>? AND rowid<=? ORDER BY rowid',
                (count, count + len(originals))).fetchall()
            assert len(indexed) == len(originals)
            for original, actual in zip(originals, indexed):
                assert actual == (str(original['_id']), original['title'], original['text'])
                digest.update(json.dumps(actual, ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
            count += len(originals)
            if count % 1000000 == 0: print('full index rows verified', count, flush=True)
        assert count == 5233329 == con.execute('SELECT COUNT(*) FROM docs').fetchone()[0]
        assert int(con.execute("SELECT v FROM state WHERE k='count'").fetchone()[0]) == count
    finally: con.close()
    target = p.ROOT / 'experiments/results/redesign_v1/index_parity_v1'
    target.mkdir(parents=True, exist_ok=True)
    result = {'full_row_parity': True, 'rows': count, 'sqlite_quick_check': 'ok',
        'ordered_docid_title_text_sha256': digest.hexdigest(),
        'corpus_sha256_expected': data.ASSETS[0][4], 'corpus_provenance': provenance,
        'elapsed_seconds': time.perf_counter() - started, 'model_calls': 0}
    # Timing differs across verification invocations; preserve the original.
    if (target / 'result.json').exists():
        prior = p.base.read(target / 'result.json')
        assert {k:v for k,v in prior.items() if k != 'elapsed_seconds'} == {k:v for k,v in result.items() if k != 'elapsed_seconds'}
    else: freeze(target / 'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'corpus_provenance'}, indent=2), flush=True)


if __name__ == '__main__': main()
