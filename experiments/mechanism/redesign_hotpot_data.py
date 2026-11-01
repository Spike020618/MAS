"""Pinned mirror downloads and streaming SQLite FTS corpus index.

No model calls, qrels, answer keys or support annotations enter the index.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import requests
from experiments.mechanism.run_upgrade_pilot import freeze, lock

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'experiments/data/raw/redesign_hotpot_v1'
ENV = ROOT / 'experiments/results/redesign_v1/retrieval_env'
ASSETS = (
    ('BeIR/hotpotqa', 'corpus/corpus-00000-of-00001.parquet', 'corpus.parquet',
     975977704, '23975eb1ce371778a38b49a4cc0ca23998b4e1979ee31d02b3032a3e0cd5d62e'),
    ('hotpotqa/hotpot_qa', 'fullwiki/validation-00000-of-00001.parquet', 'dev_fullwiki.parquet',
     28041820, '78933c0a31a5f7b420d4effdf4cd4eed573b28c6a3da6179dcf7a02b39e51d03'),
)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def storage_ok():
    if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
        raise RuntimeError('stop: fewer than10GiB free; preserve partial artifacts')


def download_one(asset):
    repo, remote, filename, size, sha = asset
    RAW.mkdir(parents=True, exist_ok=True)
    with lock(RAW / (filename + '_lock')):
        target = RAW / filename
        if target.exists():
            if target.stat().st_size != size or digest(target) != sha: raise ValueError('existing artifact hash mismatch')
            return
        metadata_path = RAW / (filename + '.provenance.json')
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        else:
            response = requests.get('https://huggingface.co/api/datasets/' + repo, timeout=(10, 30))
            response.raise_for_status(); revision = response.json()['sha']
            tree = requests.get(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true', timeout=(10, 30))
            tree.raise_for_status()
            row = next(v for v in tree.json() if v.get('path') == remote)
            if row['size'] != size or row['lfs']['oid'] != sha: raise ValueError('upstream manifest changed')
            metadata = {'repo': repo, 'revision': revision, 'path': remote, 'bytes': size, 'sha256': sha,
                'url': f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}',
                'role': 'BEIR corpus' if filename == 'corpus.parquet' else 'HotpotQA fullwiki dev mirror',
                'official_task': 'https://hotpotqa.github.io/', 'license': 'CC-BY-SA-4.0',
                'limits': 'pinned mirror bytes, not a claim of equality to original tar/JSON serialization'}
            freeze(metadata_path, metadata)
        part = RAW / (filename + '.part')
        for attempt in range(3):
            storage_ok(); offset = part.stat().st_size if part.exists() else 0
            if offset == size: break
            if offset > size: raise ValueError('partial artifact exceeds source size')
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            url = metadata['url'] + '?download=true&request_nonce=' + str(time.time_ns())
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(10, 40)) as r:
                    r.raise_for_status()
                    if offset and (r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise ValueError('resume not honored; do not overwrite partial file')
                    with part.open('ab' if offset else 'xb') as stream:
                        last = offset
                        for chunk in r.iter_content(4 * 1024 * 1024):
                            storage_ok()
                            if stream.tell() + len(chunk) > size: raise ValueError('download exceeds pinned size')
                            stream.write(chunk)
                            if stream.tell() - last >= 64 * 1024 * 1024:
                                print(filename, stream.tell(), '/', size, flush=True); last = stream.tell()
                if part.stat().st_size == size: break
            except requests.RequestException as e:
                print(filename, 'download transport failure', type(e).__name__, 'attempt', attempt + 1, flush=True)
        if not part.exists() or part.stat().st_size != size: raise RuntimeError('download incomplete, retained for resume')
        if digest(part) != sha: raise ValueError('download SHA256 mismatch; retain bytes for inspection')
        part.rename(target)
        print('validated', filename, size, flush=True)


def download():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(download_one, ASSETS))


def parquet_module():
    sys.path.insert(0, '/private/tmp/mas-production-deps')
    import pyarrow.parquet as pq
    return pq


def index():
    storage_ok(); ENV.mkdir(parents=True, exist_ok=True)
    path = RAW / 'corpus.parquet'
    if path.stat().st_size != ASSETS[0][3] or digest(path) != ASSETS[0][4]: raise ValueError('corpus not verified')
    pq = parquet_module(); pf = pq.ParquetFile(path)
    if pf.metadata.num_rows != 5233329: raise ValueError('unexpected corpus size')
    dbpath = ENV / 'wiki_fts.sqlite'
    with lock(ENV):
        con = sqlite3.connect(dbpath)
        con.execute('PRAGMA journal_mode=WAL'); con.execute('PRAGMA synchronous=NORMAL')
        con.execute('PRAGMA cache_size=-65536'); con.execute('PRAGMA temp_store=FILE')
        con.execute('CREATE TABLE IF NOT EXISTS docs(rowid INTEGER PRIMARY KEY, docid TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL)')
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(title,text,content='docs',content_rowid='rowid',tokenize='unicode61')")
        con.execute('CREATE TABLE IF NOT EXISTS state(k TEXT PRIMARY KEY,v TEXT NOT NULL)')
        con.commit()
        row = con.execute("SELECT v FROM state WHERE k='count'").fetchone(); done = int(row[0]) if row else 0
        offset = 0; started = time.perf_counter()
        for batch in pf.iter_batches(batch_size=5000, columns=['_id', 'title', 'text']):
            n = batch.num_rows
            if offset + n <= done: offset += n; continue
            if offset != done: raise ValueError('checkpoint not at row batch boundary')
            storage_ok()
            if dbpath.stat().st_size > 8 * 1024 ** 3: raise RuntimeError('index storage ceiling reached')
            rows = [(offset + i + 1, str(v['_id']), v['title'], v['text']) for i, v in enumerate(batch.to_pylist())]
            with con:
                con.executemany('INSERT INTO docs VALUES (?,?,?,?)', rows)
                con.executemany('INSERT INTO docs_fts(rowid,title,text) VALUES (?,?,?)', [(a, c, d) for a, b, c, d in rows])
                con.execute("INSERT OR REPLACE INTO state VALUES('count',?)", (str(offset + n),))
            offset += n; done = offset
            if offset % 100000 == 0: print('indexed', offset, '/5233329', 'elapsed', round(time.perf_counter() - started, 1), flush=True)
        if done != 5233329: raise ValueError('index incomplete')
        con.execute('PRAGMA wal_checkpoint(TRUNCATE)'); con.close()
        freeze(ENV / 'index_manifest.json', {'corpus_sha256': ASSETS[0][4], 'documents': done,
            'index_bytes': dbpath.stat().st_size, 'type': 'SQLite FTS5 unicode61 title+body',
            'schema': 'docid,title,text only; no qrels/answer/support annotations',
            'provenance_sha256': digest(RAW / 'corpus.parquet.provenance.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('download', 'index'))
    {'download': download, 'index': index}[p.parse_args().stage]()
