"""Generic source-bank repair: canonical-page dedup and paragraph/table extraction.

Reuses the existing general extractor; never imports the task-specific checker.
"""
import hashlib
from pathlib import Path
from urllib.parse import unquote, urlsplit

from experiments.mechanism.pilot_grounded_evidence import ranked_sources


def canonical_page(url):
    parts=urlsplit(url)
    return (parts.hostname or '').lower()+unquote(parts.path).replace('_',' ').rstrip('/').casefold()


def repair_sources(question,sources,root):
    seen=set(); selected=[]
    for source in sources:
        page=canonical_page(source['url'])
        if page in seen:continue
        seen.add(page)
        snapshot=Path(source['snapshot_path'])
        if not snapshot.is_absolute():snapshot=root/snapshot
        content=snapshot.read_bytes()
        expected=source.get('source_hash',snapshot.stem)
        if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('source snapshot changed')
        selected.append({**source,'snapshot_path':str(snapshot),'source_hash':expected})
    result=ranked_sources(question,selected)
    for source in result:
        source['text']=source.pop('excerpt')
        source['canonical_page']=canonical_page(source['url'])
        source['pipeline']='generic-blocks-canonical-dedup-v1'
    return result
