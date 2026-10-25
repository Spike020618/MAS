"""Audit label isolation, citations, paired quality, and all retrieval overhead."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.mechanism.analyze_real_qa_payoffs import paired_summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    d=args.directory
    rows=json.loads((d/'outcomes.json').read_text())
    retrievals=[json.loads(x) for x in (d/'retrieval.jsonl').read_text().splitlines()]
    calls=[json.loads(x) for x in (d/'calls.jsonl').read_text().splitlines()]
    errors=[]
    if len(rows)!=72 or len(calls)!=108 or len(retrievals)!=12:errors.append('unexpected pilot size')
    if len({c['key'] for c in calls})!=len(calls):errors.append('duplicate call')
    for c in calls:
        serialized=json.dumps(c['messages'],ensure_ascii=False,sort_keys=True)
        if hashlib.sha256(serialized.encode()).hexdigest()!=c['prompt_sha256']:errors.append('prompt hash mismatch')
        # Check for internal label-bearing fields; the problem's own proposed
        # answer and natural yes/no wording are legitimate benchmark input.
        if any(token in serialized for token in ['"ground_truth"','"correct_label"']):
            errors.append('internal label field in prompt')
    for retrieval in retrievals:
        for s in retrieval['sources']:
            if hashlib.sha256(s['excerpt'].encode()).hexdigest()!=s['excerpt_sha256']:
                errors.append('source excerpt hash mismatch')
            if s.get('snapshot_path') and not Path(s['snapshot_path']).exists():errors.append('missing page snapshot')
    contrasts={}
    for left,right in [('direct','retrieval_check'),('self_check','retrieval_check')]:
        pairs=[]
        for row in rows:
            if row['condition'] in [left,right]:
                pairs.append({**row,'action':'direct' if row['condition']==left else 'self_check'})
        contrasts[f'{right}_vs_{left}']=paired_summary(pairs)
    overhead={}
    for kind in ['query','support_judge']:
        subset=[c for c in calls if c['key'].endswith('/'+kind)]
        overhead[kind]={'calls':len(subset),'tokens':sum(c['trace']['prompt_tokens']+c['trace']['completion_tokens'] for c in subset),
                        'latency_sum_s':sum(c['trace']['latency_s'] for c in subset)}
    newly_called=[c for c in calls if not c.get('reused_from')]
    result={'integrity_pass':not errors,'errors':errors,'contrasts':contrasts,
        'new_llm_calls':len(newly_called),'reused_calls':len(calls)-len(newly_called),
        'new_actual_llm_tokens':sum(c['trace']['prompt_tokens']+c['trace']['completion_tokens'] for c in newly_called),
        'overhead':overhead,'http_attempts_recorded':sum(len(r['http']) for r in retrievals),
        'http_total_latency_s':sum(r['latency_s'] for r in retrievals),
        'missing_source_questions_retained':sum(not r['sources'] for r in retrievals),
        'temporal_scope':'live 2026 sources may conflict with historical benchmark labels; no source is treated as infallible',
        'claim_scope':'exploratory fixed-question retrieval comparison, not out-of-sample equilibrium validation',
        'hashes':{name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in ['protocol.json','retrieval.jsonl','calls.jsonl','outcomes.json']}}
    (d/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'},indent=2))


if __name__=='__main__':main()
