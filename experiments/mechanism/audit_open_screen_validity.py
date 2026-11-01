"""Explicit limits of the first finite menu: order, scarcity, no train/test claims."""
from collections import defaultdict
import json
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism import audit_open_retrieval_coverage as coverage
from experiments.mechanism.run_upgrade_pilot import freeze, read_rows

OUT=p.ROOT/'experiments/results/redesign_v1/screen_validity_v1'


def main():
    p.verify(); cases=p.base.read(p.OUT/'public_cases.json'); groups=defaultdict(list); byte_rows=[]
    for c in cases:
        r=p.base.read(p.OUT/c['id']/'retrieval.json')
        for name,e in r['environments'].items():
            byte_rows.append({'id':c['id'],'arm':name,'bytes':sum(len(d['text'].encode()) for d in e['documents']),
                'skipped_documents':sum(len(s['skipped_budget_ids']) for s in e['searches']),
                'searches':len(e['searches'])})
        for row in read_rows(p.OUT/c['id']/'responses.jsonl'):
            v=json.loads(row['request']['messages'][1]['content'])
            if 'documents' not in v or row['key'] in ('seq_plan','thinking'): continue
            key=(c['id'],tuple(sorted((d['docid'],d['text_sha256']) for d in v['documents'])))
            groups[key].append({'stage':row['key'],'ordered_ids':[d['docid'] for d in v['documents']]})
    order_groups=[{'id':key[0],'requests':v} for key,v in groups.items() if len({tuple(x['ordered_ids']) for x in v})>1]
    stats=coverage.p.base.read(coverage.OUT/'result.json')['metrics']
    gate=all(stats[s]['title_anchor']['2']['mean_support_title_recall']-stats[s]['bm25']['2']['mean_support_title_recall']>=.05
        and stats[s]['title_anchor']['10']['all_support_titles_covered']>=stats[s]['bm25']['10']['all_support_titles_covered']
        for s in ('diagnostic','confirmation'))
    OUT.mkdir(parents=True,exist_ok=True)
    result={'n':8,'max_actual_evidence_bytes':max(b['bytes'] for b in byte_rows),
        'documents_skipped_by_read_budget':sum(b['skipped_documents'] for b in byte_rows),
        'read_budget_bound_observed':False,'search_count_limit_still_imposed':True,
        'query_count_budget_not_empirically_justified_as_optimal':True,
        'same_doc_set_different_order_groups':order_groups,
        'identical_request_cache_is_NOT_order_invariance':True,
        'conditional_title_anchor_answer_run_gate_passed':gate,
        'conditional_title_anchor_answer_run_calls':0,
        'gate_design_sha256':p.base.h(p.ROOT/'docs/research/redesign_v1/检索环境修复_条件性复核协议_2026-10-25.md'),
        'actual_evidence_resources':byte_rows}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('actual_evidence_resources','same_doc_set_different_order_groups')},indent=2))


if __name__=='__main__':main()
