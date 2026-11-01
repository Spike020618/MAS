"""Post-seal trace diagnostic: provenance and retrieval coverage, not entailment."""
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism.audit_open_retrieval_coverage import title_key
from experiments.core.mechanism.evidence_acquisition import query_terms
from experiments.mechanism.run_upgrade_pilot import freeze

OUT = p.ROOT / 'experiments/results/redesign_v1/evidence_source_audit_v1'


def main():
    p.verify()
    if not (p.OUT / 'final_acceptance.json').exists(): raise ValueError('primary not accepted')
    cases = p.base.read(p.OUT / 'public_cases.json'); outputs = p.base.read(p.OUT / 'sealed_outputs.json')
    labels = p.base.read(p.OUT / 'private_labels.json')
    support_source = p.data.parquet_module().read_table(p.data.RAW / 'dev_fullwiki.parquet', columns=['id', 'supporting_facts']).to_pylist()
    supports = {r['id']: {title_key(t) for t in r['supporting_facts']['title']} for r in support_source if r['id'] in outputs}
    rows = []
    for c in cases:
        cid = c['id']; r = p.base.read(p.OUT / cid / 'retrieval.json'); envs = r['environments']; cells = {}
        for arm in p.ANSWER_ARMS:
            docs = c['native_documents'] if arm == 'native' else envs[arm]['documents'] if arm in envs else []
            if arm in envs or arm == 'native':
                ids = {d['docid'] for d in docs}; cited = outputs[cid][arm].get('docids', [])
                titles = {title_key(d['title']) for d in docs}
                cells[arm] = {'documents': len(docs), 'bytes': sum(len(d['text'].encode()) for d in docs),
                    'titles': [d['title'] for d in docs], 'support_titles_retrieved': sorted(titles & supports[cid]),
                    'support_title_recall': len(titles & supports[cid]) / len(supports[cid]),
                    'answer': outputs[cid][arm]['answer'], 'score': p.answer_score(outputs[cid][arm]['answer'], labels[cid]),
                    'cited_ids_registered': all(x in ids for x in cited),
                    'cited_ids': cited, 'citation_entailment_NOT_established': True}
        question_terms = set(query_terms(c['question'], 1000)); qs = []
        for query in r['queries'][1:]:
            qs.append({'query': query, 'tokens_not_in_question': sorted(set(query_terms(query, 1000)) - question_terms),
                'note': 'lexical novelty, NOT evidence novelty, hallucination or proof of unknown knowledge'})
        rows.append({'id': cid, 'question': c['question'], 'gold_offline_only': labels[cid],
            'support_titles_offline_only': sorted(supports[cid]), 'queries': r['queries'], 'query_provenance': qs,
            'direct_answer': outputs[cid]['direct']['answer'], 'cells': cells})
    summary = {}
    for arm in ('s0', 's1', 's2', 'p01', 'p02', 'p12', 'all', 'sequential', 'native'):
        values = [r['cells'][arm] for r in rows]
        summary[arm] = {'em_correct': sum(v['score']['em'] for v in values),
            'all_support_titles_retrieved': sum(v['support_title_recall'] == 1 for v in values),
            'no_support_titles_retrieved': sum(v['support_title_recall'] == 0 for v in values),
            'mean_support_title_recall': sum(v['support_title_recall'] for v in values) / 8,
            'sum_document_bytes': sum(v['bytes'] for v in values),
            'all_citations_reference_supplied_ids': all(v['cited_ids_registered'] for v in values)}
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'result.json', {'scope': 'post-seal eight-question diagnostic, not a new independent test',
        'n': 8, 'summary': summary, 'cases': rows, 'model_calls': 0})
    print(__import__('json').dumps(summary, indent=2))


if __name__ == '__main__': main()
