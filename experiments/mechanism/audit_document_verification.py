"""Post-hoc citation-format and intermediate-control audit; no refitting."""
import json
import re
from experiments.mechanism import run_document_verification as run
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.core.mechanism.document_verification import retrieve


def normalized(text):return re.sub(r'\s+',' ',text).strip()


def main():
    cases=json.loads((run.OUT/'public_cases.json').read_text())
    outputs=json.loads((run.OUT/'sealed_outputs.json').read_text())
    labels=json.loads((run.OUT/'private_labels.json').read_text());counts={};rows=[]
    for c in cases:
        directory=run.OUT/c['id'].replace(':','_');s=outputs[c['id']]
        windows={'full':[{'id':'article','text':c['article']}],
                 'review2':[{'id':'article','text':c['article']}],
                 'fixed':retrieve(c['article'],c['question']+' '+' '.join(c['choices'].values())),
                 'local':json.loads((directory/'retrieval.json').read_text())['windows']}
        for arm,ww in windows.items():
            byid={w['id']:w['text'] for w in ww}
            cell=counts.setdefault(arm,{'citations':0,'exact':0,'whitespace_only_matches':0,'not_matched_even_after_whitespace':0})
            for q in s[arm]['citations']:
                cell['citations']+=1
                if q['literal_source_match']:cell['exact']+=1
                else:
                    citation=q['citation']
                    compatible=(isinstance(citation,dict) and isinstance(citation.get('quote'),str)
                                and isinstance(citation.get('chunk_id'),str) and citation['chunk_id'] in byid
                                and normalized(citation['quote']) and normalized(citation['quote']) in normalized(byid[citation['chunk_id']]))
                    cell['whitespace_only_matches' if compatible else 'not_matched_even_after_whitespace']+=1
        rows.append({'id':c['id'],'gold':labels[c['id']],
            'answers':{a:s[a]['answer'] for a in (*run.ARMS,'review1')},
            'query':json.loads((directory/'retrieval.json').read_text())['query']})
    intermediate={'n':len(cases),'correct':sum(s['review1']['answer']==labels[cid] for cid,s in outputs.items()),
        'tokens':sum(s[k]['tokens'] for s in outputs.values() for k in ('full','review1')),'calls':2*len(cases)}
    result={'scope':'post-hoc diagnostics, primary exact citation metric unchanged','new_api_calls':0,
            'citation_sensitivity':counts,'ordinary_one_review_intermediate':intermediate,
            'local_vs_review2_same_selected_answer':sum(r['answers']['local']==r['answers']['review2'] for r in rows),
            'rows':rows,'limits':['whitespace-only matching does not establish entailment',
                                'unmatched quotation may be altered, paraphrased or misattributed; not automatically proof of fabrication',
                                'one-review control was not an additional preregistered primary comparison']}
    freeze(run.OUT/'diagnosis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
