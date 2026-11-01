"""Label-blind Qwen reranker on preselected native validation contexts."""
import json,time
from dotenv import load_dotenv
from experiments.paired_deliberation.native_redial_benchmark import ROOT,OUT,save,evaluate
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.transport_d2 import TransportD2,parse_d2_raw
from experiments.paired_deliberation.preflight_d2 import ACTORS
from experiments.paired_deliberation.requests_v5 import RequestsV5

def run():
    load_dotenv(ROOT/'.env.local',override=False)
    inputs=json.loads((OUT/'single_llm.inputs.json').read_text()); root=OUT/'single_llm'
    manifest={'version':'native-redial-single-v1','model':ACTORS['qwen_flash'],'selected_by':'input-ID hash',
        'ranking':'LLM orders ten of train-only retrieved100; deterministic retrieval tail; misses score zero',
        'labels_available':False,'physical_cap':40,'max_estimated_cny':1.0}
    save(root/'plan.json',manifest)
    slots=root/'slots'; provider=DurableProvider(slots,manifest,TransportD2('qwen_flash',ACTORS['qwen_flash'],slots,40,1.0),parse_d2_raw,True)
    results=[]; cost=0
    for i,row in enumerate(inputs):
        slot='native_'+str(i); cached=(slots/(slot+'.raw.json')).exists()
        req=RequestsV5.envelope('native_ranking',row,
            'Recommend movies for the CURRENT Seeker from the supplied 100 candidates. Return only JSON {"ranking": [TEN distinct movie_id STRINGS]}. '
            'Use the visible conversation prefix; predict the recommender next movie choices. Candidate order is a training retrieval prior. '
            'Do not invent dialogue or treat mentions as liking. No explanations or wrappers.',max_tokens=512)
        pool=[m['movie_id'] for m in row['candidates']]
        try:
            record=provider.complete(slot,req)
            usage=record['usage']; cost+=(usage['prompt_tokens']*.15+usage['completion_tokens']*1.5)/1e6
            if record['model'] not in ACTORS['qwen_flash']['allowed_returned_models']:raise ValueError('unexpected model')
            ranking=record['report'].get('ranking')
            if not isinstance(ranking,list) or len(ranking)!=10 or len(set(ranking))!=10 or any(m not in pool for m in ranking):raise ValueError('invalid ranking')
            ranking=ranking+[m for m in pool if m not in ranking]
            results.append({'id':row['id'],'ranking':list(map(int,ranking)),'valid':True,'usage':usage,'model':record['model']})
        except (ValueError,RuntimeError,KeyError,TypeError) as exc:
            results.append({'id':row['id'],'ranking':[],'valid':False,'error':str(exc)})
        save(root/'predictions.json',results)
        print(i+1,'valid',results[-1]['valid'],flush=True)
        if not cached:time.sleep(3)
    labels=json.loads((OUT/'single_llm.labels.json').read_text())
    assert [r['id'] for r in results]==[r['id'] for r in labels]
    summary=evaluate(labels,[r['ranking'] for r in results]); summary.update(valid_contexts=sum(r['valid'] for r in results),known_response_cost_cny=cost,
        failures_count_as_miss=True,cutoff50_note='top10 model-selected, remaining positions retrieval tail')
    save(root/'summary.json',summary); print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
