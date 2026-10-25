"""Six reserved cases, fixed piKL; API token adaptation, NOT text replication."""
import json
from math import log
from pathlib import Path
from hashlib import sha256
from experiments.mechanism.run_upgrade_pilot import freeze, lock, CallJournal, read_rows
from experiments.mechanism.run_signaling_pilot import request, parsed
from experiments.mechanism.probe_signaling_interface import transport
from experiments.mechanism.run_equilibrium_reference import readout, evaluate

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'experiments/results/resource_allocation'
OUT = BASE/'deepseek_pikl_reserved_v1'


def h(path): return sha256(path.read_bytes()).hexdigest()


def run():
    from experiments.core.llm import LLM_BACKENDS
    cfg = LLM_BACKENDS['deepseek']
    if not cfg['api_key']: raise RuntimeError('DeepSeek API configuration missing')
    with lock(OUT):
        if not (OUT/'public_cases.json').exists():
            allcases = json.loads((BASE/'reliable_readout_data_v1/public_cases.json').read_text())
            used = set()
            for name in ('reliable_readout_v1','signaling_pilot_v1'):
                used.update(c['id'] for c in json.loads((BASE/name/'public_cases.json').read_text()))
            cases = sorted((c for c in allcases if c['id'] not in used), key=lambda c:c['id'])
            assert len(cases)==6
            checked = {}
            for path in sorted((ROOT/'experiments/results').rglob('responses.jsonl')):
                if path.parent==OUT: continue
                text = path.read_text()
                for c in cases:
                    assert c['id'] not in text and c['question'] not in text
                    assert json.dumps(c['question'],ensure_ascii=False)[1:-1] not in text
                checked[str(path.relative_to(ROOT))] = h(path)
            freeze(OUT/'selection.json', {'journal_sha256':checked,
                'scope':'exact source-ID/rendered-question exclusion only; not semantic or pretraining isolation'})
            freeze(OUT/'public_cases.json',cases)
        cases = json.loads((OUT/'public_cases.json').read_text())
        planned = sum(2+len(c['choices']) for c in cases)
        files = [Path(__file__), ROOT/'experiments/core/mechanism/equilibrium_ranking_reference.py',
                 ROOT/'experiments/core/mechanism/sequence_likelihood.py',
                 ROOT/'experiments/mechanism/run_equilibrium_reference.py',
                 ROOT/'experiments/mechanism/run_signaling_pilot.py',
                 ROOT/'experiments/core/mechanism/scored_signaling.py']
        freeze(OUT/'protocol.json', {'n':6, 'planned_calls':planned, 'model_requested':cfg['model'],
            'eta':.1,'lambda':.1,'updates':5000,'temperature':1.,'thinking':'disabled',
            'primary':'ER-D vs D; all six questions retained; no parameter tuning',
            'controls':['G','MI','SC','D','ER-G','ER-D'],
            'missing_score':'entire case abstains in all six methods, disclose missing input',
            'scope':'small developmental API code-space adaptation, not full sequence reproduction',
            'retry':'none; unknown transport stops resume',
            'code_sha256':{str(p.relative_to(ROOT)):h(p) for p in files},
            'public_sha256':h(OUT/'public_cases.json')})
        journal = CallJournal(OUT,transport,planned)
        sealed = {}; identities = set()
        for c in cases:
            outputs = {}
            slots = ['g_positive','g_negative']+['d/'+x for x in c['choices']]
            slots.sort(key=lambda s:sha256((c['id']+'/'+s).encode()).hexdigest())
            for slot in slots:
                row = journal.call(c['id']+'/'+slot,request(c,slot,cfg['model']))
                if row['response']['http_status']!=200: raise RuntimeError('HTTP error preserved; stop')
                p = parsed(row,c,slot); outputs[slot]=p
                identities.add((p.get('model'),p.get('fingerprint')))
                if len(identities)!=1 or None in next(iter(identities)):
                    raise RuntimeError('incomplete or changed backend identity; stop')
            sealed[c['id']] = outputs
            print('sealed',len(sealed),'/6',flush=True)
        freeze(OUT/'sealed_outputs.json',sealed)
        readouts = {}
        for c in cases:
            ids = c['choices']; codes='ABCDEFGHIJKLMNOPQRST'[:len(ids)]; slots=sealed[c['id']]
            if not all(r['usable'] for r in slots.values()):
                readouts[c['id']]={'answers':{m:None for m in ('G','MI','SC','D','ER-G','ER-D')},
                    'candidate_ids':ids,'reason':'incomplete code scores'}
                continue
            # Encode the observed raw masses as one-position score records.
            # Explicitly token-code space, not a full-sequence likelihood claim.
            from experiments.core.mechanism.sequence_likelihood import requests_for_case
            public={'id':c['id'],'question':c['question'],'candidates':dict(zip(ids,ids)), 'background_correction':False}
            scores={}
            for slot,req in requests_for_case(public).items():
                if slot.startswith('g_'):
                    kind,cid=slot.split('/',1);mass=slots[kind]['mass'][codes[ids.index(cid)]]
                else:
                    _,cid,verdict=slot.split('/');mass=slots['d/'+cid]['mass']['A' if verdict=='correct' else 'B']
                scores[slot]={**req,'token_logprobs':[log(mass)]}
            readouts[c['id']]=readout(public,scores)
        freeze(OUT/'sealed_readouts.json',readouts)
        labels_all={c['id']:c['label'] for c in json.loads((BASE/'reliable_readout_data_v1/private_labels.json').read_text())}
        labels={c['id']:labels_all[c['id']] for c in cases}
        rows=read_rows(OUT/'responses.jsonl');usage=[json.loads(r['response']['body']).get('usage',{}).get('total_tokens') for r in rows]
        result={**evaluate(readouts,labels),'scope':'six-case fixed DeepSeek piKL API adaptation; not confirmatory or full-text reproduction',
            'http_attempts':len(read_rows(OUT/'attempts.jsonl')),'tokens_lower_bound':sum(x for x in usage if type(x)is int),
            'unknown_usage':sum(type(x)is not int for x in usage),'sum_http_seconds':sum(r['wall_seconds'] for r in rows),
            'models':sorted(identities),'complete_cases':sum('solver' in r for r in readouts.values()),
            'sealed_readouts_sha256':h(OUT/'sealed_readouts.json'),'automatic_expansion':False}
        freeze(OUT/'result.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': run()
