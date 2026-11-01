"""Same candidate actions/payoffs; pair-only readout to avoid global-answer drift."""
import json
from experiments.paired_deliberation.native_candidate_challenge import *
from experiments.paired_deliberation.native_candidate_challenge import challenge_request as original_request

def challenge_request(row,initial_rankings,target,reviewer,sources):
    req=original_request(row,initial_rankings,target,reviewer,sources)
    p=json.loads(req['messages'][1]['content']);pair=[target['a'],target['b']]
    p['candidate_pool']=[m for m in p['candidate_pool'] if m['movie_id'] in pair]
    p.pop('initial_rankings',None)
    p['pair_initial_positions']=[{m:(r.index(m)+1 if m in r else None) for m in pair} for r in initial_rankings]
    p['required_output']={'preferred':{'enum':pair+[None]},'source_id':'one exact offered source ID, or null iff preferred=null'}
    p['task']='ONLY decide between '+pair[0]+' and '+pair[1]+'. Never output a different movie, even if you prefer it overall.'
    req['messages'][1]['content']=json.dumps(p,ensure_ascii=False,sort_keys=True)
    req['messages'][0]['content']+=' preferred MUST be exactly '+pair[0]+' or '+pair[1]+' as a STRING, or null. Any other movie ID is outside this task. Retaining the incumbent is permitted with an applicable evidence source.'
    return req
