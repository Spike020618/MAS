"""No-API analysis of recorded agreement, visibility and non-revision.

Measures observable text and configuration; does not infer hidden cognition.
"""
from collections import Counter,defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from statistics import mean

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'experiments/results/mechanism/stackelberg/aeic_deliberation_dev_v1'


def normalize(text):return re.sub(r'\s+',' ',text).strip()


def analyze(directory=DIRECTORY):
    calls=[json.loads(line) for line in (directory/'calls.jsonl').read_text().splitlines()]
    by_key={r['key']:r for r in calls}
    results=json.loads((directory/'analysis.json').read_text())['rows']
    protocol=json.loads((directory/'protocol.json').read_text())
    disagreements=[];field_exact=Counter();field_ratios=defaultdict(list)
    initial_with_sources=0;revision_with_raw_source_field=0;peer_with_provenance_fields=0
    transitions=[];reference_positions=Counter();same_prompt_stances=defaultdict(set)
    for run in results:
        if len(set(run['initial_votes']))>1:
            disagreements.append({'question_id':run['question_id'],'condition':run['condition'],
                'initial_votes':run['initial_votes'],'final_votes':run['revised_votes'],
                'initial_correct':run['initial_correct'],'final_correct':run['final_correct']})
        for i,(first_key,second_key) in enumerate(zip(run['initial_keys'],run['revision_keys'])):
            first=by_key[first_key];second=by_key[second_key]
            p=json.loads(first['messages'][1]['content']);q=json.loads(second['messages'][1]['content'])
            initial_with_sources+=bool(p['sources'])
            revision_with_raw_source_field+=('sources' in q)
            peer_with_provenance_fields+=sum(any(k in peer for k in ['sources','source_hash','url','source_quotes']) for peer in q['peers'])
            a,b=first['parsed'],second['parsed']
            same_prompt_stances[first['prompt_sha256']].add(a['C'])
            exact={};ratios={}
            for field in ['A','E','I']:
                left,right=normalize(a[field]),normalize(b[field])
                exact[field]=left==right;ratios[field]=SequenceMatcher(None,left,right,autojunk=False).ratio()
                field_exact[field]+=exact[field];field_ratios[field].append(ratios[field])
            peers={peer['evidence_id']:peer for peer in q['peers'] if 'evidence_id' in peer}
            citations=run['references'][i]['valid']
            for ref in citations:
                peer=peers[ref['evidence_id']]
                relation='same_C' if peer['C']==a['C'] else 'different_C'
                reference_positions[relation+'/'+ref['disposition']]+=1
            transitions.append({'question_id':run['question_id'],'condition':run['condition'],'agent':i,
                'initial_C':a['C'],'final_C':b['C'],'AEI_exact':exact,'AEI_similarity':ratios,
                'valid_reference_count':len(citations),'initial_key':first_key,'revision_key':second_key})
    marker='Answer the question using your own knowledge and any provided source excerpts.'
    counts=[prompt.count(marker) for prompt in protocol['report_prompts']]
    revision=protocol['revision_prompt']
    return {'scope':'observational audit of completed 8-question development data; no new calls',
        'n_runs':len(results),'n_individual_revisions':len(transitions),
        'unanimous_runs':sum(len(set(r['initial_votes']))==1 for r in results),
        'unanimous_runs_by_condition':{c:sum(r['condition']==c and len(set(r['initial_votes']))==1 for r in results) for c in 'ABCD'},
        'unanimous_correct_runs':sum(len(set(r['initial_votes']))==1 and r['initial_correct'] for r in results),
        'unanimous_wrong_runs':sum(len(set(r['initial_votes']))==1 and not r['initial_correct'] for r in results),
        'n_disagreement_question_ids':len({r['question_id'] for r in disagreements}),
        'disagreements':disagreements,'all_C_unchanged':all(r['initial_C']==r['final_C'] for r in transitions),
        'initial_reports_with_sources':initial_with_sources,'initial_reports_without_sources':len(transitions)-initial_with_sources,
        'revision_payloads_with_direct_sources_field':revision_with_raw_source_field,
        'peer_messages_with_structured_provenance_fields':peer_with_provenance_fields,
        'AEI_exact_counts':dict(field_exact),'AEI_mean_text_similarity':{k:mean(v) for k,v in field_ratios.items()},
        'revisions_with_any_AEI_text_change':sum(not all(r['AEI_exact'].values()) for r in transitions),
        'valid_reference_dispositions_by_initial_C_relation':dict(reference_positions),
        'initial_prompt_unique_groups':len(same_prompt_stances),
        'same_initial_prompt_groups_with_multiple_C':sum(len(v)>1 for v in same_prompt_stances.values()),
        'unique_nonnull_response_ids':len({r['response_id'] for r in calls if r.get('response_id')}),
        'returned_models':sorted({str(r.get('returned_model')) for r in calls}),
        'duplicated_initial_instruction_counts_per_persona':counts,
        'literal_keep_C_instruction_detected':any(s in revision.casefold() for s in [
            'keep your initial conclusion','do not change your conclusion','always keep your original answer']),
        'revision_prompt_contains_initial_judgment':('initial judgment' in revision),
        'revision_prompt_forbids_majority_only_switch':('do not change merely to join a majority' in revision),
        'nonidentifiable_without_new_interventions':['causal effect of own-answer anchoring',
            'causal effect of duplicated prompt instructions','semantic correctness of all cited evidence',
            'whether exposing provenance or resolving assumptions would improve accuracy',
            'model-weight identity or server-side caching from response IDs alone'],
        'transitions':transitions}


if __name__=='__main__':
    result=analyze()
    (DIRECTORY/'stagnation_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='transitions'},ensure_ascii=False,indent=2))
