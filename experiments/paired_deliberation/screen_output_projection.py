"""Posthoc representation projection; never fills claims, citations or decisions."""
import copy
import json


def project(record,request,evidence=None):
    result=copy.deepcopy(record);report=result['report'];operations=[]
    template=json.loads(request['messages'][1]['content']).get('schema')
    if not isinstance(template,dict):return result,operations
    expected=set(template)
    if isinstance(report,dict) and len(report)==1:
        key=next(iter(report));inner=report[key]
        if key in {'answer','schema'} and isinstance(inner,dict) and set(inner)==expected:
            report=copy.deepcopy(inner);operations.append('unwrap_exact_'+key+'_object')
        elif key=='answer' and request['kind']=='ranking_v6' and expected=={'ranking'} and isinstance(inner,list):
            report={'ranking':copy.deepcopy(inner)};operations.append('unwrap_answer_ranking_list')
    if isinstance(report,dict) and set(report)==expected|{'schema'} and report.get('schema')=='json_object':
        del report['schema'];operations.append('remove_exact_format_marker')
    if request['kind']=='ranking_v6' and evidence is not None and isinstance(report,dict) and set(report)=={'ranking'}:
        # Exact integer identifiers only; never coerce floats, booleans or
        # unknown IDs, reorder selections, replace candidates, or pad a list.
        pool={r['movie_id'] for r in evidence['candidates_in_shared_retrieval_order']}
        if isinstance(report['ranking'],list):
            ranking=[];changed=False
            for mid in report['ranking']:
                if type(mid) is int and str(mid) in pool:
                    ranking.append(str(mid));changed=True
                else:ranking.append(mid)
            report['ranking']=ranking
            if changed:operations.append('canonicalize_exact_integer_candidate_ids')
    result['report']=report
    return result,operations
