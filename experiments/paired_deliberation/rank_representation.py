"""Decode an exact list serialized inside a ranking-only answer field.

No free-prose extraction, ranking repair, padding, truncation or candidate substitution.
"""
import ast
import copy
import json


def decode_rank_literal(record, request, pool, length=10):
    result = copy.deepcopy(record)
    obj = result.get('report')
    if request['kind'] != 'ranking_v6' or not isinstance(obj, dict) or set(obj) != {'answer'}:
        return result, []
    value = obj['answer']
    if not isinstance(value, str) or len(value) > 4096 or not value.strip().startswith('['):
        return result, []
    try:
        try: ranking = json.loads(value)
        except json.JSONDecodeError: ranking = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return result, []
    if (not isinstance(ranking, list) or len(ranking) != length or
            any(not isinstance(mid, str) for mid in ranking) or
            len(set(ranking)) != length or not set(ranking) <= set(pool)):
        return result, []
    result['report'] = {'ranking': ranking}
    return result, ['decode_exact_serialized_candidate_list_in_answer; order_and_ids_unchanged']
