"""Native single-target ranking metrics, separate from custom multi-target cases.

Compatible with the inspected UniCRS evaluator's per-recommendation-item unit.
No user simulator, model judge, hidden labels or model dependencies.
"""
import math


def native_item_metrics(ranking, target, ks=(1, 10, 50)):
    if not isinstance(ranking, list) or len(set(ranking)) != len(ranking):
        raise ValueError('unique ranked item list required')
    if target is None: raise ValueError('one explicit benchmark target required')
    result = {}
    position = ranking.index(target) if target in ranking else None
    for k in ks:
        if type(k) is not int or k <= 0: raise ValueError('positive integer cutoff required')
        found = position is not None and position < k
        result['recall@'+str(k)] = int(found)
        result['mrr@'+str(k)] = 1/(position+1) if found else 0.0
        result['ndcg@'+str(k)] = 1/math.log2(position+2) if found else 0.0
    return result
