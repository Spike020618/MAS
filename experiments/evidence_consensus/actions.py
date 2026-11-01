"""Real local catalog expansion and task-wide resource reservation/settlement."""
from __future__ import annotations
import copy, math, threading
from .schema import scope_allows
from .state import check_integrity


def fetch_source(state, source_id, purpose='prediction_basis'):
    """Actually copy a catalog source into shared visible evidence, in place."""
    check_integrity(state)
    if source_id not in state['catalog']: raise ValueError('unknown source ID')
    source = state['catalog'][source_id]
    if not scope_allows(source, state['task_id'], purpose): raise ValueError('source outside requested authority')
    before = list(state['visible_sources'])
    reused = source_id in state['visible_sources']
    if not reused: state['visible_sources'][source_id] = copy.deepcopy(source)
    event = {'kind': 'reuse_source' if reused else 'fetch_source', 'source_id': source_id,
             'before_visible_ids': before, 'after_visible_ids': list(state['visible_sources']),
             'new_sources': 0 if reused else 1, 'semantic_truth_certified': False}
    state['events'].append(event)
    return event


def verify_source(state, source_id, purpose='prediction_basis', model_opinion=None):
    check_integrity(state)
    source = state['visible_sources'].get(source_id)
    result = {'kind': 'verify_source', 'source_id': source_id, 'actually_visible': source is not None,
              'scope_authorized': source is not None and scope_allows(source, state['task_id'], purpose),
              'semantic_truth_certified': False}
    if model_opinion is not None: result['uncertified_model_opinion'] = copy.deepcopy(model_opinion)
    state['events'].append(copy.deepcopy(result))
    return result


class BudgetExceeded(RuntimeError): pass


class BudgetLedger:
    """Thread-safe whole-task bounds; reservations prevent concurrent overspend.

    Unknown-response usage retains its reserved upper rather than pretending zero.
    A known result charges actual returned usage, including a failed HTTP outcome.
    """
    def __init__(self, max_calls, max_tokens, max_cny):
        if type(max_calls) is not int or max_calls < 0 or type(max_tokens) is not int or max_tokens < 0 or not math.isfinite(max_cny) or max_cny < 0:
            raise ValueError('finite nonnegative task budget required')
        self.limits = {'calls': max_calls, 'tokens': max_tokens, 'cny': float(max_cny)}
        self.reservations, self.receipts = {}, {}
        self.lock = threading.RLock()

    @staticmethod
    def _amounts(calls, tokens, cny):
        if type(calls) is not int or type(tokens) is not int or calls < 0 or tokens < 0 or not math.isfinite(cny) or cny < 0:
            raise ValueError('nonnegative finite resource amounts required')
        return {'calls': calls, 'tokens': tokens, 'cny': float(cny)}

    def _used(self):
        amounts = list(self.reservations.values()) + [r['charged'] for r in self.receipts.values()]
        return {k: sum(a[k] for a in amounts) for k in self.limits}

    def reserve(self, key, calls=1, tokens=0, cny=0.):
        amount = self._amounts(calls, tokens, cny)
        if not isinstance(key, str) or not key: raise ValueError('reservation key required')
        with self.lock:
            if key in self.reservations:
                if self.reservations[key] != amount: raise ValueError('reservation changed')
                return copy.deepcopy(amount)
            if key in self.receipts: raise ValueError('already settled request must not be re-reserved')
            used = self._used()
            if any(used[k] + amount[k] > self.limits[k] + (1e-12 if k == 'cny' else 0) for k in self.limits):
                raise BudgetExceeded('whole-task budget exhausted')
            self.reservations[key] = amount
            return copy.deepcopy(amount)

    def settle(self, key, calls=1, tokens=0, cny=0., known=True):
        amount = self._amounts(calls, tokens, cny)
        with self.lock:
            if key in self.receipts:
                old = self.receipts[key]
                if old['reported'] != amount or old['known'] != known: raise ValueError('receipt changed')
                return copy.deepcopy(old)
            if key not in self.reservations: raise ValueError('reservation required before settlement')
            reserved = self.reservations.pop(key)
            charged = amount if known else {k: max(amount[k], reserved[k]) for k in reserved}
            result = {'reserved': reserved, 'reported': amount, 'charged': charged, 'known': bool(known)}
            self.receipts[key] = result
            # Preserve real overage before raising; never erase spent money.
            if any(self._used()[k] > self.limits[k] + (1e-12 if k == 'cny' else 0) for k in self.limits):
                raise BudgetExceeded('actual receipt exceeded task budget; charged and stop')
            return copy.deepcopy(result)

    def snapshot(self):
        with self.lock:
            return {'limits': copy.deepcopy(self.limits), 'used_or_reserved': self._used(),
                    'reservations': copy.deepcopy(self.reservations), 'receipts': copy.deepcopy(self.receipts),
                    'unknown_requests': [k for k, r in self.receipts.items() if not r['known']]}
