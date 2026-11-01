"""D2 typed evidence authority. Historical predictions never become dialogue."""
import copy
import math

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.atomic_evidence import validate_atom
from experiments.paired_deliberation.contracts_v6 import check_ranking
from experiments.paired_deliberation.training_evidence_view import COLUMNS

KINDS = {'catalog_fact', 'utterance', 'current_preference', 'hypothesis', 'recommendation_prediction'}
PREDICTIONS = {'hypothesis', 'recommendation_prediction'}
FIELDS = {'claim_id', 'text', 'claim_kind', 'evidence_refs', 'candidate_ids', 'depends_on',
          'scope', 'constraint_mode', 'stance'}
EVIDENCE_FIELDS = {'case_id', 'observations', 'previous_mentions_excluded_not_asserted_seen',
                   'candidates_in_shared_retrieval_order'}


class EvidenceRegistry:
    def __init__(self, evidence, training_view=None):
        if set(evidence) != EVIDENCE_FIELDS: raise ValueError('unexpected evidence fields')
        self.evidence = copy.deepcopy(evidence)
        self._entries, self._aliases = {}, {}
        candidates = evidence['candidates_in_shared_retrieval_order']
        ids = [m['movie_id'] for m in candidates]
        if len(ids) < 5 or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError('invalid closed candidate pool')
        for source in evidence['observations']:
            if not isinstance(source.get('text'), str) or not isinstance(source.get('role'), str):
                raise ValueError('dialogue source needs original text and role')
            old = source['source_id']
            self._add('dialogue:' + old, 'dialogue', old, source, ['factual_reference', 'prediction_basis'])
        for source in candidates:
            self._add('catalog:' + source['movie_id'], 'catalog', source['source_id'], source,
                      ['factual_reference', 'prediction_basis'])
        self.training_view = copy.deepcopy(training_view)
        if training_view is not None:
            view = training_view
            if (view.get('kind') != 'historical_training_prediction_evidence' or
                    view.get('current_case_labels_available') is not False or view.get('case_id') != evidence['case_id'] or
                    tuple(view.get('candidate_columns', [])) != COLUMNS or
                    [r[0] for r in view.get('candidate_rows', [])] != ids or
                    not isinstance(view.get('training_source_hash'), str) or not view['training_source_hash']):
                raise ValueError('training view provenance, schema or pool mismatch')
            for row in view['candidate_rows']:
                if len(row) != len(COLUMNS) or any(isinstance(v, float) and not math.isfinite(v) for v in row):
                    raise ValueError('invalid predictive score row')
                mid = row[0]
                source = {'case_id': evidence['case_id'], 'candidate': dict(zip(COLUMNS, row)),
                          'training_source_hash': view['training_source_hash'], 'not_current_user_fact': True}
                sid = 'train_stat:' + fingerprint(evidence['case_id'])[:12] + ':' + mid
                self._add(sid, 'train_stat', None, source, ['prediction_basis'])
            for example in view.get('historical_examples', []):
                old = example['training_case_id']
                if old == evidence['case_id'] or not isinstance(example.get('training_prefix'), str):
                    raise ValueError('historical case overlaps serving case or lacks prefix')
                targets = example.get('historical_targets_in_pool')
                if not isinstance(targets, list) or not set(targets) <= set(ids):
                    raise ValueError('historical outcome targets outside displayed pool')
                self._add('train_case:' + old, 'train_case', old, example, ['prediction_basis'])
        self.integrity = self._digest()

    def _add(self, sid, kind, alias, content, uses):
        if sid in self._aliases and self._aliases[sid] != sid: raise ValueError('canonical evidence ID collides with alias')
        if sid in self._entries: raise ValueError('duplicate registered evidence ID')
        self._entries[sid] = {'source_id': sid, 'kind': kind, 'content': copy.deepcopy(content),
            'content_hash': fingerprint(content), 'allowed_uses': list(uses),
            'visible_at': {'case_id': self.evidence['case_id'], 'boundary': 'frozen dialogue prefix or training snapshot'}}
        self._aliases[sid] = sid
        if alias:
            if alias in self._aliases and self._aliases[alias] != sid: raise ValueError('ambiguous evidence alias')
            self._aliases[alias] = sid

    def _digest(self):
        return fingerprint({'entries': self._entries, 'evidence': self.evidence, 'training_view': self.training_view})

    def assert_intact(self):
        if self._digest() != self.integrity: raise ValueError('evidence registry mutated; create a new version')

    def snapshot(self): return copy.deepcopy(self._entries)

    def resolve(self, ref):
        if not isinstance(ref, str) or ref not in self._aliases: raise ValueError('unregistered evidence reference')
        return self._aliases[ref]

    def entry(self, ref): return copy.deepcopy(self._entries[self.resolve(ref)])

    def context_hash(self, invalid_sources=()):
        self.assert_intact()
        return fingerprint({'registry': self.integrity, 'invalid_sources': sorted(invalid_sources)})

    def dialogue(self, invalid_sources=()):
        invalid = set(invalid_sources)
        return {sid: copy.deepcopy(row['content']) for sid, row in self._entries.items()
                if row['kind'] == 'dialogue' and sid not in invalid}

    def catalog(self, invalid_sources=()):
        invalid = set(invalid_sources)
        return {row['content']['source_id']: copy.deepcopy(row['content']) for sid, row in self._entries.items()
                if row['kind'] == 'catalog' and sid not in invalid}

    def validate_claim(self, claim):
        self.assert_intact()
        if not isinstance(claim, dict) or set(claim) != FIELDS: raise ValueError('D2 typed claim fields')
        cid = claim['claim_id']
        if not isinstance(cid, str) or not cid or len(cid) > 80 or ':' in cid: raise ValueError('invalid local claim ID')
        if not isinstance(claim['text'], str) or not claim['text'].strip(): raise ValueError('empty claim')
        kind = claim['claim_kind']
        if kind not in KINDS or claim['stance'] not in {'support', 'oppose', 'unknown'}: raise ValueError('claim kind/stance')
        if claim['constraint_mode'] not in {'hard', 'soft', 'unknown'}: raise ValueError('constraint mode')
        mids = claim['candidate_ids']; pool = {r['movie_id'] for r in self.evidence['candidates_in_shared_retrieval_order']}
        if not isinstance(mids, list) or any(not isinstance(m, str) for m in mids) or len(set(mids)) != len(mids) or not set(mids) <= pool:
            raise ValueError('candidate binding outside pool or duplicate')
        deps = claim['depends_on']
        if (not isinstance(deps, list) or any(not isinstance(d, str) or not d or d.count(':') > 1 for d in deps)
                or len(set(deps)) != len(deps)): raise ValueError('invalid dependencies')
        if kind in {'utterance', 'current_preference'}:
            scope = claim['scope']
            expected = 'utterance' if kind == 'utterance' else 'preference_state'
            if not isinstance(scope, dict) or set(scope) != {'scope', 'subject', 'predicate', 'object', 'polarity', 'time'}:
                raise ValueError('explicit dialogue scope required')
            validate_atom({**scope, 'reason': 'Declared scope; independently parsed at verification.'})
            if scope['scope'] != expected: raise ValueError('claim kind and declared scope differ')
        elif claim['scope'] is not None: raise ValueError('catalog/prediction scope must be null')
        if claim['constraint_mode'] == 'hard':
            scope = claim['scope'] or {}
            if not (kind == 'current_preference' and scope.get('predicate') == 'wants' and
                    scope.get('polarity') is False and scope.get('time') == 'current'):
                raise ValueError('only explicit current exclusions may request hard constraints')
        refs = claim['evidence_refs']
        if not isinstance(refs, list) or not refs: raise ValueError('claim requires registered evidence basis')
        seen = set()
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {'source_id', 'use'}: raise ValueError('typed evidence reference')
            sid = ref['source_id']
            if sid not in self._entries or sid in seen: raise ValueError('unregistered/duplicate canonical evidence reference')
            seen.add(sid); source = self._entries[sid]
            if kind in PREDICTIONS:
                if ref['use'] != 'prediction_basis': raise ValueError('prediction references cannot certify facts')
            else:
                if ref['use'] not in {'supports', 'refutes'}: raise ValueError('factual reference relation')
                required = 'catalog' if kind == 'catalog_fact' else 'dialogue'
                if source['kind'] != required: raise ValueError('evidence authority does not support claim kind')
        return claim

    def validate_report(self, report):
        if not isinstance(report, dict) or set(report) != {'ranking', 'claims', 'updates'}: raise ValueError('D2 report fields')
        check_ranking({'ranking': report['ranking']}, self.evidence)
        if report['updates'] != [] or not isinstance(report['claims'], list) or len(report['claims']) > 3:
            raise ValueError('D2 report allows at most three claims and no direct updates')
        ids = []
        for claim in report['claims']: self.validate_claim(claim); ids.append(claim['claim_id'])
        if len(set(ids)) != len(ids): raise ValueError('duplicate local claim ID')
        return report

    def migrate_legacy(self, claim, explicit_kind=None, scope=None):
        mapping = {'item_fact': 'catalog_fact', 'preference_hypothesis': 'hypothesis',
                   'recommendation_prediction': 'recommendation_prediction'}
        kind = explicit_kind or mapping.get(claim['claim_type'])
        if kind is None: raise ValueError('legacy explicit preference requires an explicit purpose/scope; no semantic guessing')
        result = {'claim_id': claim['claim_id'], 'text': claim['text'], 'claim_kind': kind,
            'evidence_refs': [{'source_id': self.resolve(sid), 'use': 'prediction_basis' if kind in PREDICTIONS else 'supports'}
                              for sid in claim['source_ids']],
            'candidate_ids': copy.deepcopy(claim['candidate_ids']), 'depends_on': copy.deepcopy(claim['depends_on']),
            'scope': copy.deepcopy(scope), 'constraint_mode': 'soft', 'stance': claim['stance']}
        self.validate_claim(result)
        return result
