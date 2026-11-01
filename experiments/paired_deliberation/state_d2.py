"""D2 observation-first state; facts and predictive bases have different authority."""
import copy

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.evidence_d2 import PREDICTIONS
from experiments.paired_deliberation.state_v6 import StateV6, ReplacementPreconditionError
from experiments.paired_deliberation.verification_d2 import VerificationD2


class StateD2:
    def __init__(self, reports, registry, allow_whole_report=False):
        self.registry, self.allow_whole_report = registry, allow_whole_report
        self.owners, self.claims, self.events, self.invalid_sources = set(reports), {}, [], set()
        for owner, report in reports.items():
            if not isinstance(owner, str) or not owner or ':' in owner: raise ValueError('invalid report owner')
            registry.validate_report(report)
            for claim in report['claims']:
                self.claims[owner + ':' + claim['claim_id']] = self._row(owner, claim)
        self._graph(self.claims)
        self._refresh(self.claims)

    def _row(self, owner, claim, revision=0):
        claim = StateV6._qualified_claim(owner, claim)
        own = 'basis_attributed' if claim['claim_kind'] in PREDICTIONS else 'unverified'
        return {'owner': owner, 'claim': claim, 'own_status': own, 'status': own, 'revision': revision,
                'verification': None, 'blocked_by': [], 'hard_constraint_active': False,
                'context_hash': self.registry.context_hash(self.invalid_sources)}

    @staticmethod
    def _graph(rows, allow_missing=False):
        StateV6._check_graph(rows, allow_missing)
        for row in rows.values():
            if row['claim']['claim_kind'] not in PREDICTIONS:
                if any(dep in rows and rows[dep]['claim']['claim_kind'] in PREDICTIONS for dep in row['claim']['depends_on']):
                    raise ValueError('prediction cannot serve as a factual certification premise')

    @staticmethod
    def _refresh(rows):
        done = set()
        def update(target):
            if target in done: return
            row = rows[target]
            for dep in row['claim']['depends_on']:
                if dep in rows: update(dep)
            blocked = [dep for dep in row['claim']['depends_on'] if dep not in rows or
                       rows[dep]['status'] not in {'source_supported', 'basis_attributed'}]
            row['blocked_by'] = blocked
            row['status'] = ('retracted' if row['own_status'] == 'retracted' else
                             'invalidated' if blocked else row['own_status'])
            row['hard_constraint_active'] = bool(row['status'] == 'source_supported' and
                row['claim']['claim_kind'] == 'current_preference' and row['claim']['constraint_mode'] == 'hard' and
                row['verification'] and row['verification'].get('hard_exclusion_witness'))
            done.add(target)
        for target in rows: update(target)

    def _target(self, reviewer, target, expected_hash):
        if reviewer not in self.owners and reviewer != 'central': raise ValueError('unknown reviewer')
        if target not in self.claims: raise ValueError('unknown target')
        row = self.claims[target]
        if expected_hash != fingerprint(row): raise ValueError('stale claim revision')
        return row

    def _router(self, verifier):
        if not isinstance(verifier, VerificationD2): raise ValueError('D2 purpose router required')
        if verifier.context_hash != self.registry.context_hash(self.invalid_sources): raise ValueError('verification context differs')

    def observe(self, reviewer, target, expected_hash, verifier, prefix='d2_observe'):
        current = self._target(reviewer, target, expected_hash); self._router(verifier)
        if (current.get('observed_claim_hash') == fingerprint(current['claim']) and
                current.get('observed_context_hash') == verifier.context_hash and current.get('verification') and
                current['verification'].get('verifier_identity') == verifier.service_id):
            verifier.validate_record(current['claim'], current['verification'])
            return {'action': 'observation_reused', 'target': target}
        record = verifier.verify(copy.deepcopy(current['claim']), prefix)
        verifier.validate_record(current['claim'], record)
        if (current.get('observed_claim_hash') == fingerprint(current['claim']) and
                current.get('observed_context_hash') == verifier.context_hash and current['verification'] == record):
            return {'action': 'observation_reused', 'target': target}
        if current['claim']['claim_kind'] in PREDICTIONS:
            status = 'basis_attributed' if record['verdict'] == 'not_evaluated' else 'basis_invalidated'
        else:
            status = {'supported': 'source_supported', 'refuted': 'retracted', 'insufficient': 'unresolved'}[record['verdict']]
        before, updated = copy.deepcopy(current), copy.deepcopy(self.claims)
        row = updated[target]
        row.update(own_status=status, revision=row['revision'] + 1, verification=copy.deepcopy(record),
                   observed_claim_hash=fingerprint(row['claim']), observed_context_hash=verifier.context_hash)
        self._refresh(updated); self.claims = updated
        event = {'action': 'observe_' + status, 'reviewer': reviewer, 'target': target, 'before': before,
                 'after': copy.deepcopy(updated[target]), 'verification': copy.deepcopy(record)}
        self.events.append(event)
        return copy.deepcopy(event)

    def propose(self, reviewer, target, proposal, verifier, prefix='d2_replacement'):
        if not isinstance(proposal, dict) or set(proposal) != {'target_claim_id', 'expected_hash', 'replacement'}:
            raise ValueError('D2 patch root fields')
        if proposal['target_claim_id'] != target: raise ValueError('assigned target differs')
        current = self._target(reviewer, target, proposal['expected_hash']); self._router(verifier)
        if (current.get('observed_claim_hash') != fingerprint(current['claim']) or
                current.get('observed_context_hash') != verifier.context_hash or
                current.get('verification', {}).get('verifier_identity') != verifier.service_id):
            raise ValueError('matching observation must be committed before repair')
        replacement = proposal['replacement']
        if replacement is None:
            event = {'action': 'acknowledge', 'target': target, 'reviewer': reviewer, 'state_changed': False}
            self.events.append(event); return copy.deepcopy(event)
        self.registry.validate_claim(replacement)
        old = current['claim']
        for field in ('claim_id', 'claim_kind', 'candidate_ids', 'depends_on'):
            if replacement[field] != old[field]: raise ValueError('replacement changes bound scope: ' + field)
        if old['scope'] is not None:
            for field in ('scope', 'subject', 'predicate', 'object', 'time'):
                if replacement['scope'][field] != old['scope'][field]: raise ValueError('replacement changes semantic scope: ' + field)
        if old['constraint_mode'] != 'hard' and replacement['constraint_mode'] == 'hard':
            raise ValueError('repair cannot silently upgrade constraint authority')
        prediction = old['claim_kind'] in PREDICTIONS
        if not prediction and (current['own_status'] != 'retracted' or current['verification']['verdict'] != 'refuted'):
            raise ReplacementPreconditionError('factual replacement requires committed refutation')
        record = verifier.verify(copy.deepcopy(replacement), prefix); verifier.validate_record(replacement, record)
        if record['verdict'] != ('not_evaluated' if prediction else 'supported'):
            raise ReplacementPreconditionError('replacement lacks valid basis or factual support')
        before, updated = copy.deepcopy(current), copy.deepcopy(self.claims)
        row = updated[target]
        row.update(claim=copy.deepcopy(replacement), own_status='basis_attributed' if prediction else 'source_supported',
                   revision=row['revision'] + 1, verification=copy.deepcopy(record),
                   observed_claim_hash=fingerprint(replacement), observed_context_hash=verifier.context_hash)
        self._refresh(updated); self.claims = updated
        event = {'action': 'revise_prediction' if prediction else 'replace_fact', 'reviewer': reviewer,
                 'target': target, 'before': before, 'after': copy.deepcopy(updated[target])}
        self.events.append(event); return copy.deepcopy(event)

    def replace_owner(self, owner, report):
        if not self.allow_whole_report: raise ValueError('whole-report revision restricted to control condition')
        if owner not in self.owners: raise ValueError('unknown report owner')
        self.registry.validate_report(report)
        before = copy.deepcopy(self.claims)
        updated = {k: copy.deepcopy(v) for k, v in before.items() if v['owner'] != owner}
        for claim in report['claims']:
            row = self._row(owner, claim); target = owner + ':' + claim['claim_id']
            row['revision'] = before.get(target, {}).get('revision', 0) + 1
            if target in before and row['claim'] == before[target]['claim']:
                row = copy.deepcopy(before[target]); row['revision'] += 1
            if set(r['source_id'] for r in claim['evidence_refs']) & self.invalid_sources:
                row['own_status'] = 'basis_invalidated' if claim['claim_kind'] in PREDICTIONS else 'unresolved'
            updated[target] = row
        introduced = {k for k in updated if k not in before or updated[k]['claim'] != before[k]['claim']}
        if any(dep not in updated for k in introduced for dep in updated[k]['claim']['depends_on']):
            raise ValueError('new report introduces missing premise')
        self._graph(updated, allow_missing=True); self._refresh(updated); self.claims = updated
        self.events.append({'action': 'whole_report', 'owner': owner, 'before': before, 'after': copy.deepcopy(updated)})

    def invalidate_evidence(self, source_ids, reason):
        """Trusted provenance maintenance, not an actor operation to erase opposition."""
        if not isinstance(reason, str) or not reason.strip(): raise ValueError('invalidation reason required')
        ids = {self.registry.resolve(sid) for sid in source_ids}
        self.invalid_sources |= ids
        before, updated = copy.deepcopy(self.claims), copy.deepcopy(self.claims)
        for row in updated.values():
            refs = {ref['source_id'] for ref in row['claim']['evidence_refs']}
            if row['claim']['claim_kind'] in PREDICTIONS:
                row['own_status'] = 'basis_invalidated' if refs & self.invalid_sources else 'basis_attributed'
            else:
                # A changed evidence context cannot retain cached state/conflict judgements.
                row['own_status'] = 'unresolved'
            row.update(verification=None, revision=row['revision'] + 1,
                       context_hash=self.registry.context_hash(self.invalid_sources))
            row.pop('observed_context_hash', None); row.pop('observed_claim_hash', None)
        self._refresh(updated); self.claims = updated
        self.events.append({'action': 'invalidate_evidence', 'sources': sorted(ids), 'reason': reason,
                            'before': before, 'after': copy.deepcopy(updated)})

    def payload(self):
        supported = {k: copy.deepcopy(v) for k, v in self.claims.items() if v['status'] == 'source_supported'}
        predictions = {k: copy.deepcopy(v) for k, v in self.claims.items() if v['status'] == 'basis_attributed'}
        uncertain = {k: copy.deepcopy(v) for k, v in self.claims.items() if v['status'] not in {'source_supported', 'basis_attributed', 'retracted'}}
        exclusions = []
        for target, row in self.claims.items():
            if not row['hard_constraint_active']: continue
            witness = row['verification']['hard_exclusion_witness']; genre = witness['genre'].strip().casefold()
            mids = [m['movie_id'] for m in self.registry.evidence['candidates_in_shared_retrieval_order']
                    if genre in {g.strip().casefold() for g in m.get('genres', [])}]
            exclusions.append({'target': target, 'candidate_ids': mids, 'witness': copy.deepcopy(witness)})
        return {'source_supported_claims': supported, 'prediction_basis_claims': predictions,
                'uncertain_claims': uncertain, 'retracted_ids': sorted(k for k, v in self.claims.items() if v['status'] == 'retracted'),
                'hard_exclusions': exclusions, 'context_hash': self.registry.context_hash(self.invalid_sources),
                'instruction': 'Source-supported semantic judgements remain model proxies. Prediction bases are soft '
                               'attributions, never current-user truths. Only hard_exclusions supply mechanical exclusions. '
                               'Unresolved/invalidated/retracted assertions cannot supply certified restrictions.'}
