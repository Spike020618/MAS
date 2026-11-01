import copy
import unittest

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.state_d2 import StateD2
from experiments.paired_deliberation.verification_d2 import VerificationD2
from experiments.paired_deliberation.training_evidence_view import COLUMNS
from experiments.tests.test_design_v6 import evidence


def registry(e=None):
    e=copy.deepcopy(e or evidence())
    view={'kind':'historical_training_prediction_evidence','case_id':e['case_id'],
          'current_case_labels_available':False,'candidate_columns':list(COLUMNS),
          'candidate_rows':[[m['movie_id']]+[0]*(len(COLUMNS)-1) for m in e['candidates_in_shared_retrieval_order']],
          'training_source_hash':'offline_fixture_training_hash',
          'historical_examples':[{'training_case_id':'train:1','training_prefix':'seeker: Please suggest Drama.',
                                  'historical_targets_in_pool':['1']}]}
    return EvidenceRegistry(e,view)


def claim(r,cid='c1',kind='catalog_fact',text='Movie 1 has catalog year 2001.',deps=None):
    predictive=kind in {'hypothesis','recommendation_prediction'}
    ref='train:1' if predictive else 'movie:1' if kind=='catalog_fact' else 'message:1'
    return {'claim_id':cid,'text':text,'claim_kind':kind,'evidence_refs':[{'source_id':r.resolve(ref),
        'use':'prediction_basis' if predictive else 'supports'}], 'candidate_ids':['1'],'depends_on':deps or [],
        'scope':None if kind=='catalog_fact' or predictive else {'scope':'utterance' if kind=='utterance' else 'preference_state',
        'subject':'seeker','predicate':'wants','object':'Drama','polarity':True,'time':'current'},
        'constraint_mode':'soft','stance':'support'}


def report(r,claims):return {'ranking':[m['movie_id'] for m in r.evidence['candidates_in_shared_retrieval_order'][:5]],'claims':claims,'updates':[]}


class D2ContractTests(unittest.TestCase):
    def test_training_case_and_stat_are_registered_prediction_bases_only(self):
        r=registry();c=claim(r,kind='hypothesis');r.validate_claim(c)
        stat=next(sid for sid,row in r.snapshot().items() if row['kind']=='train_stat')
        c['evidence_refs']=[{'source_id':stat,'use':'prediction_basis'}];r.validate_claim(c)
        for kind in ('catalog_fact','utterance','current_preference'):
            bad=claim(r,kind=kind);bad['evidence_refs']=[{'source_id':stat,'use':'supports'}]
            with self.assertRaisesRegex(ValueError,'authority'):r.validate_claim(bad)
        self.assertFalse(any(sid.startswith('train') for sid in r.dialogue()))
        self.assertFalse(any('train' in sid for sid in r.catalog()))

    def test_forged_historical_id_unknown_current_label_and_hard_prediction_rejected(self):
        r=registry();c=claim(r,kind='hypothesis')
        for change in ('source','hard'):
            bad=copy.deepcopy(c)
            if change=='source':bad['evidence_refs'][0]['source_id']='train_case:unseen'
            else:bad['constraint_mode']='hard'
            with self.assertRaises(ValueError):r.validate_claim(bad)
        e=evidence();e['relevant_movie_ids']=['1']
        with self.assertRaises(ValueError):registry(e)

    def test_registry_context_and_candidate_order_cannot_be_mutated_in_place(self):
        r=registry();r.evidence['candidates_in_shared_retrieval_order'].reverse()
        with self.assertRaisesRegex(ValueError,'mutated'):r.context_hash()

    def test_explicit_legacy_scope_is_never_guessed(self):
        from experiments.tests.test_design_v6 import claim as old_claim
        r=registry();c=old_claim(kind='explicit_preference')
        with self.assertRaisesRegex(ValueError,'explicit purpose'):r.migrate_legacy(c)


class D2StateTests(unittest.TestCase):
    def state(self,c=None):
        r=registry();c=c or claim(r)
        return r,StateD2({'current_request':report(r,[c])},r)
    def observe(self,s,v=None):
        target='current_request:c1'
        return s.observe('central',target,fingerprint(s.claims[target]),v or VerificationD2(s.registry,invalid_sources=s.invalid_sources))
    def patch(self,s,replacement=None):
        return {'target_claim_id':'current_request:c1','expected_hash':fingerprint(s.claims['current_request:c1']),'replacement':replacement}

    def test_bad_repair_preserves_committed_refutation_and_old_hash_is_rejected(self):
        r,s=self.state();stale=self.patch(s);self.observe(s)
        self.assertEqual(s.claims['current_request:c1']['status'],'retracted')
        before=copy.deepcopy(s.claims)
        invalid=claim(r,text='Movie 1 has catalog year 2000.');invalid['stance']='refute'
        with self.assertRaises(ValueError):s.propose('central','current_request:c1',self.patch(s,invalid),VerificationD2(r))
        with self.assertRaisesRegex(ValueError,'stale'):s.propose('central','current_request:c1',stale,VerificationD2(r))
        self.assertEqual(s.claims,before)
        s.propose('central','current_request:c1',self.patch(s,claim(r,text='Movie 1 has catalog year 2000.')),VerificationD2(r))
        self.assertEqual(s.claims['current_request:c1']['status'],'source_supported')

    def test_predictive_basis_is_consumed_without_truth_promotion(self):
        r=registry();c=claim(r,kind='hypothesis',text='Movie 1 might suit the seeker.')
        s=StateD2({'current_request':report(r,[c])},r);event=self.observe(s)
        self.assertEqual(event['verification']['verdict'],'not_evaluated')
        self.assertFalse(s.payload()['source_supported_claims']);self.assertTrue(s.payload()['prediction_basis_claims'])
        self.assertFalse(s.payload()['hard_exclusions'])
        updated=copy.deepcopy(c);updated['text']='Movie 1 may suit a similar historical request.'
        s.propose('central','current_request:c1',self.patch(s,updated),VerificationD2(r))
        self.assertEqual(s.events[-1]['action'],'revise_prediction')
        self.assertFalse(s.payload()['source_supported_claims'])

    def test_dependency_invalidation_and_fact_from_prediction_rejected(self):
        r=registry();c=claim(r);p=claim(r,'p','hypothesis',deps=['c1'])
        s=StateD2({'current_request':report(r,[c,p])},r)
        self.observe(s)
        self.assertEqual(s.claims['current_request:p']['status'],'invalidated')
        c['depends_on']=['p']
        with self.assertRaises(ValueError):StateD2({'current_request':report(r,[c,claim(r,'p','hypothesis')])},r)

    def test_evidence_revocation_does_not_turn_missing_support_into_falsehood(self):
        r=registry();c=claim(r,text='Movie 1 has catalog year 2000.');p=claim(r,'p','hypothesis')
        s=StateD2({'current_request':report(r,[c,p])},r);self.observe(s)
        s.invalidate_evidence(['train:1'],'Training provenance withdrawn in offline test.')
        self.assertEqual(s.claims['current_request:p']['status'],'basis_invalidated')
        self.assertEqual(s.claims['current_request:c1']['status'],'unresolved')
        self.assertFalse(s.payload()['hard_exclusions'])
        with self.assertRaisesRegex(ValueError,'context differs'):self.observe(s,VerificationD2(r))

    def test_control_whole_report_does_not_inherit_support_for_rewrites(self):
        r,s=self.state();self.observe(s)
        with self.assertRaises(ValueError):s.replace_owner('current_request',report(r,[claim(r)]))
        control=StateD2({'current_request':report(r,[claim(r,text='Movie 1 has catalog year 2000.')])},r,allow_whole_report=True)
        self.observe(control)
        control.replace_owner('current_request',report(r,[claim(r,text='Movie 1 has catalog year 2001.')]))
        self.assertEqual(control.claims['current_request:c1']['status'],'unverified')


if __name__=='__main__':unittest.main()
