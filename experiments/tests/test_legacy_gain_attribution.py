"""Guard historical replay semantics and conservative source grouping."""
from copy import deepcopy
import unittest

from experiments.mechanism.attribute_legacy_gain import vote, record_view, transitions
from experiments.mechanism.audit_legacy_source_groups import split_question, source_labels, grouped_sensitivity
from experiments.mechanism.audit_legacy_conclusion_parsing import conclusion
from experiments.mechanism.extend_legacy_sensitivity import source_screen
from experiments.mechanism.audit_legacy_aggregation import phi, decomposition


def record():
    return {'ground_truth': 'yes', 'final_decision': 'yes', 'rounds': [
        {'round_id': rid, 'agent_id': aid, 'stance': 'yes', 'chosen_action': 'low_effort',
         'token_input': 1, 'token_output': 1,
         'action_trace': [{'stage': 'choose', 'backend': 'selector', 'model': 'choice'},
                          {'stage': 'low_effort', 'backend': 'executor', 'model': 'report'}]}
        for rid in range(1, 4) for aid in range(5)]}


class LegacyGainTests(unittest.TestCase):
    def test_source_screen_keeps_first_input_not_best_result(self):
        cache = [{'question_id':i,'question':q} for i,q in enumerate([
            'Question: Q\nProposed answer: a','Question: Q\nProposed answer: b',
            'Question: Z\nProposed answer: c','Question: Z\nProposed answer: d'])]
        s = source_screen(cache,{'train':['0'],'dev':[],'test':['1','2','3']},'test')
        self.assertEqual(s['keep'],['2'])
        self.assertEqual([x['id'] for x in s['excluded']],['1','3'])

    def test_error_correlation_handles_zero_variance(self):
        self.assertIsNone(phi([True,True],[True,False])['phi'])
        self.assertEqual(phi([True,False],[True,False])['phi'],1)
        self.assertEqual(phi([True,False],[False,True])['phi'],-1)
        with self.assertRaises(ValueError):phi([],[])

    def test_zero_discordance_is_not_zero_uncertainty(self):
        from experiments.mechanism.audit_legacy_explanatory_controls import paired
        values = {('domain','a'):True,('domain','b'):False}
        r = paired(values,values)
        self.assertIsNone(r['stratified_bootstrap_percentile_95_descriptive'])
        self.assertEqual(r['exact_mcnemar_p_unadjusted_descriptive'],1.)
        with self.assertRaises(ValueError):paired(values,{})

    def test_schema_boundary_counterexample_not_factual_validation(self):
        from experiments.mechanism.audit_legacy_explanatory_controls import interface_fixture_counterexamples
        r = interface_fixture_counterexamples()
        self.assertTrue(r['constructed_not_measured_model_behavior'])
        self.assertTrue(r['empty_E_AEIC_boundary_explicit'])
        self.assertFalse(r['generic_with_evidence_words_boundary_explicit'])

    def test_aggregation_decomposition_is_an_identity_not_a_causal_claim(self):
        left = {'n':10,'majority_correct':6,'uniform_random_single_expected_correct':5.,'majority_minus_random_single_expected_correct':1.}
        right = {'n':10,'majority_correct':9,'uniform_random_single_expected_correct':7.,'majority_minus_random_single_expected_correct':2.}
        r = decomposition(left,right)
        self.assertEqual((r['majority_net_correct_gain'],r['mean_individual_correct_gain'],r['voting_surplus_gain']),(3,2.,1.))
        bad = dict(right);bad['majority_minus_random_single_expected_correct']=4
        with self.assertRaises(ValueError):decomposition(left,bad)

    def test_route_partition_reconciles_correct_vote_changes(self):
        from experiments.mechanism.audit_legacy_route_pairs import partition
        left = record();left.update(dataset='mock',question_id='q',condition='independent__no_mechanism@0.2')
        right = deepcopy(left);right['condition']='independent__static_aeic@0.2'
        left['rounds'][0]['stance']='no'
        r = partition([left,right],round_id=1)
        self.assertEqual(r['paired_slots'],5)
        self.assertEqual(r['total_net_correct_votes'],1)
        self.assertEqual(r['same_bundle_net_votes'],1)

    def test_exact_proxy_game_check_does_not_conflate_NE_with_global(self):
        from experiments.mechanism.audit_legacy_theory_bridge import exact_check
        from experiments.core.mechanism.upgrade_game import Contract
        c = Contract((0,0,0,1,0,1,1,1),((0,.01),)*3,1,'offline-theory-fixture')
        local = exact_check(c,[0,0,0])
        self.assertEqual(local['directed_unilateral_edges'],24)
        self.assertEqual(local['max_selected_regret'],0.)
        self.assertGreater(local['global_proxy_objective_gap'],0.)
        global_choice = exact_check(c,[1,0,1])
        self.assertEqual(global_choice['global_proxy_objective_gap'],0.)

    def test_semantic_fixture_does_not_claim_unchanged_generation_prompt(self):
        from experiments.mechanism.audit_legacy_theory_bridge import semantic_scope_fixture
        r = semantic_scope_fixture({'states':{},'pooled':[.5]*8})
        self.assertTrue(r['same_C_patterns_different_AEI_same_contract'])
        self.assertTrue(r['upgrade_prompt_changes_with_AEI'])
    def test_literal_suffix_not_placeholder(self):
        self.assertEqual(conclusion('Yes or No: No', 'yes', 'literal_yes_no_only'), 'no')
        self.assertEqual(conclusion('No/Yes: Yes!', 'no', 'literal_yes_no_only'), 'yes')

    def test_illegal_parser_mode_rejected(self):
        with self.assertRaises(ValueError): conclusion('Yes', 'yes', 'optimize_to_gold')

    def test_transport_probe_is_offline_and_distinguishes_requested_model(self):
        from experiments.mechanism.audit_legacy_measurement import client_probe
        r = client_probe()
        self.assertTrue(r['transport_fully_mocked'])
        self.assertFalse(r['seed_forwarded_even_if_explicitly_supplied'])
        self.assertFalse(r['response_model_retained_as_client_model'])

    def test_contract_does_not_enter_fixed_action_execution(self):
        from experiments.mechanism.audit_legacy_prompt_boundary import capture
        for action in ['low_effort', 'aeic_effort', 'verify', 'defer_imitate']:
            a,b = capture(action, 'contract-a', 0), capture(action, 'contract-b', -9)
            self.assertEqual(a['execution'], b['execution'])
            self.assertNotEqual(a['choice'], b['choice'])

    def test_ambiguous_list_not_semantically_repaired(self):
        self.assertEqual(conclusion('Yes or No: False, True', 'yes', 'literal_yes_no_only'), 'yes')
        self.assertEqual(conclusion('Yes or No: False, True', 'yes', 'unresolved_placeholder_abstains'), 'neutral')
        self.assertEqual(conclusion('No: the word yes appears in a quotation', 'no', 'unresolved_placeholder_abstains'), 'no')

    def test_historical_tie_order_preserved(self):
        self.assertEqual(vote(['no', 'yes']), 'no')
        self.assertEqual(vote(['yes', 'no']), 'yes')
        self.assertIsNone(vote([]))

    def test_duplicate_agent_rejected(self):
        r = record()
        r['rounds'][0]['agent_id'] = 1
        with self.assertRaises(ValueError): record_view(r)

    def test_final_mismatch_rejected(self):
        r = record(); r['final_decision'] = 'no'
        with self.assertRaises(ValueError): record_view(r)

    def test_view_does_not_mutate(self):
        r = record(); before = deepcopy(r)
        self.assertEqual(record_view(r)[0]['correct_votes'], 5)
        self.assertEqual(r, before)

    def test_paired_errors(self):
        self.assertEqual(transitions([False, True], [True, False])['wrong_correct'], 1)
        with self.assertRaises(ValueError): transitions([True], [])

    def test_proposal_does_not_split_multiline_answer(self):
        s,p = split_question('Question: Q\nProposed answer: first\nsecond\nIs this proposed answer factually correct?')
        self.assertEqual((s,p), ('Q', 'first\nsecond'))

    def test_native_ambiguous_not_forced(self):
        cache = [{'question_id': 0, 'question': 'Question: Q\nProposed answer: a\nIs this proposed answer factually correct?', 'ground_truth': 'yes'}]
        source = [{'question': 'Q', 'choices': ['a', 'b'], 'answer': 0},
                  {'question': 'Q', 'choices': ['a', 'b'], 'answer': 1}]
        self.assertEqual(source_labels('MMLU-Hard', cache, source)['rows'][0]['status'], 'ambiguous_source_label')

    def test_native_mismatch_not_relabelled(self):
        cache = [{'question_id': 0, 'question': 'Question: Q\nProposed answer: a.\nIs this proposed answer factually correct?', 'ground_truth': 'no'}]
        native = [{'question': 'Q', 'mc1_targets': {'a': 1, 'b': 0}}]
        result = source_labels('TruthfulQA', cache, native)['rows'][0]
        self.assertEqual(result['status'], 'label_mismatch')
        self.assertFalse(result['exact_answer_match'])
        self.assertEqual(cache[0]['ground_truth'], 'no')

    def test_group_screen_independent_of_outcome(self):
        cache = [{'question_id': i, 'question': text} for i,text in enumerate([
            'Question: Q\nProposed answer: a', 'Question: Q\nProposed answer: b',
            'Question: Z\nProposed answer: c', 'Question: Z\nProposed answer: d'])]
        split = {'train': ['0'], 'dev': [], 'test': ['1', '2', '3']}
        a = grouped_sensitivity(cache, split, {'1': (False, True), '2': (False, True), '3': (True, False)})
        b = grouped_sensitivity(cache, split, {'1': (True, False), '2': (True, False), '3': (False, True)})
        self.assertEqual(a['excluded_records'], b['excluded_records'])
        self.assertEqual(a['grouped_input_clean_first_test_only']['n'], 1)
        self.assertEqual(a['test_records_crossing_train_dev_stem'], 1)


if __name__ == '__main__':
    unittest.main()
