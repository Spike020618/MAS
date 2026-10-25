import unittest
from experiments.mechanism.analyze_deliberation_stagnation import analyze


class DeliberationStagnationTests(unittest.TestCase):
    def test_counts_and_disagreement_correction(self):
        result=analyze()
        self.assertEqual(result['n_runs'],32)
        self.assertEqual(result['unanimous_runs'],29)
        self.assertEqual(len(result['disagreements']),3)
        self.assertEqual(result['n_disagreement_question_ids'],2)
        self.assertEqual(result['unanimous_correct_runs'],20)
        self.assertEqual(result['unanimous_wrong_runs'],9)
        self.assertTrue(result['all_C_unchanged'])

    def test_body_changes_not_mistaken_for_conclusion_changes(self):
        result=analyze()
        self.assertEqual(result['revisions_with_any_AEI_text_change'],39)
        self.assertEqual(sum(result['valid_reference_dispositions_by_initial_C_relation'].values()),82)
        self.assertEqual(result['valid_reference_dispositions_by_initial_C_relation']['different_C/uncertain'],6)
        self.assertEqual(result['valid_reference_dispositions_by_initial_C_relation']['different_C/reject'],2)

    def test_visibility_limits_and_unidentified_causes(self):
        result=analyze()
        self.assertEqual(result['initial_reports_with_sources'],16)
        self.assertEqual(result['initial_reports_without_sources'],80)
        self.assertEqual(result['revision_payloads_with_direct_sources_field'],0)
        self.assertEqual(result['peer_messages_with_structured_provenance_fields'],0)
        self.assertFalse(result['literal_keep_C_instruction_detected'])
        self.assertIn('causal effect of own-answer anchoring',result['nonidentifiable_without_new_interventions'])

    def test_duplicate_prompt_instruction_reproducible_without_calls(self):
        from experiments.mechanism.pilot_source_sharing_neutral import original_prepare,prepare
        _,original=original_prepare()
        self.assertIs(original[0]['messages'],original[1]['messages'])
        _,neutral=prepare()
        marker='Answer the question using your own knowledge and any provided source excerpts.'
        self.assertEqual(neutral[0]['messages'][0]['content'].count(marker),2)


if __name__=='__main__':unittest.main()
