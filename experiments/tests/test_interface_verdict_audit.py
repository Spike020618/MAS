import json
import unittest
from experiments.mechanism.audit_interface_verdicts import verdict, run


class VerdictAuditTests(unittest.TestCase):
    def test_explanation_no_does_not_override_yes(self):
        for p, field in [('generic_structure', 'answer'), ('aeic', 'C')]:
            self.assertEqual(verdict(json.dumps({field: 'Yes: No, vampires are not real.'}), p)['stance'], 'yes')

    def test_final_marker(self):
        self.assertEqual(verdict('No possibilities remain.\n\n**Final answer:**\nYes', 'free_text')['stance'], 'yes')
        self.assertEqual(verdict('There is no evidence.  \nNo', 'free_text')['stance'], 'no')

    def test_conflicting_explicit_answers_abstain(self):
        self.assertIsNone(verdict('Yes.\n\nNo.', 'free_text')['stance'])
        self.assertIsNone(verdict('Final answer: Yes. Final answer: No.', 'free_text')['stance'])

    def test_no_incidental_match(self):
        self.assertIsNone(verdict('There is no evidence.', 'free_text')['stance'])
        self.assertIsNone(verdict('{"answer": 1}', 'generic_structure')['stance'])

    def test_full_offline_audit(self):
        import socket
        from unittest.mock import patch
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
            data = run()
        self.assertEqual(len(data['all_records']), 270)
        self.assertEqual(len(data['original_generic_AEIC_discordances']), 7)
        self.assertEqual(data['new_model_calls'], 0)


if __name__ == '__main__':
    unittest.main()
