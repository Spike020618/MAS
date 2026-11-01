import unittest
from experiments.mechanism import control_answer_order as c


class OrderTests(unittest.TestCase):
    def test_only_final_format_instruction_changes(self):
        case={'id':'x','question':'Example?','native_documents':[]}
        original=c.p.request(case,'native',[]);changed=c.reorder(original)
        self.assertEqual(changed['messages'][1:],original['messages'][1:])
        self.assertEqual(changed['messages'][0]['content'].replace(c.NEW,c.OLD),original['messages'][0]['content'])
        for key in original:
            if key!='messages':self.assertEqual(original[key],changed[key])
        self.assertIn(c.OLD,original['messages'][0]['content'])
    def test_unexpected_instruction_stops(self):
        with self.assertRaises(ValueError):c.reorder({'messages':[{'content':'different'}]})


if __name__=='__main__':unittest.main()
