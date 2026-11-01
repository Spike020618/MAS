import unittest
from experiments.paired_deliberation.assemble_light_state import null_id_row
from experiments.paired_deliberation.explore_light_register import complete_rows


class LightAssemblyTests(unittest.TestCase):
    def test_only_null_id_echo_of_single_text_parse_is_rebound(self):
        response={'claim_id':None,'kind':'unsupported','scope':None}
        result=null_id_row(response,'c3')
        self.assertEqual(result,{'claim_id':'c3','kind':'unsupported','scope':None})
        self.assertIsNone(response['claim_id'])
        self.assertIsNone(null_id_row({'claim_id':'different','kind':'unsupported','scope':None},'c3'))
    def test_duplicate_or_missing_id_rows_never_reused_by_position(self):
        row={'claim_id':'c1','kind':'hypothesis','scope':None}
        self.assertEqual(complete_rows({'claims':[row,row]},['c1']),{})
        self.assertEqual(complete_rows({'claims':[{'kind':'hypothesis','scope':None}]},['c1']),{})
        self.assertEqual(complete_rows(row,['c1']),{'c1':row})


if __name__=='__main__':unittest.main()
