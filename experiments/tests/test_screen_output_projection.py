import unittest
from experiments.paired_deliberation.screen_output_projection import project
from experiments.paired_deliberation.screen_domestic import ranking_request
from experiments.tests.test_design_v6 import evidence


class ProjectionTests(unittest.TestCase):
    def test_wrapper_and_known_id_representation_preserve_choice_and_raw(self):
        record={'report':{'answer':[1,'2','3','4','5']}}
        result,ops=project(record,ranking_request(evidence()),evidence())
        self.assertEqual(result['report'],{'ranking':['1','2','3','4','5']})
        self.assertEqual(record['report'],{'answer':[1,'2','3','4','5']})
        self.assertEqual(len(ops),2)
    def test_unknown_extra_fields_and_unknown_ids_not_repaired(self):
        request=ranking_request(evidence())
        for record in ({'report':{'answer':{'ranking':['1']},'extra':'x'}},
                       {'report':{'ranking':[999,True,1.0,'2','3']}}):
            result,ops=project(record,request,evidence())
            self.assertEqual(result,record);self.assertFalse(ops)


if __name__=='__main__':unittest.main()
