import copy
import json
import unittest
from experiments.paired_deliberation.response_format_metadata import project


class MetadataTests(unittest.TestCase):
    def test_only_exact_marker_is_removed_and_input_unchanged(self):
        request={'messages':[{'content':''},{'content':json.dumps({'schema':{'relations':[],'reason':''}})}]}
        original={'report':{'relations':[],'reason':'test','schema':'json_object'}}
        before=copy.deepcopy(original);out,changed=project(original,request)
        self.assertTrue(changed);self.assertEqual(original,before);self.assertNotIn('schema',out['report'])
        for bad in ({'relations':[],'schema':'json_object'},
                    {'relations':[],'reason':'test','schema':'different'},
                    {'relations':[],'reason':'test','schema':'json_object','extra':'x'}):
            record={'report':bad};out,changed=project(record,request)
            self.assertFalse(changed);self.assertEqual(out,record)


if __name__=='__main__':unittest.main()
