import json
import unittest
from experiments.paired_deliberation.source_alias_d2 import canonicalize


class SourceAliasTests(unittest.TestCase):
    def request(self,sources):return {'kind':'utterance_witness_d2','messages':[{'content':''},{'content':json.dumps({'sources':sources})}]}
    def test_only_unique_supplied_alias_is_mapped_without_quote_change(self):
        r={'report':{'witnesses':[{'source_id':'message:0','quote':'original quotation'}],'reason':'original reason'}}
        request=self.request({'dialogue:message:0':{'source_id':'message:0','text':'original quotation'}})
        fixed,ops=canonicalize(r,request)
        self.assertEqual(fixed['report']['witnesses'][0],{'source_id':'dialogue:message:0','quote':'original quotation'})
        self.assertEqual(r['report']['witnesses'][0]['source_id'],'message:0');self.assertEqual(len(ops),1)
    def test_unknown_historical_or_ambiguous_id_not_repaired(self):
        request=self.request({'dialogue:one':{'source_id':'same'},'dialogue:two':{'source_id':'same'}})
        for sid in ('same','redial:17072','train_case:redial:17072','invented'):
            r={'report':{'witnesses':[{'source_id':sid,'quote':'x'}],'reason':'x'}}
            fixed,ops=canonicalize(r,request)
            self.assertEqual(fixed,r);self.assertFalse(ops)


if __name__=='__main__':unittest.main()
