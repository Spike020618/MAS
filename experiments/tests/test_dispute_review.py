import json
import unittest
from experiments.mechanism.probe_dispute_review import cards,prepare,request,review_status


class DisputeReviewTest(unittest.TestCase):
    def test_sealed_shared_baseline(self):
        items,_=prepare()
        self.assertEqual(len(items),12)
        self.assertEqual(sum(i['group']=='valid_disagreement' for i in items),7)
        self.assertTrue(all('label' not in i['case'] for i in items))

    def test_fixed_card_selection(self):
        item={'reports':[{'agent':0,'C':'yes','available_AEI':{'I':'verbatim','A':'other'}},
                         {'agent':1,'C':'no','available_AEI':{'A':'skip'}},
                         {'agent':2,'C':'yes','available_AEI':{}}]}
        result=cards(item,{'C':'no'})
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['verbatim_claim'],'verbatim')
        self.assertEqual(result[0]['ref'],'peer:0:I')

    def test_equal_review_resources_and_initial(self):
        item=prepare()[0][0];a=request(item,0,'ordinary');b=request(item,0,'targeted')
        self.assertEqual(a['messages'][0],b['messages'][0])
        for k in ['model','backend','max_tokens','temperature','response_format']:self.assertEqual(a[k],b[k])
        pa=json.loads(a['messages'][1]['content']);pb=json.loads(b['messages'][1]['content'])
        self.assertNotIn('dispute_cards',pa)
        del pb['dispute_cards']
        self.assertEqual(pa,pb)

    def test_audit_references_and_external_claim(self):
        obj={'review':{'decision':'keep','checks':[{'ref':'self:A','assessment':'uncertain','reason':'not checked'}],
                       'change_basis':'no new basis','external_check':False}}
        self.assertTrue(review_status(json.dumps(obj),['self:A'])['valid'])
        self.assertFalse(review_status(json.dumps(obj),[])['valid'])
        obj['review']['external_check']=True
        self.assertFalse(review_status(json.dumps(obj),['self:A'])['valid'])
