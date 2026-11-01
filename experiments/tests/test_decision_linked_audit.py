import json,tempfile,unittest
from pathlib import Path
from experiments.audits.audit_decision_linked import classify_probe_case,classify_review_event,main_audit

def qa_task():return {'id':'t1','kind':'qa','prompt':'Can you visit the NYPL and Six Flags in one day?','options':['yes','no'],'candidates':[],'source_catalog':{'s1':{'text':'The New York Public Library main branch is in Manhattan. Six Flags Great Escape is upstate near Lake George.'}},'metadata':{}}
def qa_arm(decision,valid=True,choice=None,patch=None,error=None):
 return {'decision':decision,'event':{'valid':valid,'accepted':False,'error':error,'report':{'decision_effect':{'choice':choice,'reason':'r'}if choice else None,'patch':patch,'verdict':None,'claim':''}}}
def qa_case():return {'task_id':'t1','kind':'qa','initial_decision':'no','action':{'target_id':'claim:0','source_ids':['s1']},'arms':{'ordinary':qa_arm('yes',choice=None),'linked':qa_arm('no',choice='keep')}}

class ProbeClassificationTests(unittest.TestCase):
 def test_qa_three_questions(self):
  r=classify_probe_case(qa_case(),qa_task(),{'kind':'qa','answer':'yes'})
  self.assertEqual(r['q1_initial_quality'],0.);self.assertFalse(r['q1_initial_correct'])
  self.assertEqual(r['arms']['linked']['decision_effect']['choice'],'keep');self.assertFalse(r['arms']['linked']['changed']);self.assertTrue(r['arms']['ordinary']['changed'])
  self.assertGreaterEqual(r['q2']['question_overlap'][0]['shared'],2)

 def test_rec_mechanical_opportunity_requires_pool_target_outside_top10(self):
  task={'id':'t2','kind':'rec','prompt':'Rec','candidates':[{'movie_id':'1'},{'movie_id':'2'},{'movie_id':'3'}],'source_catalog':{},'metadata':{}}
  def case(initial):return {'task_id':'t2','kind':'rec','initial_decision':initial,'action':{'target_id':'claim:0','source_ids':[]},'arms':{'ordinary':{'decision':initial,'event':{'valid':True,'report':{}}},'linked':{'decision':initial,'event':{'valid':True,'report':{}}}}}
  r=classify_probe_case(case(['1','2','3']),task,{'kind':'rec','targets':['3']})
  self.assertFalse(r['q2']['mechanical_opportunity']);self.assertEqual(r['q2']['outside_top10_in_pool'],[])
  r2=classify_probe_case(case(['1','2']),task,{'kind':'rec','targets':['3']})
  self.assertTrue(r2['q2']['mechanical_opportunity']);self.assertEqual(r2['q2']['outside_top10_in_pool'],['3'])

class ReviewEventClassificationTests(unittest.TestCase):
 def test_all_classes(self):
  q='visit NYPL and Six Flags one day';bridge='visit NYPL and Six Flags in one day trip';far='unrelated weather patterns in Europe'
  def ev(valid=True,patch=None,verdict=None,before='no',error=None):return {'valid':valid,'accepted':None,'error':error,'before':before,'report':{'patch':patch,'verdict':verdict,'rationale':'the current answer decision stays'}}
  self.assertEqual(classify_review_event(ev(valid=False,error='assigned new evidence not used'),bridge,q,'qa')['class'],'invalid_contract')
  self.assertEqual(classify_review_event(ev(patch={'operation':'set_answer','value':'no'},verdict='refute'),bridge,q,'qa')['class'],'same_value_patch')
  self.assertEqual(classify_review_event(ev(patch={'operation':'set_answer','value':'yes'},verdict='refute'),bridge,q,'qa')['class'],'patch_not_accepted')
  self.assertEqual(classify_review_event(ev(verdict='refute'),bridge,q,'qa')['class'],'refute_bridge_no_patch')
  self.assertEqual(classify_review_event(ev(verdict='refute'),far,q,'qa')['class'],'refute_no_bridge_low_overlap')
  self.assertEqual(classify_review_event(ev(verdict='support'),far,q,'qa')['class'],'support_no_patch')
  self.assertEqual(classify_review_event(ev(verdict='unclear'),far,q,'qa')['class'],'unclear_no_patch')
  self.assertTrue(classify_review_event(ev(verdict='refute'),bridge,q,'qa')['bridge']['bridge'])
  self.assertFalse(classify_review_event(ev(verdict='refute'),far,q,'qa')['bridge']['bridge'])

class MainAuditTests(unittest.TestCase):
 def test_rejects_label_fields_in_inputs(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);(root/'cases').mkdir()
   (root/'inputs.json').write_text(json.dumps([{'id':'t1','kind':'qa','prompt':'q','gold':'yes'}]))
   with self.assertRaises(ValueError):main_audit(root)

 def test_counts_classes_and_sample(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);(root/'cases').mkdir()
   (root/'inputs.json').write_text(json.dumps([{'id':'t1','kind':'qa','prompt':'visit nypl six flags','options':['yes','no'],'candidates':[],'source_catalog':{},'metadata':{}}]))
   game={'initial_decision':'no','events':[{'phase':'initial','valid':True},{'phase':'review','valid':True,'accepted':False,'before':'no','report':{'patch':None,'verdict':'refute','target_id':'claim:0','rationale':'keeps the current answer decision'}},{'phase':'review','valid':False,'error':'assigned new evidence not used','before':'no','report':{'patch':{'operation':'set_answer','value':'no'},'verdict':'refute','target_id':'claim:0','rationale':'x'}},{'phase':'review','valid':False,'error':'quote invalid','before':'no','report':{'patch':None,'verdict':'refute','target_id':'claim:0','rationale':'y'}}],'state':{'claims':{'claim:0':{'text':'visit nypl and six flags in a day'}}}}
   (root/'cases'/'case_0.json').write_text(json.dumps({'task_id':'t1','kind':'qa','methods':{'game':game}}))
   r=main_audit(root)
   self.assertEqual(r['cases'],1);self.assertEqual(r['classes']['refute_bridge_no_patch'],1);self.assertEqual(r['classes']['same_value_patch'],1);self.assertEqual(r['classes']['invalid_contract'],1)
   self.assertEqual(len(r['patch_events']),1);self.assertEqual(len(r['sample_30_for_semantic_pass']),1);self.assertFalse(r['label_read'])

if __name__=='__main__':unittest.main()
