import copy,unittest
from experiments.fit_e2_action_calibration import split_commit,action_features
class E2CalibrationTests(unittest.TestCase):
 def task(self,id,kind='qa'):
  return {'id':id,'kind':kind,'prompt':'verify blue sky','candidates':[],'metadata':{'native_id':id},'source_catalog':{'s':{'title':'blue','kind':'web','text':'hidden body','retrieval_cosine':.2}}}
 def test_source_content_does_not_enter_features(self):
  t=self.task('q');a=action_features(t,'s','yes');t['source_catalog']['s']['text']='X'*len('hidden body');self.assertEqual(a,action_features(t,'s','yes'))
 def test_grouped_rec_turns_cannot_cross_splits(self):
  tasks=[self.task('123:1','rec'),self.task('123:9','rec')]+[self.task('q'+str(i))for i in range(20)]
  c=split_commit(tasks);self.assertEqual(c['task_groups']['123:1'],c['task_groups']['123:9']);self.assertEqual(c,split_commit(list(reversed(tasks))))
 def test_outcome_fields_do_not_enter_features(self):
  t=self.task('q');a=action_features(t,'s','no');t['reward']=100;t['answer']='yes';t['accepted']=True;self.assertEqual(a,action_features(t,'s','no'))
