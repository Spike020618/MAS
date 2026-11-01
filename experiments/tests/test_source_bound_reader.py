import unittest,json
from experiments.coordination_game_v1.source_bound_reader import request,bind
class SourceBoundTests(unittest.TestCase):
 def setUp(self):self.sources=[{'id':'d0','title':'Source','sentences':['Original sentence.','Second fact.']}]
 def test_program_binds_original_text(self):
  value=bind({'answer':'a','status':'supported','steps':[{'relation':'r','source_id':'d0','sentence_indices':[1]}]},self.sources)
  self.assertTrue(value['valid']);self.assertEqual(value['bound_steps'][0]['quotes'],['Second fact.'])
 def test_invalid_visibility_and_indices_rejected(self):
  for sid,indices in [('d1',[0]),('d0',[2]),('d0',[-1]),('d0',[True])]:
   self.assertFalse(bind({'answer':'a','status':'supported','steps':[{'relation':'r','source_id':sid,'sentence_indices':indices}]},self.sources)['valid'])
 def test_outer_schema_required(self):self.assertFalse(bind({'source_id':'d0','sentence_indices':[1]},self.sources)['valid'])
 def test_insufficient_not_forced_to_revise(self):self.assertTrue(bind({'answer':'unknown','status':'insufficient','steps':[]},self.sources)['valid'])
 def test_only_public_projection(self):
  payload=json.loads(request('q',self.sources)['messages'][1]['content']);self.assertEqual(set(payload),{'question','sources'});self.assertNotIn('gold',payload)
if __name__=='__main__':unittest.main()
