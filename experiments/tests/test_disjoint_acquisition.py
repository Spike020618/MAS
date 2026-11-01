import copy,unittest
from experiments.prepare_disjoint_acquisition import build,keys
class DisjointTests(unittest.TestCase):
 def tasks(self):
  return [{'id':str(i),'kind':'qa','prompt':'public title','metadata':{'native_id':str(i)},'source_catalog':{str(j):{'source_id':str(j),'title':'public title','text':'text '+str(j),'url':'https://example.org/'+str(j),'scope':{'task_id':str(i),'subject':'catalog'}}for j in range(60)}}for i in range(20)]
 def test_no_source_leak_and_no_mutation(self):
  ts=self.tasks();before=copy.deepcopy(ts);out,c=build(ts);self.assertEqual(ts,before);self.assertEqual(len(out),len(ts));seen={}
  for t in out:
   for s in t['source_catalog'].values():
    self.assertEqual(s['scope']['task_id'],t['id']);self.assertFalse(s['current_user_fact_authority'])
    for k in keys(s,'document'):
     self.assertEqual(seen.setdefault(k,c['task_split'][t['id']]),c['task_split'][t['id']])
 def test_one_document_cannot_be_split(self):
  ts=self.tasks()
  for t in ts:
   for s in t['source_catalog'].values():s['url']='https://example.org/shared'
  out,c=build(ts);self.assertIsNone(out);self.assertIn('error',c)
 def test_same_question_always_same_partition(self):
  ts=self.tasks();ts[1]['metadata']['native_id']=ts[0]['metadata']['native_id'];out,c=build(ts)
  self.assertEqual(c['task_split']['0'],c['task_split']['1'])

 def test_query_specific_sources_not_transferred(self):
  ts=self.tasks()
  ts[0]['source_catalog']['private']={'source_id':'private','kind':'dialogue','text':'private current user preference','scope':{'subject':'current_user'}}
  out,c=build(ts)
  self.assertEqual(c['excluded_query_specific_sources'],1)
  self.assertFalse(any(s.get('origin_source_id')=='private' for t in out for s in t['source_catalog'].values()))
