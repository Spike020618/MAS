import json,tempfile,threading,time,unittest,urllib.request,urllib.error
from pathlib import Path
from system.application.catalog import Catalog,parse_turn
from system.application.service import Service
from system.application.engine import CatalogEvidenceEngine,EngineRequest,Client,UnknownCall
from system.application.server import Server

MOVIES=[{'movie_id':str(i),'title':title,'year':year,'genres':genres,'rating_mean':4.,'rating_count':100,'semantic_tags':[{'tag':'space','relevance':.9}],'evidence_refs':{'metadata':'fixture:'+str(i)}}for i,title,year,genres in [(1,'Interstellar',2014,['Sci-Fi','Drama']),(2,'Toy Story',1995,['Animation','Comedy']),(3,'Comedy Story',2020,['Comedy']),(4,'Horror Story',2018,['Horror']),(5,'Space Drama',2016,['Drama'])]]
class AppTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();p=Path(self.tmp.name)/'movies.jsonl';p.write_text(''.join(json.dumps(x)+'\n'for x in MOVIES));self.catalog=Catalog(p);self.s=Service(Path(self.tmp.name)/'app',self.catalog);self.a=self.s.bootstrap();self.b=self.s.bootstrap();self.owner=self.a['profile']['id']
 def tearDown(self):self.s.pool.shutdown();self.tmp.cleanup()
 def finish(self,t):
  for _ in range(150):
   t=self.s.store.get('task',t['id'])
   if t['status']not in('queued','running'):return t
   time.sleep(.02)
  self.fail('task timeout')
 def test_profile_isolation(self):
  session=self.s.session(self.owner)
  with self.assertRaises(KeyError):self.s.store.get('session',session['id'],self.b['profile']['id'])
  self.assertTrue(self.a['profile']['admin']);self.assertFalse(self.b['profile']['admin'])
 def test_soft_preference_not_hard_filter(self):
  result=self.catalog.search('推荐电影',{'preferred_genres':['Sci-Fi']},{})
  self.assertEqual(len(result),5);self.assertEqual(result[0]['id'],'1')
  strict=self.catalog.search('推荐电影',{'required_genres':['Sci-Fi']},{})
  self.assertEqual([x['id']for x in strict],['1'])
 def test_negation_and_followup(self):
  c=parse_turn('不要科幻，只看喜剧',{},{});self.assertIn('sci-fi',c['excluded_genres']);self.assertIn('comedy',c['required_genres'])
  c=parse_turn('2010年以后，换一批',c,{},['1']);self.assertEqual(c['min_year'],2010);self.assertEqual(c['excluded_movie_ids'],['1'])
 def test_task_end_to_end_and_feedback(self):
  t=self.finish(self.s.submit(self.owner,{'query':'科幻','provider':'offline','top_k':2}))
  self.assertEqual(t['status'],'complete');self.assertEqual(t['result']['usage']['calls'],0);self.assertEqual(t['result']['schema'],'mas.engine.result.v1')
  session=self.s.store.get('session',t['session_id']);self.assertEqual(len(session['messages']),2)
  first=t['result']['movies'][0]['id'];self.s.feedback(self.owner,{'movie_id':first,'action':'seen'})
  next=self.finish(self.s.submit(self.owner,{'query':'换一批','session_id':session['id'],'top_k':2}))
  self.assertNotIn(first,[x['id']for x in next['result']['movies']]);self.assertNotIn('receipts',self.s.task(self.owner,t['id']))
 def test_budget_and_method_validation(self):
  for kwargs in ({'max_calls':20},{'max_cny':float('nan')},{'top_k':-1},{'method':'fake'}):
   with self.assertRaises(ValueError):self.s.submit(self.owner,{'query':'电影',**kwargs})
 def test_unknown_cannot_resume(self):
  t=self.finish(self.s.submit(self.owner,{'query':'电影'}));t.update(status='interrupted',receipts={'r':{'status':'intent'}});self.s.store.put('task',t['id'],self.owner,t)
  with self.assertRaises(ValueError):self.s.resume(self.owner,t['id'])
 def test_import_preserves_sources(self):
  with self.assertRaises(ValueError):self.catalog.import_rows([{'movie_id':'8','title':'bad','genres':[]}])
  self.catalog.import_rows([{'movie_id':'8','title':'New','genres':[],'evidence_refs':{'metadata':'test'}}]);self.assertEqual(self.catalog.detail('8')['sources']['metadata'],'test')
 def test_replacement_engine(self):
  class Alternative(CatalogEvidenceEngine):
   def capabilities(self):return {**super().capabilities(),'id':'replacement-v2'}
  self.s.engine=Alternative();t=self.finish(self.s.submit(self.owner,{'query':'电影'}));self.assertEqual(t['result']['engine']['id'],'replacement-v2')
 def test_all_methods_and_zero_budget(self):
  from system.application.engine import METHODS
  for method in METHODS:
   t=self.finish(self.s.submit(self.owner,{'query':'电影','method':method,'max_calls':0}));self.assertEqual(t['status'],'complete');self.assertEqual(t['result']['allocation'],[])
 def test_pair_scope_http(self):
  session=self.s.session(self.owner);pair=self.s.pair(self.owner,session['id']);server=Server(('127.0.0.1',0),self.s);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   def get(path,token):
    return urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:'+str(server.server_port)+path,headers={'Authorization':'Bearer '+token}))
   with get('/api/sessions/'+session['id'],pair['token'])as r:self.assertEqual(r.status,200)
   with self.assertRaises(urllib.error.HTTPError)as cm:get('/api/profile',pair['token'])
   self.assertEqual(cm.exception.code,403)
   with self.assertRaises(urllib.error.HTTPError)as cm:get('/api/sessions/'+session['id'],self.b['token'])
   self.assertEqual(cm.exception.code,404)
  finally:server.shutdown();server.server_close()
 def test_no_fake_rating_or_poster(self):
  d=self.catalog.detail('1');self.assertEqual(d['rating_scale'],5);self.assertIsNone(d['poster']);self.assertEqual(d['rating_source'],'MovieLens')
 def test_unknown_receipt_no_network(self):
  c=Client('qwen_flash',lambda *args:None,lambda *args:{'hash':'different','status':'intent'},lambda:False,3,.1)
  with self.assertRaises(ValueError):c.call({},'slot')

if __name__=='__main__':unittest.main()

class RobustnessTests(unittest.TestCase):
 setUp=AppTests.setUp
 tearDown=AppTests.tearDown
 finish=AppTests.finish
 def test_same_snapshot_comparison(self):
  a=self.finish(self.s.submit(self.owner,{'query':'科幻','method':'fixed','max_calls':1}))
  self.s.feedback(self.owner,{'movie_id':'1','action':'seen'})
  b=self.finish(self.s.submit(self.owner,{'query':'科幻','method':'game','reference_task_id':a['id'],'max_calls':3}))
  self.assertEqual(a['candidates'],b['candidates']);self.assertEqual(a['profile_snapshot'],b['profile_snapshot']);self.assertEqual(b['max_calls'],1);self.assertEqual(b['comparison_group'],a['id'])
 def test_multiple_preference_cues(self):
  c=parse_turn('不喜欢恐怖但喜欢科幻',{},{});self.assertIn('horror',c['excluded_genres']);self.assertIn('sci-fi',c['preferred_genres']);self.assertNotIn('sci-fi',c['excluded_genres'])
 def test_similarity_excludes_anchor(self):
  result=self.catalog.search('和星际穿越类似的电影',{},{});self.assertNotIn('1',[m['id']for m in result]);self.assertTrue(result)
 def test_cancel_and_resume(self):
  class Slow(CatalogEvidenceEngine):
   def run(self,r,emit,checkpoint,cancelled):
    for _ in range(10):
     if cancelled():
      from system.application.engine import Cancelled
      raise Cancelled()
     time.sleep(.015)
    return super().run(r,emit,checkpoint,cancelled)
  self.s.engine=Slow();t=self.s.submit(self.owner,{'query':'科幻'})
  with self.s.lock:
   row=self.s.store.get('task',t['id']);row['cancel']=True;self.s.store.put('task',t['id'],self.owner,row)
  self.assertEqual(self.finish(t)['status'],'cancelled')
  self.assertEqual(self.finish(self.s.resume(self.owner,t['id']))['status'],'complete')
 def test_quote_binding_no_model_copy_required(self):
  from unittest.mock import patch
  candidate=self.catalog.search('科幻',{},{});r=EngineRequest('科幻',{}, {},candidate,provider='qwen_flash',max_calls=1)
  def fake(client,payload,slot):return {'decision':'promote','reason':'资料支持','source_ids':[payload['candidate']['evidence'][0]['id']]}
  with patch.object(Client,'call',fake):out=CatalogEvidenceEngine().run(r,lambda *a:None,lambda *a:None,lambda:False)
  self.assertEqual(len(out['changes']),1);self.assertTrue(out['changes'][0]['citations'][0]['quote'])
 def test_unknown_source_cannot_change_ranking(self):
  from unittest.mock import patch
  r=EngineRequest('科幻',{}, {},self.catalog.search('科幻',{},{}),provider='qwen_flash',max_calls=1)
  with patch.object(Client,'call',return_value={'decision':'promote','reason':'fake','source_ids':['missing']}):out=CatalogEvidenceEngine().run(r,lambda *a:None,lambda *a:None,lambda:False)
  self.assertEqual(out['changes'],[]);self.assertEqual(out['reports'][-1]['decision'],'invalid')
 def test_legacy_research_adapter(self):
  from system.application.legacy_adapter import LegacyAEICAdapter
  self.s.engine=LegacyAEICAdapter();t=self.finish(self.s.submit(self.owner,{'query':'科幻','method':'fixed'}))
  self.assertEqual(t['status'],'complete');self.assertEqual(t['result']['engine']['id'],'legacy-aeic-adapter-v1')
  with self.assertRaises(ValueError):self.s.submit(self.owner,{'query':'科幻','method':'game'})

class BoundaryTests(unittest.TestCase):
 setUp=AppTests.setUp
 tearDown=AppTests.tearDown
 finish=AppTests.finish
 def test_why_uses_saved_evidence_without_new_call(self):
  first=self.finish(self.s.submit(self.owner,{'query':'科幻','top_k':2}))
  second=self.finish(self.s.submit(self.owner,{'query':'为什么推荐第一部','session_id':first['session_id']}))
  self.assertEqual(second['status'],'complete');self.assertEqual(second['result']['usage']['calls'],0)
  self.assertEqual(second['result']['explanation_source'],first['id'])
 def test_result_outside_candidate_pool_is_rejected(self):
  class Bad(CatalogEvidenceEngine):
   def run(self,*args):
    result=super().run(*args);result['movies'][0]['id']='foreign';return result
  self.s.engine=Bad();t=self.finish(self.s.submit(self.owner,{'query':'科幻'}))
  self.assertEqual(t['status'],'failed');self.assertIn('范围外',t['error'])
 def test_catalog_invalid_import_atomic(self):
  count=len(self.catalog.movies)
  with self.assertRaises(ValueError):self.catalog.import_rows([{'movie_id':'new','title':'new','genres':[],'evidence_refs':{'metadata':'x'},'rating_mean':'bad'}])
  self.assertEqual(len(self.catalog.movies),count)
