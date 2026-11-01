import unittest,tempfile,json
from pathlib import Path
from experiments.coordination_game_v1.prepare_data import prepare,load_public
class P1Data(unittest.TestCase):
 def fixture(self,d):
  import pyarrow as pa,pyarrow.parquet as pq
  path=Path(d)/'public.parquet';pq.write_table(pa.Table.from_pylist([{'id':f'{i:024x}','question':f'Which person {i}?','context':{'title':['A','B','C','D','E','F'],'sentences':[['a'],['b'],['c'],['d'],['e'],['f']]},'answer':'SECRET'+str(i),'supporting_facts':{'title':['A'],'sent_id':[0]}}for i in range(15)]),path);return path
 def test_public_projection_and_vault(self):
  with tempfile.TemporaryDirectory()as d:
   source=self.fixture(d);self.assertNotIn('SECRET',str(load_public(source)))
   output=Path(d)/'out';m=prepare(output,source,results=Path(d)/'empty');tasks=json.loads((output/'tasks.json').read_text());self.assertEqual(len(tasks),12);self.assertNotIn('SECRET',str(tasks));self.assertFalse(m['fresh_test_certified']);self.assertEqual(len(json.loads((output/'private/scoring.json').read_text())),12)
   with self.assertRaises(ValueError):prepare(output,source)
 def test_label_changes_do_not_change_selection(self):
  with tempfile.TemporaryDirectory()as d:
   import pyarrow as pa,pyarrow.parquet as pq
   source=self.fixture(d);a=prepare(Path(d)/'a',source,results=Path(d)/'empty');rows=pq.read_table(source).to_pylist()
   for r in rows:r['answer']='CHANGED'
   pq.write_table(pa.Table.from_pylist(rows),source);b=prepare(Path(d)/'b',source,results=Path(d)/'empty');self.assertEqual(a['task_ids'],b['task_ids']);self.assertEqual(a['tasks_sha256'],b['tasks_sha256'])
