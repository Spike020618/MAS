import itertools,tempfile,unittest
from pathlib import Path
from experiments.consensus_mechanism.allocation import SetAllocator
from experiments.consensus_mechanism.report import report

class EvidenceSetTests(unittest.TestCase):
 def rows(self):
  return [dict(id='a',target_id='claim',source_ids=['a'],gains=[-.1]*3,cost=.01),dict(id='b',target_id='claim',source_ids=['b'],gains=[-.1]*3,cost=.01),dict(id='ab',target_id='claim',source_ids=['a','b'],gains=[.5]*3,cost=.01)]
 def test_joint_interaction_and_ablation(self):
  g=SetAllocator(self.rows());a=g.solve('game');self.assertIn(2,a['assignment']);self.assertGreater(a['potential'],0)
  self.assertEqual(SetAllocator(self.rows(),interactions=False).solve('game')['assignment'],(None,None,None))
 def test_duplicate_not_paid_twice_but_cost_is(self):
  rows=self.rows();rows[0]['gains']=[.4]*3;g=SetAllocator(rows)
  self.assertAlmostEqual(g.potential((0,0,None)),.38)
  self.assertAlmostEqual(SetAllocator(rows,deduplicate=False).potential((0,0,None)),.78)
 def test_shared_money_calls_and_unique_sources(self):
  g=SetAllocator(self.rows(),calls=1,sources=2,money=.01)
  self.assertTrue(g.feasible((2,None,None)));self.assertFalse(g.feasible((0,1,None)))
  self.assertFalse(SetAllocator(self.rows(),money=.005).feasible((2,None,None)))
 def test_exact_marginal_potential_for_every_feasible_deviation(self):
  g=SetAllocator(self.rows())
  for s in itertools.product([None,0,1,2],repeat=3):
   if not g.feasible(s):continue
   for i in range(3):
    for t in g.deviations(s,i):self.assertAlmostEqual(g.utility(t,i)-g.utility(s,i),g.potential(t)-g.potential(s))
  game=g.solve('game');self.assertLessEqual(game['max_feasible_deviation'],1e-9);self.assertLessEqual(game['potential'],g.solve('central')['potential']+1e-9)
 def test_local_equilibrium_can_miss_central_optimum(self):
  rows=self.rows();rows[0]['gains']=[.3]*3;rows[1]['gains']=[.3]*3;rows[2]['gains']=[1.]*3
  # Remove bundled actions from feasible deviations to isolate complementarity.
  g=SetAllocator(rows);base=g.feasible;g.feasible=lambda s:base(s)and all(q!=2 for q in s)
  # Different fixture: negative singles require coordinated entry.
  rows[0]['gains']=[-.1]*3;rows[1]['gains']=[-.1]*3
  g=SetAllocator(rows);base=g.feasible;g.feasible=lambda s:base(s)and all(q!=2 for q in s)
  self.assertEqual(g.solve('game')['potential'],0);self.assertGreater(g.solve('central')['potential'],0)
 def test_report_marks_missing_stages(self):
  with tempfile.TemporaryDirectory()as d:
   result=report(Path(d));self.assertEqual(result['completed_stage_summaries'],[])
if __name__=='__main__':unittest.main()

class LifecycleTests(unittest.TestCase):
 def test_real_stage_orchestration_with_fixture_transport_and_resume(self):
  import contextlib,io,json
  from unittest.mock import patch
  from experiments.tests.test_consensus_mechanism import API
  from experiments.tests.test_protocol_v3 import task
  from experiments.consensus_mechanism import runner
  from experiments.consensus_mechanism.engine import PRIMARY,ABLATIONS
  from experiments.consensus_recommendation.common import create
  with tempfile.TemporaryDirectory()as d,contextlib.redirect_stdout(io.StringIO()):
   root=Path(d);train=[]
   for i in range(6):
    t=task('qa'if i%2==0 else'rec','fit'+str(i));t['metadata']['acquisition_partition']='fit'if i<4 else'calibration';train.append(t)
   test=task('qa','newheld');test['metadata']['acquisition_partition']='diagnostic';create(root/'train_inputs.json',train);create(root/'test_inputs.json',[test])
   labels={t['id']:({'kind':'qa','answer':'yes'}if t['kind']=='qa'else{'kind':'rec','targets':['11']})for t in train+[test]}
   with patch.object(runner,'Runtime',side_effect=lambda r,p,provider:API(r)),patch.object(runner,'audit',return_value={}),patch.object(runner,'labels_for',side_effect=lambda ts,p:{t['id']:labels[t['id']]for t in ts}):
    stats=runner.collect_training(root/'train_inputs.json',root/'train');self.assertTrue(stats['contract_gate_passed']);self.assertEqual(stats['matched_branches'],18)
    runner.fit(root/'train',root/'model');model=root/'model/model.joblib'
    summary=runner.evaluate(root/'test_inputs.json',root/'main',PRIMARY+ABLATIONS,model=model)
    self.assertEqual(summary['methods']['game']['qa_correct'],1)
    self.assertEqual(runner.evaluate(root/'test_inputs.json',root/'main',PRIMARY+ABLATIONS,model=model),summary)
    value=runner.value_diagnostic(root/'test_inputs.json',root/'value',model);self.assertEqual(value['actionable_tasks'],1)
    self.assertEqual(value['observed_interaction']if 'observed_interaction'in value else value['records'][0]['observed_interaction'],-1)
   self.assertNotIn('missing_stages',summary)
 def test_exact_duplicate_material_does_not_create_extra_actions(self):
  import copy
  from experiments.tests.test_protocol_v3 import task
  from experiments.tests.test_consensus_mechanism import API
  from experiments.consensus_mechanism import core as c
  from experiments.consensus_mechanism.engine import initialize
  with tempfile.TemporaryDirectory()as d:
   t=task();s,_,_,_=initialize(t,0,API(Path(d)));before=c.actions(t,s)
   sid=next(k for k in t['source_catalog']if k not in s['visible']);t['source_catalog']['duplicate']=copy.deepcopy(t['source_catalog'][sid]);t['source_catalog']['duplicate']['source_id']='duplicate'
   self.assertEqual(c.actions(t,s),before)

class ContractAndDataTests(unittest.TestCase):
 def test_rec_request_explicit_array_contract_rejects_observed_failures(self):
  import json
  from experiments.tests.test_protocol_v3 import task
  from experiments.consensus_mechanism import core as c
  t=task('rec');s=c.expose(t,c.fresh(t),['s0']);req=c.request(t,s,'initial',0);payload=json.loads(req['messages'][1]['content'])
  self.assertEqual(payload['output_contract']['decision']['minItems'],10)
  for decision in ('0',['0','1','2','3']):
   with self.assertRaises(ValueError):c.validate_report(t,s,{'decision':decision,'claim':'A premise','span_ids':[]},'initial',0)
 def test_reviewer_context_is_actually_transmitted(self):
  import json
  from experiments.tests.test_protocol_v3 import task
  from experiments.consensus_mechanism import core as c
  t=task();s=c.fresh(t);req=c.request(t,s,'review',0,[{'claim':'first'},{'claim':'second'}],{'target_id':'x','source_ids':[]},reviewer=1)
  payload=json.loads(req['messages'][1]['content']);self.assertEqual(payload['reviewer_own_report']['claim'],'second')
 def test_native_paper_inputs_quality_and_full_hotpot_support(self):
  from experiments.consensus_mechanism.data import read
  root=Path('experiments/results/consensus_mechanism_paper_v2')
  if not(root/'input_quality.json').exists():self.skipTest('prepared local paper data unavailable')
  quality=read(root/'input_quality.json');self.assertEqual(quality['sets']['primary']['tasks'],400);self.assertEqual(quality['sets']['primary']['groups'],400)
  manifest=read(root/'benchmark_data/manifest.json');self.assertEqual(manifest['selected_tasks'],458);self.assertEqual(manifest['support_sentence_coverage'],1.)
  tasks=read(root/'benchmark_data/tasks.json');self.assertTrue(all('supporting_facts'not in t and 'answer'not in t for t in tasks))
  self.assertEqual(len(read(root/'data/auxiliary.json')),64)

class MemberValueTests(unittest.TestCase):
 def test_reviewer_assignment_uses_member_specific_values(self):
  rows=[{'id':'a','target_id':'premise','source_ids':['a'],'gains':[.1,.4,.7],'cost':.01}]
  g=SetAllocator(rows);result=g.solve('game')
  self.assertEqual(result['assignment'],(None,None,0));self.assertAlmostEqual(result['potential'],.69)
