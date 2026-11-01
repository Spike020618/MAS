import copy
import unittest
from experiments.bound_verdict_adapter import canonicalize

class VerdictAdapterTests(unittest.TestCase):
    def test_only_declared_alias_converted_without_mutation(self):
        task={'metadata':{'uncertainty_answers':['not_enough_info']}}
        raw={'verdict':'not_enough_info','patch':None,'span_ids':[],'decision_effect':{'choice':'keep'}}
        before=copy.deepcopy(raw);after,ops=canonicalize(task,raw)
        self.assertEqual(after['verdict'],'unclear');self.assertEqual(raw,before)
        self.assertEqual(after['patch'],raw['patch']);self.assertEqual(after['span_ids'],raw['span_ids'])
        self.assertEqual(after['decision_effect'],raw['decision_effect']);self.assertEqual(len(ops),1)
    def test_undeclared_alias_stays_invalid(self):
        raw={'verdict':'not_enough_info'}
        self.assertEqual(canonicalize({'metadata':{}},raw),(raw,[]))
    def test_valid_and_unknown_values_unchanged(self):
        task={'metadata':{'uncertainty_answers':['not_enough_info']}}
        for verdict in ['support','refute','unclear','maybe']:
            raw={'verdict':verdict};self.assertEqual(canonicalize(task,raw),(raw,[]))
    def test_answer_and_patch_never_rewritten(self):
        task={'metadata':{'uncertainty_answers':['not_enough_info']}}
        raw={'verdict':'not_enough_info','decision':'yes','patch':{'operation':'set_answer','value':'no'}}
        after,_=canonicalize(task,raw)
        self.assertEqual(after['decision'],'yes');self.assertEqual(after['patch'],raw['patch'])
