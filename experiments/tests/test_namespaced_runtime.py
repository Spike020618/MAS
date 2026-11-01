import unittest
from experiments.namespaced_runtime import NamespacedAPI
class NamespaceTests(unittest.TestCase):
    def test_aliased_policy_methods_do_not_collide(self):
        class API:
            def __init__(self):self.slots={};self.requests=[]
            def call(self,slot,req):
                if slot in self.slots and self.slots[slot]!=req:raise ValueError('immutable slot collision')
                self.slots[slot]=req;self.requests.append(req);return req
        api=API();a={'x':1};b={'x':2}
        self.assertEqual(NamespacedAPI(api,'base').call('0_new_game_0',a),a)
        self.assertEqual(NamespacedAPI(api,'calibrated').call('0_new_game_0',b),b)
        self.assertEqual(len(api.slots),2);self.assertIs(api.requests[0],a)
        self.assertNotIn('method',a)
    def test_unknown_exception_not_retried(self):
        class API:
            def __init__(self):self.calls=0
            def call(self,*args):self.calls+=1;raise RuntimeError('unknown completion')
        api=API()
        with self.assertRaises(RuntimeError):NamespacedAPI(api,'calibrated').call('s',{})
        self.assertEqual(api.calls,1)
    def test_path_namespaces_rejected(self):
        for name in ('','a/b','a\\b'):
            with self.assertRaises(ValueError):NamespacedAPI(None,name)
if __name__=='__main__':unittest.main()
