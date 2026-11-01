import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.paired_deliberation.transport_d2 import TransportD2
from experiments.paired_deliberation.preflight_d2 import ACTORS

class StableLedgerTests(unittest.TestCase):
    def request(self):return {'messages':[{'role':'user','content':'ranking'}],'max_tokens':192,'labels_available':False}
    def test_intent_transition_between_glob_and_read_is_retried_without_losing_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);intent=folder/'a.intent.json';row={'request':self.request()};intent.write_text(json.dumps(row))
            transport=TransportD2('qwen_flash',ACTORS['qwen_flash'],folder,10,1)
            original=Path.read_text;triggered=[]
            def transition(path,*args,**kwargs):
                if path==intent and not triggered:
                    triggered.append(True);intent.unlink();(folder/'a.raw.json').write_text(json.dumps(row));raise FileNotFoundError('concurrent completed writer')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',transition):count,cost=transport.reserved()
            self.assertEqual(count,1);self.assertEqual(cost,transport.cost(self.request()))
    def test_raw_and_its_not_yet_removed_intent_count_one_physical_slot(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);row={'request':self.request()}
            for suffix in ('raw','intent'):(folder/('a.'+suffix+'.json')).write_text(json.dumps(row))
            t=TransportD2('qwen_flash',ACTORS['qwen_flash'],folder,10,1)
            self.assertEqual(t.reserved(),(1,t.cost(self.request())))
    def test_partial_or_conflicting_ledger_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);p=folder/'a.intent.json';p.write_text('{');t=TransportD2('qwen_flash',ACTORS['qwen_flash'],folder,10,1)
            with self.assertRaisesRegex(RuntimeError,'unstable budget ledger'):t.reserved()
            p.write_text(json.dumps({'request':self.request()}));other=self.request();other['max_tokens']=200
            (folder/'a.raw.json').write_text(json.dumps({'request':other}))
            with self.assertRaisesRegex(RuntimeError,'conflicting requests'):t.reserved()
if __name__=='__main__':unittest.main()
