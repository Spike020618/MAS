"""Integration replay using a saved paid case; skipped without local artifacts."""
import copy,json,unittest
from experiments.paired_deliberation.native_consensus_experiment import RESULT
from experiments.paired_deliberation.native_consensus_replay import verify_case
class SavedNativeReplayTests(unittest.TestCase):
    def setUp(self):
        path=RESULT/'cases/case_0.json'
        if not path.exists():self.skipTest('native paid case not staged')
        self.case=json.loads(path.read_text());self.row=json.loads((RESULT/'inputs.json').read_text())[0]
    def test_saved_allocations_and_bounded_updates_replay(self):
        audit=verify_case(RESULT,0,self.row,self.case)
        self.assertGreater(audit['valid_review_events_replayed'],0)
    def test_modified_source_or_aggregate_cannot_pass_replay(self):
        broken=copy.deepcopy(self.case);bound=broken['methods']['fixed_review']['events'][0]['bound_selection']
        bound['bound_sources'][0]['text']='Fabricated source text'
        with self.assertRaises(AssertionError):verify_case(RESULT,0,self.row,broken)
        broken=copy.deepcopy(self.case);ranking=broken['methods']['independent3']['ranking'];ranking[0],ranking[1]=ranking[1],ranking[0]
        with self.assertRaises(AssertionError):verify_case(RESULT,0,self.row,broken)
if __name__=='__main__':unittest.main()
