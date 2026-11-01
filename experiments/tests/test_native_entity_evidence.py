import copy,json,unittest
from pathlib import Path
from experiments.paired_deliberation.native_entity_evidence import NativeEvidence,constraints,infer_links,canonical,merge_dialog,OUT

class NativeIdentityTests(unittest.TestCase):
    def test_singletons_and_unique_remaining_identity_only(self):
        a='<http://dbpedia.org/resource/Film_A>';b='<http://dbpedia.org/resource/Film_B>'
        links,conflict=infer_links([({'1'},{a}),({'1','2'},{a,b})])
        self.assertEqual(links,{'1':a,'2':b});self.assertFalse(conflict)
        ambiguous,_=infer_links([({'1','2'},{a,b})]);self.assertFalse(ambiguous)
    def test_conflicting_annotations_remain_unresolved(self):
        links,conflict=infer_links([({'1'},{'A'}),({'1'},{'B'})])
        self.assertNotIn('1',links);self.assertEqual(conflict['1'],['A','B'])
    def test_missing_annotations_cannot_force_unlinked_reference_into_known_entity(self):
        u={'text':['@1','and','@2'],'movies':['A']}
        self.assertEqual(constraints([u],{'A':1}),[])
    def test_render_does_not_guess_unknown_movie_or_change_native_ids(self):
        uri='<http://dbpedia.org/resource/Film_A_(2000_film)>'
        self.assertEqual(canonical('like @7 but @8 unknown',{'7':uri},{uri:432}),
                         'like Film A (2000 film) [native_movie_id=432; redial_ref=@7] but @8 unknown')
    def test_same_role_merge_preserves_exact_text_and_duplicate_mentions(self):
        dialog=[{'role':'Seeker','text':['a'],'movies':['m']},{'role':'Seeker','text':['b'],'movies':['m']}]
        merged=merge_dialog(dialog);self.assertEqual(merged[0]['text'],['a','b']);self.assertEqual(merged[0]['movies'],['m','m'])

class PrefixIsolationTests(unittest.TestCase):
    def test_current_and_future_response_payload_is_never_inspected(self):
        class Future(dict):
            def __getitem__(self,key):
                if key in ('text','movies'):raise AssertionError('future payload leak')
                return super().__getitem__(key)
        obj=object.__new__(NativeEvidence)
        obj.conversations={'x':{'dialog':[{'role':'Seeker','text':['request'],'movies':[]},Future(role='Recommender')]}}
        visible=obj.visible_prefix('x:1');self.assertEqual(visible,[{'role':'Seeker','text':['request'],'movies':[]}])
    def test_prefix_unknown_case_or_turn_is_rejected(self):
        obj=object.__new__(NativeEvidence);obj.conversations={'x':{'dialog':[]}}
        with self.assertRaises(ValueError):obj.visible_prefix('unknown:1')
        with self.assertRaises(ValueError):obj.visible_prefix('x:3')

class SavedNativeEvidenceTests(unittest.TestCase):
    def test_shared_materials_preserve_case_identity_and_other_user_scope(self):
        p=OUT/'fixed32.inputs.json'
        if not p.exists():self.skipTest('shared-material artifact not staged')
        rows=json.loads(p.read_text())
        for row in rows:
            for neighbor in row['training_neighbors']:
                self.assertNotEqual(neighbor['origin_case_id'].rsplit(':',1)[0],row['id'].rsplit(':',1)[0])
                card=row['sources'][neighbor['source_id']]
                self.assertEqual(card['allowed_use'],'prediction_basis');self.assertFalse(card['current_user_fact_authority'])
            for source in row['sources'].values():
                if source['kind']!='dialogue' or source.get('role')!='seeker':self.assertFalse(source['current_user_fact_authority'])
                self.assertFalse(source['semantic_entailment_certified'])
            self.assertEqual(set(row['candidate_evidence']),{m['movie_id'] for m in row['candidates']})
            self.assertNotIn('targets',row)

if __name__=='__main__':unittest.main()
