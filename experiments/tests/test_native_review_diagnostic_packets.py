import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.paired_deliberation import native_review_diagnostic_packets as packets

class DiagnosticPacketTests(unittest.TestCase):
    def test_prefix_roundtrip_and_only_known_training_references(self):
        text='Seeker: I saw @12\ncontinued words\nRecommender: Try @99'
        rows=packets.parse_visible(text,{'12':'uri12'},{'uri12':7})
        self.assertEqual(rows,[{'role':'Seeker','text':['I saw @12\ncontinued words'],'movies':['uri12']},{'role':'Recommender','text':['Try @99'],'movies':[]}])
        self.assertNotIn('uri99',str(rows))
    def test_rejects_unscoped_text(self):
        with self.assertRaises(ValueError):packets.parse_visible('I saw a movie',{}, {})
    def test_safe_directory_does_not_link_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fold=root/'fold';fold.mkdir();out=root/'safe'
            (fold/'valid_data.json').write_text('DO NOT READ TARGET SECRET')
            for name in ('train_data.json','entity2id.json','movie_ids.json','dbpedia_subkg.json'):(fold/name).write_text('{}')
            with patch.object(packets,'FOLD',fold):packets.prepare_safe_directory(out)
            self.assertFalse((out/'valid_data.json').is_symlink())
            self.assertEqual(json.loads((out/'valid_data.json').read_text()),[])
            self.assertEqual((fold/'valid_data.json').read_text(),'DO NOT READ TARGET SECRET')
    def test_prefix_loader_refuses_nonempty_validation(self):
        def fake_init(obj,data_dir):obj.valid_raw=[{'future':'forbidden'}]
        with patch.object(packets.NativeEvidence,'__init__',fake_init):
            with self.assertRaisesRegex(ValueError,'no validation dialogue'):packets.PrefixOnlyEvidence('.',[])
