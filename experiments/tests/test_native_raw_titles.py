import unittest
from experiments.paired_deliberation.native_raw_titles import extract_title_metadata,normalized_title
class RawTitleMetadataTests(unittest.TestCase):
    def test_only_allowed_train_identity_fields_are_used(self):
        class Guarded(dict):
            def get(self,key,default=None):
                if key not in ('conversationId','movieMentions'):raise AssertionError('prediction/label payload read')
                return super().get(key,default)
        rows=[Guarded(conversationId='train',movieMentions={'7':'Title (2000)'},messages='SECRET',initiatorQuestions='SECRET'),
              Guarded(conversationId='query',movieMentions={'8':'Future movie'},messages='SECRET')]
        titles,origin,counts=extract_title_metadata(rows,{'train'})
        self.assertEqual(set(titles),{'7'});self.assertEqual(origin['7'],{'train'});self.assertEqual(counts['official_train_overlap_conversations'],1)
    def test_empty_mentions_skipped_and_titles_not_fuzzy_merged(self):
        titles,_,counts=extract_title_metadata([{'conversationId':'a','movieMentions':[]},{'conversationId':'b','movieMentions':{'1':'A & B (2000)','2':'A and B (2000)'}}],{'a','b'})
        self.assertEqual(counts['empty_or_non_dict_movie_mentions'],1)
        self.assertNotEqual(normalized_title(titles['1'].most_common()[0][0]),normalized_title(titles['2'].most_common()[0][0]))
if __name__=='__main__':unittest.main()
