import unittest
from experiments.paired_deliberation.native_external_genres import StrictNativeGenres,native_key,external_key,normalize_title
class NativeExternalGenreTests(unittest.TestCase):
    def test_explicit_year_required_and_article_normalization_is_structural(self):
        self.assertEqual(native_key('<http://dbpedia.org/resource/The_Thing_(1982_film)>')[0],external_key('Thing, The (1982)'))
        self.assertIsNone(native_key('<http://dbpedia.org/resource/The_Thing_(film)>')[0]);self.assertIsNone(native_key('75962')[0])
        self.assertNotEqual(normalize_title('A & B'),normalize_title('A and B'))
    def test_native_universe_not_filtered_and_genres_are_metadata_only(self):
        entities={'<http://dbpedia.org/resource/It_(2017_film)>':5,'<http://dbpedia.org/resource/Sinister_(film)>':7}
        linker=StrictNativeGenres(entities,[5,7],[{'movieId':'99','title':'It (2017)','genres':'Horror'}])
        self.assertEqual(linker.ids,[5,7]);self.assertEqual(linker.lookup(5)['native_movie_id'],'5');self.assertEqual(linker.lookup(5)['genres'],['Horror'])
        self.assertFalse(linker.lookup(7)['linked']);self.assertEqual(linker.lookup(7)['reason'],'no_explicit_simple_uri_year')
    def test_external_or_native_ambiguity_and_missing_genres_never_link(self):
        entities={'<http://dbpedia.org/resource/A_(2000_film)>':1,'<http://dbpedia.org/resource/A_(2000)>':2}
        row={'movieId':'3','title':'A (2000)','genres':'Drama'}
        self.assertFalse(StrictNativeGenres(entities,[1,2],[row]).lookup(1)['linked'])
        one={next(iter(entities)):1}
        self.assertEqual(StrictNativeGenres(one,[1],[row,{**row,'movieId':'4'}]).lookup(1)['reason'],'ambiguous_external_title_year')
        self.assertEqual(StrictNativeGenres(one,[1],[{**row,'genres':'(no genres listed)'}]).lookup(1)['reason'],'genre_missing')
    def test_no_fuzzy_title_or_year_tolerance(self):
        e={'<http://dbpedia.org/resource/Film_(2000_film)>':1}
        rows=[{'movieId':'8','title':'Film (2001)','genres':'Drama'},{'movieId':'9','title':'The Film (2000)','genres':'Drama'}]
        self.assertEqual(StrictNativeGenres(e,[1],rows).lookup(1)['reason'],'no_exact_external_title_year')
if __name__=='__main__':unittest.main()
