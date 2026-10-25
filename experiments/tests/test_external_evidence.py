import unittest
from experiments.mechanism.pilot_external_evidence import ParagraphParser, quote_provenance


class ExternalEvidenceTests(unittest.TestCase):
    def test_paragraph_extract_and_exact_quote_check(self):
        p=ParagraphParser();p.feed('<p>Myoglobin binds <b>oxygen</b> in muscle tissue.</p>')
        sources=[{'id':'S1','excerpt':p.paragraphs[0]}]
        good=quote_provenance([{'source_id':'S1','quote':'Myoglobin binds oxygen in muscle tissue.'}],sources)
        self.assertTrue(good['all_valid'])
        bad=quote_provenance([{'source_id':'S1','quote':'Myoglobin is the same as blood hemoglobin.'}],sources)
        self.assertFalse(bad['all_valid'])

    def test_empty_or_unknown_sources_never_validate(self):
        self.assertFalse(quote_provenance([],[])['all_valid'])
        self.assertFalse(quote_provenance([{'source_id':'fake','quote':'This is a long fabricated quotation.'}],[])['all_valid'])


if __name__=='__main__':unittest.main()
