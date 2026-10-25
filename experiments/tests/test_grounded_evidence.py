import unittest
from experiments.mechanism.pilot_grounded_evidence import EvidenceBlocks,accept_revision
from experiments.core.mechanism.source_checks import parse_ranked_kg_rows,compare_entities,check_tea_comparison


class GroundedEvidenceTests(unittest.TestCase):
    def test_tables_preserve_headers_and_rows(self):
        parser=EvidenceBlocks()
        parser.feed('<table><tr><th>Country and annual consumption per capita</th></tr>'
                    '<tr><td>23 Canada 0.51 kg (1.12 lb)</td></tr></table>')
        self.assertTrue(any('Canada 0.51 kg' in x['text'] and 'annual' in x['text'] for x in parser.blocks))

    def test_ambiguity_veto_and_quote_requirement(self):
        source={'id':'s','excerpt':'The character carved only the initial letter of his name.'}
        candidate={'schema_valid':True,'ambiguity':True,'coverage':'sufficient','draft_contradicted':True,
                   'citations':[{'source_id':'s','quote':source['excerpt']}]}
        self.assertFalse(accept_revision(candidate,[source])[0])
        candidate['ambiguity']=False
        self.assertTrue(accept_revision(candidate,[source])[0])
        candidate['citations'][0]['quote']='A fabricated source quote that does not exist.'
        self.assertFalse(accept_revision(candidate,[source])[0])

    def test_numeric_check_requires_entities_and_same_units(self):
        rows=parse_ranked_kg_rows('Rank Country | 23 Canada 0.51 kg (1.12 lb)\nRank Country | 37 United States 0.23 kg (0.50 lb)')
        result=compare_entities(rows,'United States',['Canada'])
        self.assertTrue(result['all_greater'])
        self.assertEqual(compare_entities(rows,'United States',['Ireland'])['status'],'insufficient')

    def test_conflicting_rows_rejected(self):
        with self.assertRaises(ValueError):
            parse_ranked_kg_rows('1 Canada 0.51 kg\n2 Canada 0.90 kg')

    def test_typed_checker_rejects_partial_country_coverage(self):
        source={'title':'List of countries by tea consumption per capita','url':'https://example.org',
                'excerpt':'23 Canada 0.51 kg\n37 United States 0.23 kg'}
        question='Question: Which countries consume more tea on average than Americans?\nProposed answer: Canada and Ireland consume more tea on average than Americans.'
        self.assertEqual(check_tea_comparison(question,[source])['status'],'insufficient')
        source['excerpt']+='\n2 Ireland 2.36 kg'
        self.assertEqual(check_tea_comparison(question,[source])['stance'],'yes')
        question=question.replace('Canada and Ireland','Canada, and Ireland')
        self.assertEqual(check_tea_comparison(question,[source])['stance'],'yes')


if __name__=='__main__':unittest.main()
