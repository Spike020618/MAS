import unittest
from experiments.mechanism.pilot_aeic_deliberation import peer_messages,audit_references,majority,CONDITIONS


def initial():
    return [{'parsed':{'C':'yes','A':f'assumption {i}','E':f'Independent evidence from agent {i} is visible.',
                       'I':'reasoning'}} for i in range(3)]


class AEICDeliberationTests(unittest.TestCase):
    def test_C_only_does_not_leak_peer_fields(self):
        peers=peer_messages(initial(),0,'C')
        self.assertEqual(len(peers),2)
        self.assertTrue(all(set(p)=={'agent_id','C'} for p in peers))

    def test_full_AEIC_has_no_self_and_contains_all_fields(self):
        peers=peer_messages(initial(),1,'AEIC')
        self.assertEqual({p['agent_id'] for p in peers},{'0','2'})
        self.assertTrue(all(set(['A','E','I','C','evidence_id'])<=set(p) for p in peers))

    def test_valid_quote_not_confused_with_factual_truth(self):
        peers=peer_messages(initial(),0,'AEIC')
        row={'parsed':{'peer_evidence_used':[{'evidence_id':'1:E','disposition':'accept',
            'quote':'Independent evidence','reason':'relevant'}]}}
        result=audit_references(row,peers)
        self.assertEqual(len(result['valid']),1)
        self.assertIn('not source support',result['scope'])
        self.assertEqual(len(audit_references(row,peer_messages(initial(),0,'C'))['invalid']),1)

    def test_fabricated_or_paraphrased_quote_rejected(self):
        row={'parsed':{'peer_evidence_used':[{'evidence_id':'1:E','disposition':'accept',
            'quote':'A fabricated lengthy citation','reason':'claim'}]}}
        self.assertEqual(len(audit_references(row,peer_messages(initial(),0,'AEIC'))['invalid']),1)

    def test_orthogonal_cells_and_fixed_majority(self):
        self.assertEqual({(c['budget'],c['visibility']) for c in CONDITIONS.values()},
                         {(0,'C'),(1,'C'),(0,'AEIC'),(1,'AEIC')})
        self.assertEqual(majority(['yes','no','abstain']),'abstain')

    def test_two_stage_execution_and_resume_without_api(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from experiments.mechanism.pilot_aeic_deliberation import execute
        class FakeCaller:
            def __init__(self):self.calls={}
            def call(self,key,messages):
                payload=json.loads(messages[1]['content'])
                report={'C':'yes','A':'assumption','E':'A visible sufficiently long evidence statement.','I':'inference'}
                if '/revision/' in key:
                    peers=payload['peers'];report['peer_evidence_used']=[]
                    if peers and 'evidence_id' in peers[0]:
                        report['peer_evidence_used']=[{'evidence_id':peers[0]['evidence_id'],
                            'disposition':'accept','quote':peers[0]['E'],'reason':'shared claim'}]
                row={'key':key,'parsed':report,'response':json.dumps(report),'usage':{'total_tokens':10},'wall_seconds':.01}
                self.calls[key]=row;return row
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';out=root/'out'
            for path in [source/'banks',source/'snapshots',out/'runs',out/'commitments',out/'initial_rounds']:path.mkdir(parents=True)
            content=b'<html><p>A sufficiently detailed factual paragraph for a generic evidence extraction fixture.</p></html>'
            sid=hashlib.sha256(content).hexdigest();(source/'snapshots'/f'{sid}.html').write_bytes(content)
            bank={'sources':[{'source_hash':sid,'url':'https://example.org/article','title':'Article','text':'old excerpt'}]}
            bank_path=source/'banks'/'one.json';bank_path.write_text(json.dumps(bank))
            protocol={'conditions':CONDITIONS,'bank_hashes':{'one':hashlib.sha256(bank_path.read_bytes()).hexdigest()},
                'costs':{str(i):[0,.75] for i in range(3)},'report_prompts':['initial']*3,'revision_prompt':'revise'}
            case={'question_id':'fixture/one','question':'A fixture?'};caller=FakeCaller()
            result=execute(out,protocol,case,source,caller,'D')
            self.assertEqual(len(caller.calls),6)
            self.assertEqual(list(caller.calls)[3:],[f'fixture/one/D/revision/{i}' for i in range(3)])
            self.assertEqual(result['regrets'],[0,0,0])
            self.assertEqual(sum(len(r['valid']) for r in result['references']),3)
            self.assertEqual(execute(out,protocol,case,source,caller,'D'),result)
            self.assertEqual(len(caller.calls),6)


if __name__=='__main__':unittest.main()
