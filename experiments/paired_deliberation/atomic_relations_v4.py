"""Candidate polarity-specific witnesses; scope relevance remains model judged."""
import copy

from experiments.paired_deliberation.atomic_evidence import (
    dialogue_sources, parser_request, validate_atom, relation_request as legacy_request,
    reduce_relations as legacy_reduce)
from experiments.paired_deliberation.verified_state import VerificationError


def relation_request(atom, sources):
    import json
    request = legacy_request(atom, sources)
    payload = json.loads(request['messages'][1]['content'])
    payload['schema']['relations'] = [{'source_id': 'one supplied ID',
        'relation': 'supports | refutes | both | irrelevant | unclear',
        'support_quote': 'exact matching-polarity proposition, otherwise empty',
        'refute_quote': 'exact opposite-polarity proposition, otherwise empty'}]
    request['messages'][1]['content'] = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    request['messages'][0]['content'] += (
        ' Use support_quote and refute_quote instead of quote. Extract the smallest complete proposition '
        'that retains its subject, predicate, object, negation and temporal modifiers. '
        'A source may contain several objects: a negative preference about Comedy is IRRELEVANT to '
        'Mystery, even in the same sentence. Match subject, predicate, object and time BEFORE polarity. '
        'supports requires only support_quote; refutes requires only refute_quote; both requires two '
        'distinct complete propositions at the SAME target scope and opposite polarities. Never cite '
        'one positive proposition as evidence for both. irrelevant/unclear require both quotes empty. '
        'Exact quotation is necessary but does not establish scope matching by itself.')
    return request


def reduce_relations(atom, sources, result):
    if not isinstance(result, dict) or set(result) != {'relations', 'reason', 'usage'}:
        raise VerificationError('polarity witness response schema')
    if not isinstance(result['relations'], list): raise VerificationError('polarity witness rows')
    converted, witnesses = [], []
    for row in result['relations']:
        if not isinstance(row, dict) or set(row) != {'source_id', 'relation', 'support_quote', 'refute_quote'}:
            raise VerificationError('polarity witness row schema')
        sid, relation = row['source_id'], row['relation']
        positive, negative = row['support_quote'], row['refute_quote']
        if not isinstance(positive, str) or not isinstance(negative, str):
            raise VerificationError('polarity witnesses must be strings')
        if not isinstance(sid, str) or sid not in sources: raise VerificationError('unknown witness source')
        required = {'supports': (True, False), 'refutes': (False, True), 'both': (True, True),
                    'irrelevant': (False, False), 'unclear': (False, False)}
        if not isinstance(relation, str) or relation not in required or (bool(positive), bool(negative)) != required[relation]:
            raise VerificationError('missing or unexpected polarity witness')
        if positive and negative and (positive in negative or negative in positive):
            raise VerificationError('opposite polarities need distinct non-nested proposition witnesses')
        for quote in (positive, negative):
            if quote and quote not in sources[sid]['text']: raise VerificationError('fabricated polarity witness')
        converted.append({'source_id': sid, 'relation': relation, 'quote': positive or negative})
        witnesses.append(copy.deepcopy(row))
    reduced = legacy_reduce(atom, sources, {**result, 'relations': converted})
    reduced['evidence_spans'] = [{'source_id': r['source_id'], 'quote': r[key]}
        for r in witnesses for key in ('support_quote', 'refute_quote') if r[key]]
    reduced.update(polarity_witness_rows=witnesses, relation_contract='polarity_witness_v4_candidate')
    return reduced


class AtomicEvidencePipeline:
    def __init__(self, sources, client):
        self.sources, self.client, self.last_atom = dialogue_sources(sources), client, None

    def verify(self, claim, prefix):
        original = copy.deepcopy(claim)
        self.last_atom = None
        parsed = self.client.complete(prefix + '_parse', parser_request(original, self.sources))
        atom = validate_atom(parsed['report'])
        self.last_atom = copy.deepcopy(atom)
        if atom['scope'] == 'unsupported':
            return {'verdict': 'insufficient', 'kind': 'unsupported_scope', 'reason': atom['reason'],
                    'atom': copy.deepcopy(atom), 'source_ids': list(self.sources), 'evidence_spans': [],
                    'usage': copy.deepcopy(parsed['usage']), 'original_claim': original,
                    'certification_attempted': False}
        judged = self.client.complete(prefix + '_sources', relation_request(atom, self.sources))
        result = reduce_relations(atom, self.sources, {**judged['report'], 'usage': judged['usage']})
        result.update(original_claim=original, certification_attempted=True, parser_usage=copy.deepcopy(parsed['usage']))
        return result
