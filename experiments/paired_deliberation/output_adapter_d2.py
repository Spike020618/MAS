"""Frozen D2 representation policy: no selection, citation or semantic repairs."""
import copy
import json

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.screen_output_projection import project

VERSION = 'd2_representation_v1'


def adapt(record, request):
    request = copy.deepcopy(request)
    payload = json.loads(request['messages'][1]['content'])
    template = payload.get('schema')
    result = copy.deepcopy(record)
    operations = []
    if isinstance(template, dict) and isinstance(result.get('report'), dict):
        obj = result['report']
        if 'schema' not in template and set(obj) == set(template) | {'schema'} and obj.get('schema') == template:
            del obj['schema']; operations.append('remove_exact_template_echo')
    if request['kind'] == 'ranking_d2': request['kind'] = 'ranking_v6'
    result, more = project(result, request, payload.get('evidence'))
    operations += more
    # Report rankings receive the identical lossless ID representation rule.
    if isinstance(result['report'], dict) and 'ranking' in result['report'] and isinstance(payload.get('evidence'), dict):
        shadow = copy.deepcopy(request)
        shadow['kind'] = 'ranking_v6'
        shadow['messages'][1]['content'] = json.dumps({'schema': {'ranking': []}})
        ranking, more = project({'report': {'ranking': result['report']['ranking']}}, shadow, payload['evidence'])
        result['report']['ranking'] = ranking['report']['ranking']; operations += more
    event = {'adapter': VERSION, 'raw_report_hash': fingerprint(record['report']),
             'projected_report_hash': fingerprint(result['report']), 'operations': operations}
    return result, event


class D2Client:
    def __init__(self, client): self.client, self.adaptations = client, []
    def __getattr__(self, name): return getattr(self.client, name)
    def complete(self, slot, request):
        record, event = adapt(self.client.complete(slot, request), request)
        self.adaptations.append({'slot': slot, **event})
        return record
