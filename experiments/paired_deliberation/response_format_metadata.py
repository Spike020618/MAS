"""Offline candidate adapter for exact transport-format metadata, not task repairs."""
import copy
import json


def project(record, request):
    template=json.loads(request['messages'][1]['content']).get('schema')
    report=record['report']
    if (isinstance(template,dict) and 'schema' not in template and isinstance(report,dict)
            and set(report)==set(template)|{'schema'} and report['schema']=='json_object'):
        result=copy.deepcopy(record);del result['report']['schema']
        return result,True
    return record,False
