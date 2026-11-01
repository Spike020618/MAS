"""Explicit wire-format declaration required by JSON-object provider APIs."""
import copy


def json_request(request):
    result=copy.deepcopy(request)
    if not any('json' in message['content'].casefold() for message in result['messages']):
        result['messages'][0]['content']+=' Return one JSON object only.'
    return result


class JSONContractClient:
    def __init__(self,client):self.client=client
    def __getattr__(self,name):return getattr(self.client,name)
    def complete(self,slot,request):return self.client.complete(slot,json_request(request))
