"""JSON declaration is applied BEFORE joint reservation and durable dispatch."""
from experiments.paired_deliberation.budget_d2 import ChannelClient as BaseChannelClient
from experiments.paired_deliberation.json_contract_d2 import json_request


class ChannelClient(BaseChannelClient):
    def complete(self,slot,request):return super().complete(slot,json_request(request))
    def account_shared(self,slot,request,record):return super().account_shared(slot,json_request(request),record)
