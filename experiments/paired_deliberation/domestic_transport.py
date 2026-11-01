"""Allowlisted domestic API transport with secret-free receipts and batch caps."""
import json
import os
import time
from pathlib import Path

from experiments.paired_deliberation.budget_v6 import AttemptClient

PROVIDERS = {
    'glm_flash': {'url':'https://open.bigmodel.cn/api/paas/v4/chat/completions',
                  'model':'glm-4-flash-250414','key_env':'MAS_GLM_API_KEY',
                  'input_cny_per_million':0.0,'output_cny_per_million':0.0},
    'qwen_flash': {'url':'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                   'model':'qwen-flash-2025-07-28','key_env':'MAS_DASHSCOPE_API_KEY',
                   'input_cny_per_million':0.15,'output_cny_per_million':1.5},
}


class DomesticTransport:
    def __init__(self, provider, slots, max_attempts=36, max_cny=1.0):
        if provider not in PROVIDERS: raise ValueError('unknown domestic provider')
        self.name,self.config,self.slots=provider,dict(PROVIDERS[provider]),Path(slots)
        self.key=os.environ.get(self.config['key_env'],'')
        self.max_attempts,self.max_cny=max_attempts,max_cny

    def request_upper_cny(self,request):
        return (AttemptClient.upper(request)*self.config['input_cny_per_million']+
                request['max_tokens']*self.config['output_cny_per_million'])/1_000_000

    def reserved(self):
        requests=[]
        for path in self.slots.glob('*.raw.json'):
            requests.append(json.loads(path.read_text())['request'])
        for path in self.slots.glob('*.intent.json'):
            requests.append(json.loads(path.read_text())['request'])
        return len(requests),sum(self.request_upper_cny(r) for r in requests)

    def preflight(self,request):
        if not self.key: raise RuntimeError('configured provider key missing; no request sent')
        if request.get('labels_available') is not False:raise ValueError('label isolation missing')
        if type(request.get('max_tokens')) is not int or not 1<=request['max_tokens']<=1280:
            raise ValueError('screen output cap exceeded')
        count,cost=self.reserved()
        if count>=self.max_attempts or cost+self.request_upper_cny(request)>self.max_cny:
            raise RuntimeError('domestic batch attempt/cost cap reached before dispatch')

    def __call__(self,request):
        # DurableProvider has now created the intent, which counts as reserved.
        count,cost=self.reserved()
        if count>self.max_attempts or cost>self.max_cny:raise RuntimeError('domestic batch cap reached')
        if not self.key:raise RuntimeError('provider key missing')
        body={'model':self.config['model'],'messages':request['messages'],'temperature':0,
              'max_tokens':request['max_tokens'],'response_format':{'type':'json_object'},'stream':False}
        if self.name=='qwen_flash':body['enable_thinking']=False
        import requests
        started=time.monotonic()
        try:
            response=requests.post(self.config['url'],headers={'Authorization':'Bearer '+self.key},
                                   json=body,timeout=(15,120),allow_redirects=False)
        except requests.RequestException:
            # Do not print request objects or authorization. The intent stays.
            raise RuntimeError('transport failed; pending request retained; automatic resend forbidden') from None
        return {'status_code':response.status_code,'body':response.text.replace(self.key,'[redacted]'),
                'elapsed_ms':int((time.monotonic()-started)*1000)}
