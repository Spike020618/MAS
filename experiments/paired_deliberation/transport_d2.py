"""D2 allowlisted non-thinking transport, receipt-based physical and cost caps."""
import json
import os
import time
from pathlib import Path

from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.client_v5 import parse_raw

ENDPOINTS={
 'deepseek':('https://api.deepseek.com/v1/chat/completions','DEEPSEEK_API_KEY'),
 'glm_flash':('https://open.bigmodel.cn/api/paas/v4/chat/completions','MAS_GLM_API_KEY'),
 'qwen_flash':('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions','MAS_DASHSCOPE_API_KEY'),
 'qwen_plus':('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions','MAS_DASHSCOPE_API_KEY')}


class TransportD2:
    def __init__(self,provider,profile,slots,max_attempts,max_cny,max_prompt=64000,max_output=4096):
        if provider not in ENDPOINTS:raise ValueError('unregistered API origin')
        self.provider,self.profile,self.slots=provider,dict(profile),Path(slots)
        self.url,env=ENDPOINTS[provider];self.key=os.environ.get(env,'')
        self.max_attempts,self.max_cny,self.max_prompt,self.max_output=max_attempts,max_cny,max_prompt,max_output
    def cost(self,request):
        return (AttemptClient.upper(request)*self.profile['input_cny_per_million']+request['max_tokens']*self.profile['output_cny_per_million'])/1_000_000
    def reserved(self):
        # Another collector can replace intent with raw between glob and read.
        # Require a stable, fully readable snapshot; never silently omit a slot.
        def paths():
            return tuple(sorted(p for pattern in ('*.raw.json','*.intent.json') for p in self.slots.glob(pattern)))
        for _ in range(10):
            before=paths()
            try:
                rows=[(p,json.loads(p.read_text())) for p in before]
            except (FileNotFoundError,json.JSONDecodeError):
                time.sleep(.005)
                continue
            if paths()!=before:
                time.sleep(.005)
                continue
            slots={}
            for path,row in rows:
                slot=path.name.removesuffix('.raw.json').removesuffix('.intent.json')
                if slot in slots and slots[slot]['request']!=row['request']:
                    raise RuntimeError('conflicting requests in one budget slot; no dispatch')
                slots[slot]=row
            return len(slots),sum(self.cost(r['request']) for r in slots.values())
        raise RuntimeError('unstable budget ledger; no dispatch; retry collection after writers settle')
    def preflight(self,request):
        if not self.key:raise RuntimeError('configured service key missing; no intent created')
        if self.provider.startswith('qwen_') and 'json' not in ' '.join(m.get('content','') for m in request.get('messages',[])).lower():
            raise ValueError('Qwen JSON-object mode requires JSON in messages; no intent created')
        if request.get('labels_available') is not False or type(request.get('max_tokens')) is not int or not 0<request['max_tokens']<=self.max_output:
            raise ValueError('D2 transport request/output contract')
        if AttemptClient.upper(request)>self.max_prompt:raise ValueError('physical prompt upper exceeds frozen limit')
        count,cost=self.reserved()
        if count>=self.max_attempts or cost+self.cost(request)>self.max_cny+1e-12:raise RuntimeError('physical attempt/currency cap reached')
    def __call__(self,request):
        count,cost=self.reserved()
        if count>self.max_attempts or cost>self.max_cny+1e-12:raise RuntimeError('physical batch cap reached')
        body={'model':self.profile['requested_model'],'messages':request['messages'],'temperature':0,
              'max_tokens':request['max_tokens'],'response_format':{'type':'json_object'},'stream':False}
        if self.provider=='deepseek':body['thinking']={'type':'disabled'}
        if self.provider.startswith('qwen_'):body['enable_thinking']=False
        import requests
        started=time.monotonic()
        try:response=requests.post(self.url,headers={'Authorization':'Bearer '+self.key},json=body,timeout=(15,120),allow_redirects=False)
        except requests.RequestException:raise RuntimeError('transport interrupted; intent retained; automatic resend forbidden') from None
        return {'status_code':response.status_code,'body':response.text.replace(self.key,'[redacted]'),
                'elapsed_ms':int((time.monotonic()-started)*1000)}


def parse_d2_raw(raw,request):
    if raw['status_code']!=200:raise RuntimeError('HTTP failure persisted (%s); batch halted without retry'%raw['status_code'])
    return parse_raw(raw,request)
