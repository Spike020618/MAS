"""Allowlisted cheap provider; instantiated only by an explicitly approved collector."""
import json,os,urllib.request,urllib.error
from experiments.paired_deliberation.domestic_transport import PROVIDERS

# deepseek is served by the transport_d2 stack (not the domestic PROVIDERS dict); mirror its
# endpoint/model here so the coordination line can route heterogeneous members without
# touching the shared domestic module.
_EXTRA={'deepseek':{'url':'https://api.deepseek.com/v1/chat/completions',
                    'model':'deepseek-chat','key_env':'DEEPSEEK_API_KEY'}}
ALLOWED={**PROVIDERS,**_EXTRA}

class Transport:
    def __init__(self,provider='qwen_flash'):
        if provider not in ALLOWED:raise ValueError('provider not allowed')
        self.name=provider;self.config=dict(ALLOWED[provider])
    def __call__(self,request):
        key=os.environ.get(self.config['key_env'])
        if not key:raise RuntimeError('provider credential not configured')
        body={'model':self.config['model'],'messages':request['messages'],'max_tokens':request['max_tokens'],'temperature':0,'response_format':{'type':'json_object'}}
        if self.name=='qwen_flash':body['enable_thinking']=False
        if self.name=='deepseek':body['thinking']={'type':'disabled'}
        req=urllib.request.Request(self.config['url'],data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=120)as response:raw=response.read().decode()
        except urllib.error.HTTPError as e:return {'http_status':e.code,'valid':False,'error':'provider_rejected'}
        except Exception:raise RuntimeError('transport uncertain; intent retained')from None
        value=json.loads(raw)
        try:parsed=json.loads(value['choices'][0]['message']['content'])
        except Exception:parsed=None
        return {'http_status':200,'raw':raw.replace(key,'[redacted]'),'parsed':parsed,'usage':value.get('usage',{})}
