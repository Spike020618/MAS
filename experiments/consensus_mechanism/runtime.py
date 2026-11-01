"""Reuse durable raw-first transport with explicit sampling for published baselines."""
import json,os,time
from experiments.paired_deliberation.native_optimization_runtime import Runtime as ExistingRuntime
from experiments.paired_deliberation.transport_d2 import TransportD2
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.transport_d2 import parse_d2_raw
from experiments.consensus_recommendation.common import ROOT,fingerprint

class SamplingTransport(TransportD2):
 def reserved(self):
  for _ in range(10):
   files=tuple(sorted(p for pattern in ('*.raw.json','*.intent.json')for p in self.slots.glob(pattern)))
   try:rows=[(p,json.loads(p.read_text()))for p in files]
   except (FileNotFoundError,json.JSONDecodeError):time.sleep(.005);continue
   if files!=tuple(sorted(p for pattern in ('*.raw.json','*.intent.json')for p in self.slots.glob(pattern))):time.sleep(.005);continue
   slots={}
   for path,row in rows:
    key=path.name.removesuffix('.raw.json').removesuffix('.intent.json')
    if key in slots and slots[key]['request']!=row['request']:raise RuntimeError('conflicting request slot')
    if key not in slots or path.name.endswith('.raw.json'):slots[key]=row
   count=0;spent=0.
   for row in slots.values():
    if row.get('reused_from'):continue
    count+=1
    try:
     usage=json.loads(row['raw']['body'])['usage'];spent+=(usage['prompt_tokens']*self.profile['input_cny_per_million']+usage['completion_tokens']*self.profile['output_cny_per_million'])/1e6
    except (KeyError,TypeError,ValueError):spent+=self.cost(row['request'])
   return count,spent
  raise RuntimeError('unstable budget ledger; stop before dispatch')
 def __call__(self,request):
  count,cost=self.reserved()
  if count>self.max_attempts or cost>self.max_cny+1e-12:raise RuntimeError('batch budget reached')
  temperature=request.get('temperature',0.)
  if not isinstance(temperature,(int,float))or not 0<=temperature<=1:raise ValueError('temperature outside registered range')
  body={'model':self.profile['requested_model'],'messages':request['messages'],'temperature':temperature,'max_tokens':request['max_tokens'],'response_format':{'type':'json_object'},'stream':False}
  if self.provider=='deepseek':body['thinking']={'type':'disabled'}
  if self.provider.startswith('qwen_'):body['enable_thinking']=False
  import requests
  started=time.monotonic()
  try:r=requests.post(self.url,headers={'Authorization':'Bearer '+self.key},json=body,timeout=(15,120),allow_redirects=False)
  except requests.RequestException:raise RuntimeError('transport interrupted; unknown completion retained; no resend')from None
  return {'status_code':r.status_code,'body':r.text.replace(self.key,'[redacted]'),'elapsed_ms':int((time.monotonic()-started)*1000)}

class Runtime(ExistingRuntime):
 def __init__(self,root,plan,provider='qwen_flash'):
  super().__init__(root,plan,cap=plan['physical_cap'],max_cny=plan['max_cny'],provider=provider)
  self.provider.transport=SamplingTransport(provider,self.profile,root/'slots',plan['physical_cap'],plan['max_cny'])

  # A fatal account error escapes per-event RuntimeError fallback after raw persistence.
  from .account_guard import install
  install(self.provider)
