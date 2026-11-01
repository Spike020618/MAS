"""Immutable request receipts and fail-closed resumption; no implicit retries."""
import json,os,threading
from pathlib import Path
from .schema import digest

class UnknownRequest(RuntimeError):pass

def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True));os.replace(temp,path)

class DurableRuntime:
    def __init__(self,root,plan,transport=None,run=False,approved_cny=0):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.slots=self.root/'slots';self.slots.mkdir(exist_ok=True)
        self.plan=plan;self.transport=transport;self.run=run;self.approved_cny=approved_cny;self.lock=threading.RLock()
        p=self.root/'plan.json'
        if p.exists()and json.loads(p.read_text())!=plan:raise ValueError('frozen plan mismatch')
        if not p.exists():atomic(p,plan)
    def estimate(self,request):
        size=len(json.dumps(request['messages'],ensure_ascii=False).encode())
        if size>self.plan.get('input_byte_cap',4096):raise ValueError('input admission cap exceeded')
        out=request.get('max_tokens',768)
        if type(out)is not int or not 1<=out<=self.plan.get('output_token_cap',768):raise ValueError('output cap exceeded')
        return (size*.15+out*1.5)/1e6
    def path(self,slot):return self.slots/(digest(slot)+'.json')
    def reserve_batch(self,requests):
        with self.lock:
            additions=[]
            for slot,request in requests:
                p=self.path(slot);cost=self.estimate(request)
                if p.exists():
                    r=json.loads(p.read_text())
                    if r['request_hash']!=digest(request):raise ValueError('slot request mismatch')
                else:additions.append((p,{'slot':slot,'request_hash':digest(request),'request':request,'status':'reserved','reserved_cny':cost}))
            old=[json.loads(p.read_text())for p in self.slots.glob('*.json')]
            if len(old)+len(additions)>self.plan['max_requests']:raise ValueError('request cap exceeded')
            if sum(r['reserved_cny']for r in old)+sum(r['reserved_cny']for _,r in additions)>min(self.approved_cny,self.plan['max_cny']):raise ValueError('approved cost cap exceeded')
            for p,r in additions:atomic(p,r)
    def call(self,slot,request):
        with self.lock:
            p=self.path(slot)
            if p.exists():
                r=json.loads(p.read_text())
                if r['request_hash']!=digest(request):raise ValueError('slot request mismatch')
                if r['status']=='complete':return r['response']
                if r['status']=='rejected':return None
                if r['status']=='intent':raise UnknownRequest('unknown completion; no automatic resend')
            if not self.run or self.transport is None:raise RuntimeError('explicit run and transport required')
            self.reserve_batch([(slot,request)]);r=json.loads(p.read_text());r['status']='intent';atomic(p,r)
            response=self.transport(request)
            r.update(status='complete',response=response);atomic(p,r);return response
    def seal(self,expected_slots):
        receipts=[]
        for slot in expected_slots:
            p=self.path(slot)
            if not p.exists():raise ValueError('missing slot')
            r=json.loads(p.read_text())
            if r['status']not in('complete','rejected'):raise ValueError('unresolved slot')
            receipts.append({'slot':slot,'hash':digest(r)})
        result={'plan_hash':digest(self.plan),'receipts':receipts}
        p=self.root/'generation_complete.json'
        if p.exists()and json.loads(p.read_text())!=result:raise ValueError('seal mismatch')
        atomic(p,result);return result
