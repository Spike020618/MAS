"""One logical budget across actor/verifier services, failures and shared calls."""
import copy
import json
import math
from pathlib import Path

from experiments.paired_deliberation.budget_v6 import AttemptClient, BudgetExceeded

ACTOR_KINDS={'initial_d2','initial_central_d2','patch_d2','whole_review_d2','select_d2','ranking_d2'}
VERIFIER_KINDS={'atomic_interpretation','atomic_source_relations','utterance_witness_d2'}


class JointBudget:
    def __init__(self,limits,profiles):
        if set(profiles)!={'actor','verifier'}:raise ValueError('actor and public verifier profiles required')
        self.limits,self.profiles=copy.deepcopy(limits),copy.deepcopy(profiles)
        for key in ('max_calls','per_call_prompt_tokens','per_call_completion_tokens','total_prompt_tokens','total_completion_tokens','max_evidence_checks'):
            if type(self.limits.get(key)) is not int or self.limits[key]<0:raise ValueError('explicit nonnegative integer budget required')
        if not isinstance(self.limits.get('max_estimated_cny'),(int,float)) or not math.isfinite(self.limits['max_estimated_cny']) or self.limits['max_estimated_cny']<0:
            raise ValueError('explicit finite currency ceiling required')
        for profile in self.profiles.values():
            if not isinstance(profile.get('service_id'),str) or not profile['service_id']:raise ValueError('frozen service identity required')
            models=profile.get('allowed_returned_models')
            if not isinstance(models,list) or not models or any(not isinstance(m,str) or not m for m in models):raise ValueError('returned model allowlist required')
            for key in ('input_cny_per_million','output_cny_per_million'):
                if type(profile.get(key)) not in (int,float) or not math.isfinite(profile[key]) or profile[key]<0:raise ValueError('nonnegative verified unit price required')
        self.records=[];self.prompt_reserved=0;self.output_reserved=0;self.cost_reserved=0.0;self.halted=False

    def _price(self,channel,prompt,output):
        p=self.profiles[channel]
        return (prompt*p['input_cny_per_million']+output*p['output_cny_per_million'])/1_000_000

    def fits(self,plan):
        if self.halted:return False
        prompt=output=0;cost=0.0
        for item in plan:
            channel,p,o=item['channel'],item['prompt_upper'],item['completion_cap']
            if channel not in self.profiles or type(p) is not int or type(o) is not int or p<0 or o<=0:return False
            if p>self.limits['per_call_prompt_tokens'] or o>self.limits['per_call_completion_tokens']:return False
            prompt+=p;output+=o;cost+=self._price(channel,p,o)
        return (len(self.records)+len(plan)<=self.limits['max_calls'] and
                self.prompt_reserved+prompt<=self.limits['total_prompt_tokens'] and
                self.output_reserved+output<=self.limits['total_completion_tokens'] and
                self.cost_reserved+cost<=self.limits['max_estimated_cny']+1e-12)

    def reserve(self,channel,slot,request,shared=False):
        allowed=ACTOR_KINDS if channel=='actor' else VERIFIER_KINDS if channel=='verifier' else set()
        if request.get('kind') not in allowed or request.get('labels_available') is not False:raise ValueError('service channel/request authority mismatch')
        if shared and request['kind'] not in {'initial_d2','initial_central_d2'}:raise ValueError('only frozen initial calls can be shared')
        if any(r['slot']==slot for r in self.records):raise ValueError('duplicate logical slot')
        item={'channel':channel,'prompt_upper':AttemptClient.upper(request),'completion_cap':request.get('max_tokens')}
        if not self.fits([item]):raise BudgetExceeded('joint actor/verifier budget exceeded before dispatch')
        self.prompt_reserved+=item['prompt_upper'];self.output_reserved+=item['completion_cap']
        cost=self._price(channel,item['prompt_upper'],item['completion_cap']);self.cost_reserved+=cost
        row={'slot':slot,'channel':channel,'kind':request['kind'],'shared':shared,'status':'attempted',
             'prompt_upper':item['prompt_upper'],'completion_reserved':item['completion_cap'],'cost_reserved_cny':cost,
             'usage':None,'model':None,'elapsed_ms':None}
        self.records.append(row);return row

    def finish(self,row,record):
        usage=record.get('usage',{})
        if (any(type(usage.get(k)) is not int or usage[k]<0 for k in ('prompt_tokens','completion_tokens')) or
            usage['prompt_tokens']>row['prompt_upper'] or usage['completion_tokens']>row['completion_reserved']):
            row.update(status='invalid_usage');self.halted=True
            raise RuntimeError('provider usage missing/invalid or exceeded reservation; dispatch halted')
        row.update(usage=copy.deepcopy(usage),model=record.get('model'),elapsed_ms=record.get('elapsed_ms'))
        if row['model'] not in self.profiles[row['channel']]['allowed_returned_models']:
            row['status']='model_mismatch';self.halted=True
            raise RuntimeError('returned model outside frozen channel allowlist; dispatch halted')
        previous={r['model'] for r in self.records if r is not row and r['channel']==row['channel'] and r['status']=='success'}
        if previous and previous!={row['model']}:
            row['status']='model_changed';self.halted=True
            raise RuntimeError('returned model changed within frozen service channel; dispatch halted')
        row['status']='success'

    def recover_failure(self,row,provider,exc):
        row.update(status='failed',error=type(exc).__name__+': '+str(exc))
        root=getattr(provider,'root',None)
        path=Path(root)/(row['slot']+'.raw.json') if root is not None else None
        if path is not None and path.exists():
            try:
                raw=json.loads(path.read_text())['raw'];body=json.loads(raw['body']);usage=body['usage']
                if all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')):
                    row['usage']={k:usage[k] for k in ('prompt_tokens','completion_tokens')}
                row.update(model=body.get('model'),elapsed_ms=raw.get('elapsed_ms'))
                if row['model'] and row['model'] not in self.profiles[row['channel']]['allowed_returned_models']:self.halted=True
                if row['usage'] and (row['usage']['prompt_tokens']>row['prompt_upper'] or row['usage']['completion_tokens']>row['completion_reserved']):self.halted=True
            except (ValueError,KeyError,TypeError):pass
        if isinstance(exc,RuntimeError):self.halted=True

    def summary(self):
        def totals(rows):
            complete=all(r['usage'] is not None for r in rows)
            prompt=sum(r['usage']['prompt_tokens'] for r in rows if r['usage'] is not None)
            output=sum(r['usage']['completion_tokens'] for r in rows if r['usage'] is not None)
            cost=sum(self._price(r['channel'],r['usage']['prompt_tokens'],r['usage']['completion_tokens']) for r in rows if r['usage'] is not None)
            return {'attempts':len(rows),'successes':sum(r['status']=='success' for r in rows),'usage_complete':complete,
                    'prompt_tokens':{'known':prompt,'total':prompt if complete else None},
                    'completion_tokens':{'known':output,'total':output if complete else None},
                    'estimated_list_price_cny':{'known':cost,'total':cost if complete else None},
                    'returned_models':sorted({r['model'] for r in rows if isinstance(r['model'],str)})}
        return {**totals(self.records),'by_channel':{c:totals([r for r in self.records if r['channel']==c]) for c in self.profiles},
                'reserved_prompt_upper':self.prompt_reserved,'reserved_output':self.output_reserved,
                'reserved_cost_cny':self.cost_reserved,'actual_currency_cost':None,'dispatch_halted':self.halted,
                'cost_basis':'logical full list price incl shared initial; physical batch receipts audited separately',
                'records':copy.deepcopy(self.records)}


class ChannelClient:
    def __init__(self,ledger,channel,provider):
        if channel not in ledger.profiles:raise ValueError('unknown budget channel')
        self.ledger,self.channel,self.provider=ledger,channel,provider
    def complete(self,slot,request):
        row=self.ledger.reserve(self.channel,slot,request)
        try:record=self.provider.complete(slot,request)
        except Exception as exc:
            self.ledger.recover_failure(row,self.provider,exc);raise
        self.ledger.finish(row,record);return record
    def account_shared(self,slot,request,record):
        self.ledger.finish(self.ledger.reserve(self.channel,slot,request,shared=True),record)
    def summary(self):return self.ledger.summary()
