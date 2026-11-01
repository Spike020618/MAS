from dataclasses import dataclass
@dataclass
class Ledger:
    call_limit:int
    token_limit:int
    reserved_calls:int=0
    reserved_tokens:int=0
    def reserve(self,actions,readout_calls=1,input_cap=4096,output_cap=768):
        if readout_calls!=1 or input_cap<0 or output_cap<1:raise ValueError('invalid readout budget')
        calls=sum(a.calls for a in actions)+readout_calls
        tokens=sum(a.input_tokens+a.output_tokens for a in actions)+readout_calls*(input_cap+output_cap)
        if self.reserved_calls+calls>self.call_limit or self.reserved_tokens+tokens>self.token_limit:raise ValueError('whole round exceeds remaining budget')
        self.reserved_calls+=calls;self.reserved_tokens+=tokens
        return {'calls':calls,'tokens':tokens}
