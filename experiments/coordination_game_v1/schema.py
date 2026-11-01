from dataclasses import dataclass,field,asdict
import hashlib,json

def digest(value):
    if hasattr(value,'__dataclass_fields__'):value=asdict(value)
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

@dataclass(frozen=True)
class MemberState:
    member_id:str
    observed_sources:tuple[str,...]=()
    local_reports:tuple[str,...]=()
    history_hash:str=''

@dataclass(frozen=True)
class CheckTask:
    target_id:str
    relation:str
    scope:str
    provenance:tuple[str,...]
    trigger:str='unresolved'

@dataclass(frozen=True)
class Action:
    action_id:str
    member_id:str
    target_id:str
    source_ids:tuple[str,...]
    operation:str='verify_relation'
    calls:int=1
    input_tokens:int=4096
    output_tokens:int=768
    def __post_init__(self):
        if any(not isinstance(x,str)or not x for x in (self.action_id,self.member_id,self.target_id)):raise ValueError('nonempty action identity required')
        if not isinstance(self.source_ids,tuple)or any(not isinstance(x,str)for x in self.source_ids):raise ValueError('immutable source tuple required')
        if type(self.calls)is not int or self.calls<1 or type(self.input_tokens)is not int or self.input_tokens<0 or type(self.output_tokens)is not int or self.output_tokens<1:raise ValueError('invalid resource cost')

@dataclass(frozen=True)
class RoundPlan:
    state_hash:str
    assignments:tuple[Action,...]
    predicted_value:float
    readout_calls:int=1

@dataclass(frozen=True)
class Review:
    action_id:str
    observations:tuple[str,...]
    citation_ids:tuple[str,...]
    valid:bool=True
    error:str=''
