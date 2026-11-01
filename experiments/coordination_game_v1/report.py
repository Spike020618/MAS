from .schema import Action
from .solver import solve
from .value import FrozenValues
from .baselines import fixed

def counterexamples():
    actions=tuple(Action(m+k,m,'relation',(k,))for m in ('i','j')for k in ('A','B'))
    value=FrozenValues({a.action_id:(1 if a.source_ids==('A',)else 4)for a in actions})
    f=fixed(actions,2);g=solve(actions,value,2)
    # Members have identical source valuations, yet action policies differ.
    first_only=0;whole=1
    return {'exchangeable_members':{'fixed_value':value(f),'game_value':g['value'],'central_value':g['central_value'],'all_allocators_equivalent':False},
            'joint_execution':{'v_empty':0,'v_A':0,'v_B':0,'v_AB':whole,'execute_first_then_stop_at_zero':first_only,'masking_requires_discard_or_stop':True},
            'empirical_claim':'synthetic logical counterexamples, not evidence of natural coordination gain'}
