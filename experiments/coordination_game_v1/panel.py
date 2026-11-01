from .schema import digest

def branch_identity(task_id,repeat,branch,request):
    return digest({'task':task_id,'repeat':repeat,'branch':branch,'request':request})

def budget(tasks=12,repeats=2,input_cap=4096,output_cap=768):
    initial=3;checks=9+3+3*2;readouts=1+9+3+3
    calls=tasks*repeats*(initial+checks+readouts)
    return {'tasks':tasks,'repeats':repeats,'initial_per_repeat':initial,'check_per_repeat':checks,'readout_per_repeat':readouts,'requests':calls,
            'input_token_cap':calls*input_cap,'output_token_cap':calls*output_cap,
            'estimated_upper_cny_configured_rates':calls*(input_cap*.15+output_cap*1.5)/1e6,
            'scope':'configured token admission caps; exact prompt preflight required; no retries included; not provider quote'}
