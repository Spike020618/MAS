"""Nonzero-temperature DeepSeek-only probe after observed T0 degeneration.

No GLM/Qwen requests. New request distribution, not a retry or efficacy study.
"""
from hashlib import sha256
from pathlib import Path
from experiments.mechanism import probe_signaling_interface as base
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    previous=base.OUT;previous_backends=base.BACKENDS;original=base.request
    base.OUT=previous.with_name('signaling_interface_probe_v3')
    base.BACKENDS=('deepseek',)
    base.OUT.mkdir(parents=True,exist_ok=True)
    freeze(base.OUT/'fixture_amendment.json',{
        'reason':'T0 returned selected probability1 and other entries -9999; test T1 before concluding unavailable',
        'previous_directory':'experiments/results/resource_allocation/signaling_interface_probe_v2',
        'temperature':1.,'fixture':'3+3; A6/B7',
        'scope':'does not establish original full-sequence scoring compatibility or GLM support',
        'wrapper_sha256':sha256(Path(__file__).read_bytes()).hexdigest()})
    def amended(backend,model,slot):
        req=original(backend,model,slot)
        req['temperature']=1.
        req['messages'][1]['content']=('What is 3 + 3?\nA: 6\nB: 7' if slot.startswith('generate') else
            'Question: What is 3 + 3?\nProposed answer: '+('6' if slot=='verify_A' else '7'))
        return req
    base.request=amended
    try:base.run()
    finally:base.request=original;base.OUT=previous;base.BACKENDS=previous_backends


if __name__=='__main__':main()
