"""New synthetic fixture after sandbox DNS interruption; never replays old key.

The old unresolved attempt is retained, not cleared or billed as zero. This is
a distinct arithmetic fixture (3+3 instead of2+2), not an automatic retry of
the ambiguous prior request. Run with approved network access.
"""
from hashlib import sha256
import json
from pathlib import Path
from experiments.mechanism import probe_signaling_interface as base
from experiments.mechanism.run_upgrade_pilot import freeze, read_rows


def main():
    previous=base.OUT
    base.OUT=previous.with_name('signaling_interface_probe_v2')
    base.OUT.mkdir(parents=True,exist_ok=True)
    freeze(base.OUT/'fixture_amendment.json',{
        'reason':'sandbox DNS interruption; separate new fixture, no original key resubmission',
        'previous_directory':str(previous.relative_to(base.ROOT)),
        'previous_attempts':len(read_rows(previous/'attempts.jsonl')),
        'previous_responses':len(read_rows(previous/'responses.jsonl')),
        'previous_cost':'unresolved transport attempt retained; not assumed zero',
        'new_fixture':'3 + 3; option A6, option B7; synthetic interface only',
        'wrapper_sha256':sha256(Path(__file__).read_bytes()).hexdigest()})
    original=base.request
    def amended(backend,model,slot):
        req=original(backend,model,slot)
        req['messages'][1]['content']=('What is 3 + 3?\nA: 6\nB: 7' if slot.startswith('generate') else
            'Question: What is 3 + 3?\nProposed answer: '+('6' if slot=='verify_A' else '7'))
        return req
    base.request=amended
    try:base.run()
    finally:
        base.request=original;base.OUT=previous


if __name__=='__main__':main()
