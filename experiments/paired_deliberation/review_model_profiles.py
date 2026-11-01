"""Explicit low-cost reviewer profiles; context stays below the 128k price tier.

Plus price checked 2026-10-28 against official Model Studio qwen-plus page.
Non-thinking only; no assumption that higher price guarantees useful reviews.
"""
from experiments.paired_deliberation.preflight_d2 import ACTORS
PROFILES={**ACTORS,'qwen_plus':{'provider':'qwen_plus','requested_model':'qwen-plus-2025-12-01',
    'allowed_returned_models':['qwen-plus-2025-12-01'],'input_cny_per_million':.8,'output_cny_per_million':2.,
    'price_source':'https://help.aliyun.com/zh/model-studio/qwen-plus','price_checked':'2026-10-28',
    'price_tier':'Beijing/global non-thinking, input<=128k; runtime input upper<=64k'}}
