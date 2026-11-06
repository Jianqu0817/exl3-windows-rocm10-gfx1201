"""Numerically checked, bounded Windows sweep at the actual 30K verify geometry."""
import json
import os
from pathlib import Path
import statistics
import sys

import torch
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode, paged_attn_triton_prefill

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream_tests'))
from testlib.attention import rand_packed_cache

keys = ['EXL3_DEC_Q4W', 'EXL3_DEC_GROUP', 'EXL3_DEC_WARPS', 'EXL3_DEC_BLOCK_N', 'EXL3_DEC_STAGES', 'EXL3_DEC_SPLIT_MULT']
configs = [
    ('recipe', [1, 1, 4, 16, 2, None]),
    ('packed-default', [0, 0, None, None, None, None]),
    ('packed-group', [0, 1, 4, 16, 1, None]),
    ('q4w-stage1', [1, 1, 4, 16, 1, None]),
    ('q4w-n32', [1, 1, 4, 32, 1, None]),
    ('q4w-n32-stage2', [1, 1, 4, 32, 2, None]),
    ('q4w-w8', [1, 1, 8, 16, 2, None]),
    ('q4w-split1', [1, 1, 4, 16, 2, 1]),
    ('q4w-split2', [1, 1, 4, 16, 2, 2]),
    ('q4w-split8', [1, 1, 4, 16, 2, 8]),
    ('q4w-split16', [1, 1, 4, 16, 2, 16]),
]
torch.manual_seed(42)
with torch.inference_mode():
    kc, vc, qc = rand_packed_cache(128, 256, 4, 256, 4, 'cuda:0')
    q = torch.randn((1, 4, 24, 256), dtype=torch.half, device='cuda:0')
    common = dict(q=q, k=None, v=None, k_cache=kc, v_cache=vc,
        block_table=torch.arange(128, device='cuda:0', dtype=torch.int32).view(1, 128),
        cache_seqlens=torch.tensor([29996], device='cuda:0', dtype=torch.int32),
        causal=True, qc=qc, max_kv_len=30000, pre_appended_len=4, n_kv_heads_override=4)
    reference = paged_attn_triton_prefill(**common)
    results = []
    for name, values in configs:
        for key, value in zip(keys, values):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        actual = paged_attn_triton_decode(**common)
        torch.testing.assert_close(actual, reference, rtol=0, atol=2e-3)
        for _ in range(10):
            paged_attn_triton_decode(**common)
        times = []
        for _ in range(5):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(50):
                paged_attn_triton_decode(**common)
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) * 1000 / 50)
        row = dict(name=name, microseconds=statistics.median(times), samples=times,
                   max_absolute_error=(actual-reference).abs().max().item(),
                   environment={k:os.environ.get(k) for k in keys})
        results.append(row)
        (ROOT/'logs/attention-tuning.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
print('BEST', json.dumps(min(results,key=lambda r:r['microseconds'])),flush=True)
