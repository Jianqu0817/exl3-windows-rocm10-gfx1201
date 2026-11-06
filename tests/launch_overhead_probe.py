"""Quantify Windows kernel-submission overhead and torch.profiler availability.

Part 1: microbench back-to-back tiny kernel launches (the per-layer small ops that
dominate launch count in a verify round) versus a real GEMV, measuring host-side
submission cost per launch and end-to-end latency.
Part 2: guarded torch.profiler test (profiler chain was fatal under WSL dxg; native
Windows ROCm 10.1 availability unknown). If available, print top kernels.

Diagnostic only. No model load required.
"""
import json
import time
import torch

dev = torch.device('cuda:0')
torch.cuda.init()
print('GPU', torch.cuda.get_device_name(0), flush=True)

results = {}

def bench(label, fn, iters=2000, sync_every=0):
    # warmup
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        fn()
        if sync_every and (i + 1) % sync_every == 0:
            torch.cuda.synchronize()
    if not sync_every:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    results[label] = round(dt * 1e6, 2)
    print(f'{label:44s} {dt*1e6:9.2f} us/op', flush=True)
    return dt

# --- Part 1: submission overhead ---
a16 = torch.randn(1, 4, 5120, dtype=torch.float16, device=dev)
b16 = torch.randn(1, 4, 5120, dtype=torch.float16, device=dev)
c32 = torch.randn(1, 4, 5120, dtype=torch.float32, device=dev)
d32 = torch.randn(1, 4, 5120, dtype=torch.float32, device=dev)
w = torch.randn(5120, 1536, dtype=torch.float16, device=dev) * 0.01

bench('tiny add f16 (residual-like), submit-only', lambda: a16.add_(b16))
bench('tiny clamp_ f16, submit-only', lambda: a16.clamp_(-65504, 65504))
bench('tiny cast f32->f16, submit-only', lambda: c32.half())
bench('cast+clamp as two ops, submit-only', lambda: c32.half().clamp_(-65504, 65504))
bench('rms-like mul f32, submit-only', lambda: d32.mul_(0.5))
bench('GEMV 4x5120 @ 5120x1536 f16, submit-only', lambda: torch.matmul(a16, w))
bench('tiny add f16, sync every op (round trip)', lambda: a16.add_(b16), iters=500, sync_every=1)

# launch-latency estimate: enqueue N ops, measure time until GPU actually finishes
# (approximates pipelined throughput when host is the bottleneck)
for n in (100, 1000):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        a16.add_(b16)
    t_submit = time.perf_counter() - t0
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0
    print(f'n={n:5d} adds: submit {t_submit*1e6/n:8.2f} us/op, total(incl drain) {t_total*1e6/n:8.2f} us/op', flush=True)
    results[f'drain_n{n}_submit_us'] = round(t_submit * 1e6 / n, 2)
    results[f'drain_n{n}_total_us'] = round(t_total * 1e6 / n, 2)

# --- Part 2: torch.profiler availability ---
prof_ok = False
try:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(100):
            a16.add_(b16)
        torch.cuda.synchronize()
    evs = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA or 'add' in e.key.lower()]
    names = [(e.key, e.count, getattr(e, 'device_time_total', 0)) for e in prof.key_averages()][:12]
    print('PROFILER_OK', json.dumps([str(n) for n in names]), flush=True)
    prof_ok = True
    results['profiler'] = 'ok'
except BaseException as e:
    print('PROFILER_UNAVAILABLE', repr(e)[:300], flush=True)
    results['profiler'] = f'unavailable: {type(e).__name__}: {str(e)[:200]}'

with open(__file__.replace('.py', '.json').replace('tests', r'logs'), 'w') as f:
    json.dump(results, f, indent=2)
print('DONE', json.dumps(results), flush=True)
