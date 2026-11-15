"""Bounded graph replay and reconstruction checks, without loading a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/upstream'))
from testlib.exl3 import rand_linear, generator


def bench(fn, count=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(count):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / count


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=['graph', 'prefill', 'had'], required=True)
    args = parser.parse_args()
    results = {'mode': args.mode, 'graphs': os.environ['EXL3_GRAPHS'], 'cases': []}
    if args.mode == 'graph':
        for k, n in [(512, 512), (5120, 17408)]:
            weights = rand_linear(k, n, 3, generator(k), 'cuda')
            inner = LinearEXL3(None, k, n, **weights, mul1=torch.tensor(1))
            up = LinearEXL3(None, k, n, **rand_linear(k, n, 3, generator(k+1), 'cuda'), mul1=torch.tensor(1))
            down = LinearEXL3(None, n, k, **rand_linear(n, k, 3, generator(k+2), 'cuda'), mul1=torch.tensor(1))
            scratch = lambda *shape: torch.empty(shape, device='cuda', dtype=torch.half)
            mlp = ext.BC_GatedMLP(scratch(2,32,k), scratch(2,32,n), scratch(1,32,n),
                                  scratch(1,32,n), None, None, None, 0, False, False,
                                  True, False, False, inner.bc, up.bc, down.bc, 0.)
            for rows in [1, 4, 8]:
                xs = [torch.randn(rows, k, device='cuda', dtype=torch.half) * .1 for _ in range(3)]
                ys = [scratch(rows, k) for _ in range(3)]
                for j in range(12):
                    x = xs[j % 3]
                    y = ys[j % 3]
                    mlp.run_bszN(x, y)
                    g = inner.bc.run_alloc(x, n, False)
                    u = up.bc.run_alloc(x, n, False)
                    a = torch.empty_like(g)
                    ext.silu_mul(g, u, a, 0.)
                    ref = down.bc.run_alloc(a, k, False)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(y, ref, rtol=0, atol=0)
                ms = bench(lambda: mlp.run_bszN(xs[0], ys[0]), 200)
                free0 = torch.cuda.mem_get_info()[0]
                start = time.perf_counter()
                for j in range(2000):
                    mlp.run_bszN(xs[j % 3], ys[j % 3])
                    if j % 100 == 0:
                        torch.cuda.synchronize()
                torch.cuda.synchronize()
                row = {'k': k, 'n': n, 'm': rows, 'ms': ms,
                       'replay_ms': (time.perf_counter()-start)*1000/2000,
                       'free_delta_mib': (torch.cuda.mem_get_info()[0]-free0)/2**20}
                results['cases'].append(row)
                print(json.dumps(row), flush=True)
    elif args.mode == 'had':
        for bits in [1, 2, 3, 4, 5, 6, 7, 8, 1.5, 2.5, 3.5]:
            for cb in ([2] if bits != int(bits) else [0, 1, 2]):
                w = rand_linear(256, 384, bits, generator(int(bits*10)), 'cuda')
                for off, cols in [(0, 384), (128, 128)]:
                    out = torch.empty(256, cols, device='cuda', dtype=torch.half)
                    ext.reconstruct_had_slice(out, w['trellis'], w['suh'], w['svh'][off:],
                                              bits, cb == 1, cb == 2, off)
                    digest = hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()
                    results['cases'].append({'bits':bits, 'cb':cb, 'offset':off, 'sha256':digest})
                batch = torch.empty(2,256,384,device='cuda',dtype=torch.half)
                ptrs = lambda tensor: torch.tensor([tensor.data_ptr()]*2,device='cuda',dtype=torch.int64)
                ext.reconstruct_had_batch(batch, ptrs(w['trellis']), ptrs(w['suh']), ptrs(w['svh']),
                                          bits, cb == 1, cb == 2)
                torch.cuda.synchronize()
                assert torch.equal(batch[0], batch[1])
                results['cases'].append({'bits':bits,'cb':cb,'batch':2,
                    'sha256':hashlib.sha256(batch.cpu().numpy().tobytes()).hexdigest()})
        for k, n in [(5120,17408),(17408,5120),(5120,6144)]:
            w = rand_linear(k, n, 3, generator(k), 'cuda')
            out = torch.empty(k, n, device='cuda', dtype=torch.half)
            ms = bench(lambda: ext.reconstruct_had_slice(out, w['trellis'], w['suh'],
                        w['svh'], 3, False, True, 0), 50)
            row = {'k':k,'n':n,'ms':ms}
            results['cases'].append(row)
            print(json.dumps(row), flush=True)
    elif args.mode == 'prefill':
        for k, n in [(5120, 17408), (17408, 5120), (5120, 6144)]:
            weights = rand_linear(k, n, 3, generator(k), 'cuda')
            inner = LinearEXL3(None, k, n, **weights, mul1=torch.tensor(1))
            for m in [133, 1024, 2048]:
                x = torch.randn(m, k, device='cuda', dtype=torch.half) * .1
                inner._fused_reconstruct = False
                ref = inner.reconstruct_hgemm(x, torch.half)
                unfused = bench(lambda: inner.reconstruct_hgemm(x, torch.half))
                inner._fused_reconstruct = True
                result = inner.reconstruct_hgemm(x, torch.half)
                fused = bench(lambda: inner.reconstruct_hgemm(x, torch.half))
                row = {'k': k, 'n': n, 'm': m, 'unfused_ms': unfused, 'fused_ms': fused,
                       'fused_active': m >= 1024,
                       'rel_rms': ((result.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt()).item()}
                results['cases'].append(row)
                print(json.dumps(row), flush=True)
    results['status'] = 'PASS'
    Path(args.out).write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
