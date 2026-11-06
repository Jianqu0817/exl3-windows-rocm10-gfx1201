"""Microbenchmark: one real EXL3 MLP matmul through its production paths at various m.

Loads the main model, then times (sync-to-sync):
  - bc.run_alloc  (m <= 144: exl3_gemm dispatch -> multirow GEMV at m<=32, coop mgemm above)
  - reconstruct_hgemm (m > 144 production path)
at m = 1, 4, 8, 32, 33, 64, 133, 144, 160, 256, 512, 1024 on
  a gate/up projection (5120 -> intermediate) and the down projection.

Diagnostic only.
"""
import argparse
import json
import os
import time

os.environ.setdefault('EXL3_GEMV_WMMA', '1')
os.environ.setdefault('EXL3_GEMV_MAX_M', '32')
import torch
from exllamav3 import model_init


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_draft_model_args=False)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = {}
    with torch.inference_mode():
        model, config, cache, tok = model_init.init(args)[:4]
        torch.cuda.synchronize()

        # sample blocks: an MLP-heavy transformer block
        blocks = [m for m in model.modules if type(m).__name__ == 'TransformerBlock']
        blk = blocks[8]
        gate = blk.mlp.gates[0].inner      # LinearEXL3 inner (5120 -> interm)
        down = blk.mlp.downs[0].inner      # LinearEXL3 inner (interm -> 5120)
        print(f'gate: {gate.in_features} -> {gate.out_features}, K={gate.K}, mul1={gate.mul1}, mcg={gate.mcg}')
        print(f'down: {down.in_features} -> {down.out_features}, K={down.K}')
        attn_blk = next((b for b in blocks if hasattr(b, 'attn') and hasattr(b.attn, 'q_proj')), None)
        q = attn_blk.attn.q_proj.inner
        print(f'q:    {q.in_features} -> {q.out_features}, K={q.K}')

        for name, inner in (('gate_up', gate), ('down', down), ('q', q)):
            K_dim, N_dim = inner.in_features, inner.out_features
            for m in (1, 4, 8, 32, 33, 64, 133, 144, 160, 256, 512, 1024):
                x = (torch.randn(1, m, K_dim, dtype=torch.float16, device='cuda') * 0.1)
                if m <= 144:
                    dt = bench(lambda: inner.bc.run_alloc(x, inner.out_features, False))
                else:
                    dt = bench(lambda: inner.reconstruct_hgemm(x, torch.half))
                out.setdefault(name, {})[m] = round(dt, 4)
                print(f'{name:8s} m={m:5d}: {dt:8.4f} ms  ({N_dim*m/dt/1e6:8.1f} Mtok-out/s)', flush=True)

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('DONE', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
