"""Bound the benefit of reconstructing once for 30 token chunks; no model executor changes."""
import json
from pathlib import Path
import statistics
import sys

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests/upstream'))
from testlib.exl3 import generator,rand_linear


@torch.inference_mode()
def main():
    rows=[]
    for k,n in [(5120,17408),(17408,5120)]:
        weights=rand_linear(k,n,3,generator(k),'cuda')
        inner=LinearEXL3(None,k,n,**weights,mul1=torch.tensor(1))
        inner._fused_reconstruct=True
        x=torch.randn(1024,k,device='cuda',dtype=torch.half)*.1
        w=torch.empty(k,n,device='cuda',dtype=torch.half)
        def reconstruct():
            ext.reconstruct_had_slice(w,weights['trellis'],weights['suh'],weights['svh'],3,False,True,0)
        def gemm():
            y=torch.empty(1024,n,device='cuda',dtype=torch.half)
            ext.hgemm_recon(x,w,y)
            return y
        reconstruct()
        ref=inner.reconstruct_hgemm(x,torch.half)
        assert torch.equal(ref,gemm())
        for _ in range(3):
            inner.reconstruct_hgemm(x,torch.half)
            gemm()
        times={0:[],1:[]}
        for repeat in range(3):
            for reuse in ((0,1) if repeat%2==0 else (1,0)):
                torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                if reuse:
                    reconstruct()
                for _ in range(30):
                    y=gemm() if reuse else inner.reconstruct_hgemm(x,torch.half)
                end.record();end.synchronize()
                times[reuse].append(start.elapsed_time(end))
                assert torch.equal(ref,y)
        row=dict(k=k,n=n,chunk=1024,chunks=30,one_weight_mib=w.numel()*2/2**20,
                 reconstruct_each_ms=times[0],reuse_weight_ms=times[1],
                 speedup=statistics.median(times[0])/statistics.median(times[1]))
        rows.append(row);print(json.dumps(row),flush=True)
    result={'scope':'single synthetic projection, same input repeated; excludes executor/state/staging cost',
            'status':'PASS','cases':rows}
    (ROOT/'logs/prefill-weight-reuse.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
