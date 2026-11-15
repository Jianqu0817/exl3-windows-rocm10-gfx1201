"""Check selective dispatch against existing paths for single and bundled linears."""
import json
import os
from pathlib import Path
import sys

import torch
from exllamav3.ext import exllamav3_ext as ext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/upstream'))
from testlib.exl3 import generator, rand_linear


@torch.inference_mode()
def main():
    checks = []
    os.environ['EXL3_ROCM_GEMV_TILES'] = '1'
    for rows, bits, cb in [(4,3,'mul1'), (8,3,'mul1'), (4,4,'mul1'), (4,3,'mcg')]:
        weights = [rand_linear(512,512,bits,generator(10+i),'cuda') for i in range(2)]
        x = torch.randn(1,rows,512,device='cuda',dtype=torch.half) * .1
        ptr = lambda key: torch.tensor([w[key].data_ptr() for w in weights],device='cuda',dtype=torch.int64)
        tr, su, sv = ptr('trellis'), ptr('suh'), ptr('svh')
        for dtype in (torch.half,torch.float):
            for bundled in (False,True):
                y = torch.empty((2,rows,512) if bundled else (rows,512),device='cuda',dtype=dtype)
                xh = torch.empty((2,rows,512) if bundled else (rows,512),device='cuda',dtype=torch.half)
                def run(wmma, selective):
                    os.environ['EXL3_GEMV_WMMA'] = str(wmma)
                    os.environ['EXL3_GEMV_M4_FDOT2'] = str(selective)
                    if bundled:
                        ext.exl3_mgemm(x,tr,y,su,xh,sv,None,None,bits,-1,cb=='mcg',cb=='mul1',-1,-1,0,1,None,None)
                    else:
                        w = weights[0]
                        ext.exl3_gemm(x[0],w['trellis'],y,w['suh'],xh,w['svh'],0,cb=='mcg',cb=='mul1',0)
                    torch.cuda.synchronize()
                    assert torch.isfinite(y).all()
                    return y.clone()
                actual = run(1,1)
                reference = run(0,0) if (rows,bits,cb)==(4,3,'mul1') else run(1,0)
                assert torch.equal(actual,reference), (rows,bits,cb,str(dtype),bundled)
                checks.append(dict(rows=rows,bits=bits,codebook=cb,dtype=str(dtype),bundled=bundled))
    result = {'status':'PASS','bit_exact_checks':len(checks),'cases':checks}
    (ROOT/'logs/m4-dispatch-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
