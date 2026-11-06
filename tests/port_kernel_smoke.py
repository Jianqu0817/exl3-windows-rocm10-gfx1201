"""Small synthetic WMMA check; no checkpoint or model is loaded."""
import json
import os
from pathlib import Path
import sys

import torch
from exllamav3.ext import exllamav3_ext as ext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream_tests"))
from testlib.exl3 import CODEBOOKS, codebook_flags, generator, linear_ref, rand_linear

checks = 0
worst = 0.0
os.environ["EXL3_GEMV_WMMA"] = "1"
os.environ["EXL3_GEMV_MAX_M"] = "32"
with torch.inference_mode():
    for bits in (2, 3, 4):
        for codebook in CODEBOOKS:
            gen = generator(bits)
            weights = rand_linear(512, 512, bits, gen, "cuda:0")
            x = (torch.randn((32, 512), generator=gen) * 0.5).half().cuda()
            reference = linear_ref(x, **weights, K=bits, codebook=codebook).float()
            for unroll in (1, 2, 4):
                os.environ["EXL3_GEMV_WMMA_U"] = str(unroll)
                for dtype in (torch.half, torch.float):
                    full = None
                    for rows in (32, 2, 3, 4, 8, 9, 16, 17):
                        a = x[:rows].contiguous()
                        out = torch.empty((rows, 512), dtype=dtype, device=a.device)
                        ext.exl3_gemm(a, weights["trellis"], out, weights["suh"], torch.empty_like(a),
                                     weights["svh"], 0, *codebook_flags(codebook), 0)
                        torch.cuda.synchronize()
                        ref = reference[:rows]
                        error = ((out.float() - ref).square().mean().sqrt() /
                                 ref.square().mean().sqrt().clamp_min(1e-6)).item()
                        assert error < 0.02, (bits, codebook, unroll, dtype, rows, error)
                        if full is None:
                            full = out.clone()
                        else:
                            assert torch.equal(out, full[:rows]), ("row invariance", bits, codebook, unroll, dtype, rows)
                        worst = max(worst, error)
                        checks += 1
result = {"status": "PASS", "checks": checks, "worst_relative_rms": worst,
          "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
          "extension": ext.__file__, "torch": torch.__version__}
(ROOT / "logs/port-kernel-smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result), flush=True)
