"""Hot m=4 / K=3 mul1 probe of existing RDNA paths, without loading a model.

Run via runtime.py script tests/smallm_existing_probe.py --out <json>.
Each shape creates one synthetic trellis and shares it, inputs, scales and
scratch across all variants. Timings include input/output Hadamards. The fdot2
direct core is the numerical baseline; this does not measure end-to-end decode.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from exllamav3.ext import exllamav3_ext as ext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/upstream"))
from testlib.exl3 import generator, rand_linear

VARIANTS = {
    "wmma": {"EXL3_GEMV_WMMA": "1", "EXL3_ROCM_GEMV_TILES": "1"},
    "fdot2_tiles": {"EXL3_GEMV_WMMA": "0", "EXL3_ROCM_GEMV_TILES": "1"},
    "fdot2_direct": {"EXL3_GEMV_WMMA": "0", "EXL3_ROCM_GEMV_TILES": "0"},
}


def bench(fn, iterations):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return {"gpu_us": start.elapsed_time(end) * 1000 / iterations,
            "wall_us": (time.perf_counter() - wall_start) * 1e6 / iterations}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    if not torch.version.hip:
        raise RuntimeError("This probe requires the ROCm extension")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    if not arch.startswith(("gfx1200", "gfx1201")):
        raise RuntimeError(f"gfx12 WMMA requires gfx1200/gfx1201, got {arch}")

    os.environ.update({"EXL3_GEMV_MAX_M": "32", "EXL3_GEMV_LDS": "0", "EXL3_GEMV_M4_FDOT2": "0",
                       "EXL3_GEMV_WMMA_U": "4"})
    for key in ("EXL3_GEMV_SPLITK_WARPS", "EXL3_GEMV_TILES_T"):
        os.environ.pop(key, None)
    report = {"scope": "hot synthetic trellis, m4 K3 mul1, fp16 output, Hadamards included",
              "torch": torch.__version__, "hip": torch.version.hip,
              "device": props.name, "arch": arch, "extension": ext.__file__,
              "iterations": args.iterations, "repeats": args.repeats,
              "fixed_env": {key: value for key, value in os.environ.items()
                            if key.startswith("EXL3_")}, "cases": []}
    for k, n in ((5120, 17408), (17408, 5120), (5120, 6144)):
        gen = generator(k + n)
        weights = rand_linear(k, n, 3, gen, "cuda:0")
        x = (torch.randn((4, k), generator=gen) * 0.1).half().cuda()
        xh = torch.empty_like(x)
        y = torch.empty((4, n), dtype=torch.half, device=x.device)

        def call():
            ext.exl3_gemm(x, weights["trellis"], y, weights["suh"], xh,
                         weights["svh"], 0, False, True, 0)

        outputs = {}
        samples = {name: [] for name in VARIANTS}
        # Alternate order to reduce a systematic first/last-variant bias.
        for repeat in range(args.repeats):
            order = list(VARIANTS)
            if repeat % 2:
                order.reverse()
            for name in order:
                os.environ.update(VARIANTS[name])
                samples[name].append(bench(call, args.iterations))
                if name in outputs:
                    assert torch.equal(y, outputs[name]), (k, n, name, "repeat differs")
                else:
                    outputs[name] = y.clone()
        assert torch.equal(outputs["fdot2_tiles"], outputs["fdot2_direct"]), \
            (k, n, "tiles/direct fdot2 differ")
        ref = outputs["fdot2_direct"].float()
        case = {"m": 4, "k": k, "n": n, "bits": 3, "codebook": "mul1", "variants": {}}
        for name, output in outputs.items():
            assert torch.isfinite(output).all(), (k, n, name, "nonfinite output")
            delta = output.float() - ref
            row = {"samples": samples[name],
                   "gpu_us": statistics.median(s["gpu_us"] for s in samples[name]),
                   "wall_us": statistics.median(s["wall_us"] for s in samples[name]),
                   "exact_vs_fdot2_direct": torch.equal(output, outputs["fdot2_direct"]),
                   "max_abs": delta.abs().max().item(),
                   "relative_rms": (delta.square().mean().sqrt() /
                                    ref.square().mean().sqrt().clamp_min(1e-12)).item(),
                   "relative_max": (delta.abs().max() / ref.abs().max().clamp_min(1e-12)).item()}
            assert row["relative_rms"] < 0.005 and row["relative_max"] < 0.005, (k, n, name, row)
            case["variants"][name] = row
        base_us = case["variants"]["wmma"]["gpu_us"]
        for row in case["variants"].values():
            row["speedup_vs_wmma"] = base_us / row["gpu_us"]
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    report["status"] = "PASS"
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {args.out}", flush=True)


if __name__ == "__main__":
    main()
