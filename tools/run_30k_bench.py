"""Run the existing 30K fixture against this port, with WDDM memory samples."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
spec = importlib.util.spec_from_file_location(
    "gpu_counters", ROOT / "tools/gpu_counters.py")
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--draft-model", type=Path,
                        default=Path(os.environ.get("EXL3_DRAFT_DIR", WORKSPACE / "Qwen3.8-27B-DFlash2-EXL3-3.0bpw-tuned")))
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--input-tokens", type=int, default=30000)
    parser.add_argument("--cache-size", type=int, default=32768)
    parser.add_argument("--max-existing-vram-gib", type=float, default=2.75)
    parser.add_argument("--prompt-dir", type=Path, default=ROOT/"tests/fixtures/30k")
    parser.add_argument("--m4-ab", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "results.json").exists():
        raise SystemExit("Choose a new output directory; existing results are preserved")
    command = [os.environ.get("EXL3_PYTHON") or (str(ROOT/"python/python.exe") if (ROOT/"python/python.exe").is_file() else r"D:\LLM_local\tools\exl3-gfx1201\python\python.exe"), "-I", "-X", "utf8", "-u",
        str(ROOT / "runtime.py"), "script", "tests/bench_30k.py", "-m",
        os.environ.get("EXL3_MODEL_DIR", str(WORKSPACE / "models/Huihui-Qwen3.8-27B-abliterated-EXL3-3.0bpw")),
        "-dm", str(args.draft_model.resolve()),
        "-ndt", "3", "-cs", str(args.cache_size), "-cq", "4", "-rcs", "0", "-chunk_size", "1024",
        "--out", str(args.out.resolve()), "--runs", str(args.runs)]
    command += ["--output-tokens", str(args.output_tokens), "--input-tokens", str(args.input_tokens)]
    if args.profile:
        command.append("--profile")
    if args.m4_ab:
        command.append("--m4-ab")
    if args.prompt_dir:
        command += ["--prompt-dir", str(args.prompt_dir.resolve())]
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
    overrides = dict(item.split("=", 1) for item in args.env)
    env.update(overrides)
    (args.out / "tuning.json").write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    counters = gpu.GpuCounters()
    try:
        time.sleep(1)
        before = counters.read()
        # This port measured 12.67 GiB process-local usage on the 16 GiB card.
        # Budget 12.75 GiB for it and another 0.5 GiB beyond existing desktop use.
        if max(before.get("adapter_dedicated", {}).values(), default=2**40) > args.max_existing_vram_gib * 2**30:
            raise SystemExit(f"GPU test deferred: more than {args.max_existing_vram_gib} GiB VRAM already in use")
        busy = [v for k, v in before.get("engine_utilization", {}).items()
                if "engtype_Compute" in k and v > 10]
        if busy:
            raise SystemExit("GPU test deferred: another compute workload is active")
        (args.out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        with (args.out / "benchmark.log").open("w", encoding="utf-8") as log, \
                (args.out / "gpu-memory.jsonl").open("w", encoding="utf-8") as samples:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            print(f"Benchmark PID {child.pid}; output={args.out}", flush=True)
            while child.poll() is None:
                samples.write(json.dumps({"time": time.time(), "pid": child.pid,
                    "memory": counters.for_pid(counters.read(), child.pid)}) + "\n")
                samples.flush()
                time.sleep(1)
        (args.out / "exit.json").write_text(json.dumps({"exit_code": child.returncode}), encoding="utf-8")
        raise SystemExit(child.returncode)
    finally:
        counters.close()
