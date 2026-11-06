"""Reuse upstream's decode/prefill reference checks at Qwen's GQA/head geometry."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build/test-deps"))
sys.path.insert(0, str(ROOT / "upstream_tests"))
import pytest

spec = importlib.util.spec_from_file_location(
    "decode_layout_checks", ROOT / "upstream_tests/kernels/attention/test_paged_decode_layout.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
count = 0
for rows in (1, 4, 8):
    for splits in (1, 16):
        checks.test_decode_layout_vs_prefill("cuda:0", 256, 4, rows, 6, splits)
        count += 1
for order in ((1, 16), (16, 1)):
    with pytest.MonkeyPatch.context() as patch:
        checks.test_decode_stage_fallback("cuda:0", patch, order)
    count += 1
torch.cuda.synchronize()
result = {"status": "PASS", "cases": count, "head_dim": 256, "kv_bits": 4,
          "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
(ROOT / "logs/port-attention-smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result), flush=True)
