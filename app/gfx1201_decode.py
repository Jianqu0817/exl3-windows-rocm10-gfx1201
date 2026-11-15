"""Single-stream Huihui/DFlash2: recover a slow WDDM output-head placement before serving."""
import os
from statistics import median
import torch
from exllamav3 import model_init

RESTART_HEAD_FIRST = 75

@torch.inference_mode()
def head_time_ms(head):
    x = torch.zeros((1, 4, head.in_features), dtype=torch.half, device=head.device)
    stream = torch.cuda.current_stream(head.device)
    head.forward(x, {})
    samples = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        head.forward(x, {})
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return median(samples)

def placement_ms(loaded):
    model, _, _, _, draft, _, _ = loaded
    if draft is None or not hasattr(draft, "selector") or model.logit_layer_idx is None:
        return None
    head = model.modules[model.logit_layer_idx]
    if (head.device != torch.device("cuda:0") or getattr(head, "quant_type", None) != "exl3"
            or head.in_features != 5120 or head.out_features != 248320
            or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1201")):
        return None
    return head_time_ms(head)

def load_for_decode(args, **kwargs):
    first = os.environ.get("EXL3_DECODE_PLACEMENT") == "head-first"
    loaded = model_init.init(args, early_output_head=first, **kwargs)
    ms = placement_ms(loaded)
    if ms is None:
        return loaded
    print(f"DECODE_HEAD {'head-first' if first else 'normal'} {ms:.2f} ms", flush=True)
    if ms >= 10 and not first:
        # A new process is intentional: unloading/reloading in the same HIP context
        # retained the ~40 ms placement in the pressure test. No jobs admitted yet.
        print("DECODE_HEAD slow: requesting one fresh head-first launch", flush=True)
        return None
    if ms >= 10:
        print("DECODE_HEAD still slow; serving without another restart", flush=True)
    return loaded
