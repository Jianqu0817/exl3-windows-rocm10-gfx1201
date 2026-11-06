"""Opt-in stage timing; CUDA event intervals include stream idle gaps from host work."""
import functools
import json
import time
from collections import defaultdict

import torch
from exllamav3.generator.job import Job


class DecodeProfile:
    def __init__(self, model, draft, generator_type, limit=64):
        self.active = False
        self.limit = limit
        self.rounds = 0
        self.samples = defaultdict(list)
        self.originals = []
        targets = [
            (model, 'forward', 'target.forward'),
            (draft, 'forward', 'draft.forward'),
            (draft, 'sample_from_state', 'draft.select'),
            (draft, 'update_kv_from_target', 'draft.update'),
            (generator_type, 'iterate_draftmodel_dflash_gen', 'draft.total'),
            (generator_type, 'iterate_gen', 'verify.total'),
            (generator_type, 'iterate', 'iteration'),
            (Job, 'receive_logits', 'sampling.logits'),
            (Job, 'receive_sample', 'sampling.accept'),
            (Job, 'prepare_logit_mask', 'sampling.mask'),
        ]
        seen = set()
        for block in model.modules:
            for role in ('attn', 'mlp'):
                module = getattr(block, role, None)
                kind = type(module).__name__
                if module is not None and kind not in seen:
                    targets.append((module, 'forward', 'representative.' + kind))
                    seen.add(kind)
        for obj, name, label in targets:
            original = getattr(obj, name)
            self.originals.append((obj, name, original))

            def wrap(fn, tag):
                @functools.wraps(fn)
                def timed(*a, **kw):
                    if not self.active or self.rounds >= self.limit:
                        return fn(*a, **kw)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    t = time.perf_counter()
                    result = fn(*a, **kw)
                    elapsed = time.perf_counter() - t
                    end.record()
                    self.samples[tag].append((elapsed, start, end))
                    if tag == 'iteration':
                        self.rounds += 1
                    return result
                return timed

            setattr(obj, name, wrap(original, label))

    def save(self, path):
        torch.cuda.synchronize()
        result = {'rounds': self.rounds,
                  'note': 'Instrumented timing; stream intervals include host gaps and nested stages overlap.',
                  'stages': {}}
        for name, values in self.samples.items():
            result['stages'][name] = {
                'calls': len(values),
                'cpu_ms_per_call': sum(x[0] for x in values) * 1000 / len(values),
                'stream_ms_per_call': sum(x[1].elapsed_time(x[2]) for x in values) / len(values),
            }
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
