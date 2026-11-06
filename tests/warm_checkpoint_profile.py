"""Task-A profiling: warm multi-round recurrent checkpoint stash/restore breakdown.

Repeats one ~6021-token request with -rcs 0.5 and measures wall time of:
  page lookup/allocate, recurrent-state restore (H2D, target/draft sides),
  stash (D2H), target prefill forwards, draft forwards (prefill vs decode phases),
  and the job-reported prefill time. Greedy outputs of repeated rounds must match
  round 1 (checkpoint round-trip sanity). Diagnostic only; not a speed result.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('EXL3_GEMV_WMMA', '1')
os.environ.setdefault('EXL3_GEMV_MAX_M', '32')
import torch
from exllamav3 import Generator, GreedySampler, Job, model_init
from exllamav3.cache.recurrent import PairedState
from exllamav3 import cache as _cache_pkg
import exllamav3.generator.job as _job_mod


class Timings:
    def __init__(self):
        self.data = {}

    def add(self, key, seconds, count=1):
        e = self.data.setdefault(key, {'seconds': 0.0, 'count': 0})
        e['seconds'] += seconds
        e['count'] += count

    def dump(self):
        return {k: {'total_ms': round(v['seconds'] * 1000, 3), 'calls': v['count'],
                    'avg_ms': round(v['seconds'] * 1000 / max(1, v['count']), 4)}
                for k, v in sorted(self.data.items())}


T = Timings()
phase = {'mode': 'idle'}          # 'prefill' | 'decode' | 'idle'
host_copy_stats = {'count': 0, 'bytes': 0, 'seconds': 0.0}


def wrap_forward(module_obj, label):
    orig = module_obj.forward
    def timed_forward(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            if phase['mode'] != 'idle':
                torch.cuda.synchronize()
                T.add(f'{label}.forward.{phase["mode"]}', time.perf_counter() - t0)
    module_obj.forward = timed_forward
    return orig


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_sampling_args=True, add_draft_model_args=True,
                        default_cache_size=32768, default_recurrent_cache_size=0.5,
                        default_chunk_size=1024)
    parser.add_argument('--out', required=True)
    parser.add_argument('--input-tokens', type=int, default=6021)
    parser.add_argument('--output-tokens', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--fixture', default=None,
                        help='defaults: rp512.tasks for <=8192-token targets, rp30000.tasks above')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'benchmark.pid').write_text(str(os.getpid()))
    started = time.perf_counter()

    with torch.inference_mode():
        model, config, cache, tok, dmodel, _, dcache = model_init.init(args)

    # --- instrumentation (script-level wrappers; no library source modified) ---
    wrap_forward(model, 'target')
    wrap_forward(dmodel, 'draft')

    import types as _types
    for obj, name, label in ((model, 'prefill', 'target.model.prefill'),
                             (dmodel, 'update_kv_from_target', 'draft.update_kv_from_target'),
                             (dmodel, 'prefill', 'draft.model.prefill')):
        if obj is None or not hasattr(obj, name):
            continue
        orig = getattr(obj, name)   # bound method captured before patching
        def _mk(orig, label):
            def timed(*a, **kw):    # plain function on the instance: no self injection
                t0 = time.perf_counter()
                try:
                    return orig(*a, **kw)
                finally:
                    torch.cuda.synchronize()
                    T.add(label, time.perf_counter() - t0)
            return timed
        setattr(obj, name, _mk(orig, label))
    orig_job_prefill = _job_mod.Job.prefill
    def timed_job_prefill(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_job_prefill(self, *a, **kw)
        finally:
            T.add('job.prefill(total)', time.perf_counter() - t0)
    _job_mod.Job.prefill = timed_job_prefill

    orig_prep = model.prepare_inputs
    def timed_prep(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_prep(*a, **kw)
        finally:
            T.add('target.prepare_inputs', time.perf_counter() - t0)
    model.prepare_inputs = timed_prep

    import exllamav3.model.model_ls as _mls
    _PBD = _mls.PREFILL_BREAKDOWN

    # per-submodule CUDA-event timing for selected blocks (prefill phase): attn/mlp and
    # their Linear children (projections), to attribute GPU time inside a block
    from exllamav3.modules.linear import Linear
    event_items = []   # (label, start_ev, end_ev)
    for mi, m in enumerate(getattr(model, 'modules', [])):
        if mi % 8 != 0 or not hasattr(m, 'attn'):
            continue
        for role in ('attn', 'mlp'):
            sub = getattr(m, role, None)
            if sub is None or not hasattr(sub, 'forward'):
                continue
            targets = [(f'blk{mi:02d}.{role}', sub)]
            for cname in ('q_proj', 'k_proj', 'v_proj', 'kv_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
                ch = getattr(sub, cname, None)
                if ch is not None and isinstance(ch, Linear) and hasattr(ch, 'forward'):
                    targets.append((f'blk{mi:02d}.{role}.{cname}', ch))
            for lbl, obj in targets:
                o_f = obj.forward
                def _mke(o_f, lbl):
                    def timed(*a, **kw):
                        if phase['mode'] != 'prefill':
                            return o_f(*a, **kw)
                        s = torch.cuda.Event(enable_timing=True)
                        e = torch.cuda.Event(enable_timing=True)
                        s.record()
                        try:
                            return o_f(*a, **kw)
                        finally:
                            e.record()
                            event_items.append((lbl, s, e))
                    return timed
                obj.forward = _mke(o_f, lbl)

    # per-submodule timing for selected blocks (attn vs mlp split, prefill phase)
    sub_times = {}
    for mi, m in enumerate(getattr(model, 'modules', [])):
        if mi % 8 != 0 or not hasattr(m, 'attn'):
            continue
        for role in ('attn', 'mlp'):
            sub = getattr(m, role, None)
            if sub is None or not hasattr(sub, 'forward'):
                continue
            osub = sub.forward
            lbl = f'blk{mi:02d}.{role}'
            def _mksb(osub, lbl):
                def timed(*a, **kw):
                    if phase['mode'] != 'prefill':
                        return osub(*a, **kw)
                    t0 = time.perf_counter()
                    try:
                        return osub(*a, **kw)
                    finally:
                        sub_times.setdefault(lbl, []).append((time.perf_counter() - t0) * 1000)
                return timed
            sub.forward = _mksb(osub, lbl)

    # per-block prefill timing (rounds >= 2 only, prefill phase only)
    block_times = {}
    for mi, m in enumerate(getattr(model, 'modules', [])):
        if not hasattr(m, 'forward'):
            continue
        om = m.forward
        lbl = f'blk{mi:02d}.{type(m).__name__}'
        def _mkbk(om, lbl):
            def timed(*a, **kw):
                if phase['mode'] != 'prefill':
                    return om(*a, **kw)
                t0 = time.perf_counter()
                try:
                    return om(*a, **kw)
                finally:
                    block_times.setdefault(lbl, []).append((time.perf_counter() - t0) * 1000)
            return timed
        m.forward = _mkbk(om, lbl)

    orig_host_copy = _cache_pkg.recurrent.host_copy
    import exllamav3.modules.gated_delta_net as _gdn
    import exllamav3.modules.sliding_attn as _swa
    import exllamav3.modules.short_conv as _sc
    def timed_host_copy(src):
        t0 = time.perf_counter()
        dst = orig_host_copy(src)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        host_copy_stats['count'] += 1
        host_copy_stats['bytes'] += src.numel() * src.element_size()
        host_copy_stats['seconds'] += dt
        return dst
    _cache_pkg.recurrent.host_copy = timed_host_copy
    _gdn.host_copy = timed_host_copy
    _swa.host_copy = timed_host_copy
    if hasattr(_sc, 'host_copy'):
        _sc.host_copy = timed_host_copy

    orig_stash = PairedState.stash
    def timed_stash(self):
        t0 = time.perf_counter()
        try:
            return orig_stash(self)
        finally:
            torch.cuda.synchronize()
            T.add('paired_state.stash(D2H)', time.perf_counter() - t0)
    PairedState.stash = timed_stash

    import types
    for side_name, cache_obj in (('target', cache), ('draft', dcache)):
        if cache_obj is None:
            continue
        orig_fn = type(cache_obj).new_from_stashed
        def make_timed(orig_fn, side_name):
            def timed_new_from_stashed(cache_self, stashed_state, position):
                t0 = time.perf_counter()
                try:
                    return orig_fn(cache_self, stashed_state, position)
                finally:
                    torch.cuda.synchronize()
                    T.add(f'new_from_stashed.{side_name}(H2D)', time.perf_counter() - t0)
            return timed_new_from_stashed
        cache_obj.new_from_stashed = types.MethodType(make_timed(orig_fn, side_name), cache_obj)

    orig_allocate = _job_mod.Job.allocate_pages
    def timed_allocate(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_allocate(self, *a, **kw)
        finally:
            T.add('job.allocate_pages', time.perf_counter() - t0)
    _job_mod.Job.allocate_pages = timed_allocate
    # --- end instrumentation ---

    info = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'load_seconds': time.perf_counter() - started,
            'args': vars(args), 'pid': os.getpid()}
    print('LOAD_COMPLETE', json.dumps(info), flush=True)

    def encode(text):
        return tok.encode(text, add_bos=False, encode_special_tokens=True)

    # Match bench_30k.py's fixture choice: the small task file for short targets (the 30K
    # file's base prompt already exceeds ~8K tokens), the long one for long targets.
    fixture = args.fixture or str(
        Path('E:/LLM_local_NInfer_SPK_test/benchmarks/tasks') /
        ('rp512.tasks' if args.input_tokens <= 8192 else 'rp30000.tasks'))
    raw = Path(fixture).read_bytes()
    source_text = raw.decode('utf-8-sig').replace('\r\n', '\n')
    assert source_text.startswith('###ID ')
    body = source_text.split('\n', 1)[1]
    background, task = body.rsplit('</背景材料>', 1)
    import re
    background = re.sub(r'(?m)^(?: x)+\n', '', background)
    prefix = '<|im_start|>user\n' + background
    suffix = '</背景材料>' + task + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    base = encode(prefix + suffix).shape[-1]
    assert base <= args.input_tokens, f'fixture base prompt {base} exceeds target {args.input_tokens}'
    padding = args.input_tokens - base
    ids = encode(prefix + ' x' * padding + ('\n' if padding else '') + suffix)
    for _ in range(32):
        if ids.shape[-1] == args.input_tokens:
            break
        padding += args.input_tokens - ids.shape[-1]
        assert padding >= 0
        ids = encode(prefix + ' x' * padding + ('\n' if padding else '') + suffix)
    else:
        raise SystemExit('padding did not converge')
    assert ids.shape[-1] == args.input_tokens
    print('PROMPT_TOKENS', ids.shape[-1], flush=True)

    with torch.inference_mode():
        gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dmodel, draft_cache=dcache,
                        num_draft_tokens=args.num_draft_tokens, max_batch_size=1,
                        max_chunk_size=args.chunk_size, cpu_cache_size=0,
                        recurrent_cache_size=int(args.recurrent_cache_size * 1024**3))

    text = tok.decode(ids, decode_special_tokens=True)[0]
    (out / 'prompt.txt').write_text(text, encoding='utf-8')

    rounds = []
    completions = []
    for r in range(1, args.rounds + 1):
        T.data.clear()
        block_times.clear()
        sub_times.clear()
        event_items.clear()
        sub_times.clear()
        for _k in _PBD:
            _PBD[_k] = [] if _k == 'events' else (0 if _k == 'calls' else 0.0)
        host_copy_stats.update(count=0, bytes=0, seconds=0.0)
        phase['mode'] = 'prefill'
        first, last = None, None
        prev_prefill_done = False
        iterate_wall = {'prefill': 0.0, 'decode': 0.0}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            job = Job(input_ids=ids.clone(), max_new_tokens=args.output_tokens, min_new_tokens=args.output_tokens,
                      sampler=GreedySampler(), stop_conditions=config.eos_token_id_list or [])
            gen.enqueue(job)
            while gen.num_remaining_jobs():
                done = job.is_prefill_done()
                if done != prev_prefill_done:
                    phase['mode'] = 'decode' if done else 'prefill'
                    prev_prefill_done = done
                t_iter = time.perf_counter()
                results = gen.iterate()
                iterate_wall[phase['mode']] += time.perf_counter() - t_iter
                for result in results:
                    if result.get('error'):
                        print('JOB_ERROR', json.dumps(str(result['error']))[:2000], flush=True)
                    if result.get('token_ids') is not None and result['token_ids'].numel() and first is None:
                        torch.cuda.synchronize()
                        first = time.perf_counter() - t0
                    if result.get('eos'):
                        last = result
            phase['mode'] = 'idle'
        torch.cuda.synchronize()
        # resolve the per-module CUDA events queued by the temp prefill_ls instrumentation
        evs = _PBD.get('events') or []
        if evs:
            agg = {}
            for idx, name, s, e in evs:
                agg.setdefault((idx, name), [0.0, 0])
                agg[(idx, name)][0] += s.elapsed_time(e)
                agg[(idx, name)][1] += 1
            top = sorted(((f'{k[0]}:{k[1]}', round(v[0], 3), v[1]) for k, v in agg.items()),
                         key=lambda x: -x[1])[:14]
            row_events = top
            total_ev = round(sum(v[0] for v in agg.values()), 2)
        else:
            row_events, total_ev = None, None
        _PBD['events'] = []
        agg_sub = {}
        for lbl, s, e in event_items:
            agg_sub.setdefault(lbl, [0.0, 0])
            agg_sub[lbl][0] += s.elapsed_time(e)
            agg_sub[lbl][1] += 1
        row_subevents = sorted(((k, round(v[0], 3), v[1]) for k, v in agg_sub.items()),
                               key=lambda x: -x[1])[:16] if agg_sub else None
        wall = time.perf_counter() - t0
        assert last is not None
        row = {'round': r,
               'wall_seconds': round(wall, 4),
               'ttft_seconds': round(first, 4) if first else None,
               'time_prefill': last.get('time_prefill'),
               'result_keys': sorted(last.keys()),
               'cached_tokens': last.get('cached_tokens'),
               'prompt_tokens': last.get('prompt_tokens'),
               'new_tokens': last.get('new_tokens'),
               'accepted_draft_tokens': last.get('accepted_draft_tokens'),
               'rejected_draft_tokens': last.get('rejected_draft_tokens'),
               'phases': T.dump(),
               'prefill_ls_breakdown': {k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in _PBD.items() if k != 'events'},
               'prefill_events_top': row_events,
               'prefill_events_total_ms': total_ev,
               'submodule_events': row_subevents,
               'block_prefill_top': sorted(
                   ((k, round(sum(v), 2), len(v)) for k, v in block_times.items()),
                   key=lambda x: -x[1])[:12] if r >= 2 else None,
               'submodule_prefill': sorted(
                   ((k, round(sum(v), 2), len(v)) for k, v in sub_times.items()),
                   key=lambda x: -x[1]) if r >= 2 else None,
               'iterate_wall_s': {k: round(v, 4) for k, v in iterate_wall.items()},
               'host_copy': {'count': host_copy_stats['count'],
                             'mib': round(host_copy_stats['bytes'] / 2**20, 2),
                             'total_ms': round(host_copy_stats['seconds'] * 1000, 3)},
               'peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        completions.append(last.get('full_completion'))
        rounds.append(row)
        print('ROUND', json.dumps(row), flush=True)
        (out / 'results.json').write_text(json.dumps({'environment': info, 'rounds': rounds},
                                                     ensure_ascii=False, indent=2), encoding='utf-8')

    identical = all(c == completions[0] for c in completions if c is not None)
    warm_prefills = [r['time_prefill'] for r in rounds[1:] if r.get('time_prefill') is not None]
    summary = {'rounds': args.rounds, 'input_tokens': args.input_tokens,
               'all_round_outputs_identical': identical,
               'warm_prefill_median': statistics.median(warm_prefills) if warm_prefills else None}
    (out / 'summary.json').write_text(json.dumps({'summary': summary, 'rounds': rounds}, indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps(summary), flush=True)
    gen.close()


if __name__ == '__main__':
    main()
