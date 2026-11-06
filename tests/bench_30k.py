"""Single-stream EXL3 benchmark; every measured request has zero cached tokens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

os.environ.setdefault('EXL3_GEMV_WMMA', '1')
os.environ.setdefault('EXL3_GEMV_MAX_M', '32')
import torch
from exllamav3 import Generator, GreedySampler, Job, model_init


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_sampling_args=True, add_draft_model_args=True,
                        default_cache_size=32768, default_recurrent_cache_size=0.0, default_chunk_size=1024)
    parser.add_argument('--out', required=True)
    parser.add_argument('--input-tokens', type=int, default=30000)
    parser.add_argument('--output-tokens', type=int, default=512)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'benchmark.pid').write_text(str(os.getpid()))
    started = time.perf_counter()
    with torch.inference_mode():
        model, config, cache, tok, dmodel, _, dcache = model_init.init(args)
    profiler = None
    if args.profile:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from decode_profile import DecodeProfile
        profiler = DecodeProfile(model, dmodel, Generator)
    def new_generator():
        return Generator(model=model, cache=cache, tokenizer=tok, draft_model=dmodel, draft_cache=dcache,
                         num_draft_tokens=args.num_draft_tokens, max_batch_size=1,
                         max_chunk_size=args.chunk_size, cpu_cache_size=0,
                         recurrent_cache_size=int(args.recurrent_cache_size * 1024**3))
    info = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'hip': torch.version.hip, 'effective_exl3_environment': {k:v for k,v in os.environ.items() if k.startswith('EXL3_')}, 'args': vars(args), 'pid': os.getpid(),
            'load_seconds': time.perf_counter() - started,
            'method': 'greedy; no vision; no drafting unless explicitly enabled; fresh generator per request; 0 cached tokens required',
            'fixture': 'E:/LLM_local_NInfer_SPK_test/benchmarks/tasks/rp30000.tasks',
            'tuning': {k:os.environ.get(k) for k in ('EXL3_DEC_GROUP', 'EXL3_DEC_Q4W',
                       'EXL3_DEC_WARPS', 'EXL3_DEC_BLOCK_N', 'EXL3_DEC_STAGES', 'EXL3_GRAPHS')}}
    print('LOAD_COMPLETE', json.dumps(info), flush=True)

    def encode(text):
        return tok.encode(text, add_bos=False, encode_special_tokens=True)

    def prompt(run, target):
        name = 'rp512.tasks' if run == 'warmup' else 'rp30000.tasks'
        source = Path('E:/LLM_local_NInfer_SPK_test/benchmarks/tasks') / name
        raw = source.read_bytes()
        source_text = raw.decode('utf-8-sig').replace('\r\n', '\n')
        assert source_text.startswith('###ID ')
        body = source_text.split('\n', 1)[1]
        background, task = body.rsplit('</背景材料>', 1)
        background = re.sub(r'(?m)^(?: x)+\n', '', background)
        prefix = '<|im_start|>user\n' + background
        suffix = '</背景材料>' + task + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        base = encode(prefix + suffix).shape[-1]
        padding = target - base
        assert padding >= 0
        ids = encode(prefix + ' x' * padding + ('\n' if padding else '') + suffix)
        while ids.shape[-1] != target:
            padding += target - ids.shape[-1]
            assert padding >= 0
            ids = encode(prefix + ' x' * padding + ('\n' if padding else '') + suffix)
        assert ids.shape[-1] == target
        (out / f'{run}.fixture.json').write_text(json.dumps({'source': str(source),
            'source_sha256': hashlib.sha256(raw).hexdigest(), 'padding_units': padding,
            'prepared_tokens': ids.shape[-1], 'change': 'Reuse original SYSTEM/background/task; rebuild x padding and EXL3 chat template.'}, indent=2), encoding='utf-8')
        return ids

    rows = []
    def run(label, ids, tokens, measured):
        with torch.inference_mode():
            gen = new_generator()
        text = tok.decode(ids, decode_special_tokens=True)[0]
        (out / f'{label}.prompt.txt').write_text(text, encoding='utf-8')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        wall_start = time.time()
        t0 = time.perf_counter()
        first, last, last_progress = None, None, t0
        with torch.inference_mode():
            job = Job(input_ids=ids, max_new_tokens=tokens, min_new_tokens=tokens,
                      sampler=GreedySampler(), stop_conditions=config.eos_token_id_list or [])
            gen.enqueue(job)
            while gen.num_remaining_jobs():
                if profiler is not None:
                    profiler.active = measured and job.is_prefill_done()
                results = gen.iterate()
                for result in results:
                    if result.get('token_ids') is not None and result['token_ids'].numel() and first is None:
                        torch.cuda.synchronize()
                        first = time.perf_counter() - t0
                        print('FIRST_TOKEN', label, first, flush=True)
                    if result.get('eos'):
                        last = result
                if time.perf_counter() - last_progress > 10:
                    print('PROGRESS', label, 'elapsed_s', round(time.perf_counter()-t0, 1), 'generated', job.new_tokens, flush=True)
                    last_progress = time.perf_counter()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        assert last is not None
        assert last['cached_tokens'] == 0, last
        assert last['prompt_tokens'] == ids.shape[-1], last
        assert last['new_tokens'] == tokens, last
        completion = last['full_completion']
        (out / f'{label}.completion.txt').write_text(completion, encoding='utf-8')
        row = {'label': label, 'measured': measured, 'started_at': wall_start,
               'finished_at': time.time(), 'prompt_tokens': last['prompt_tokens'],
               'output_tokens': last['new_tokens'], 'cached_tokens': last['cached_tokens'],
               'prefill_seconds': last['time_prefill'], 'prefill_tps': last['prompt_tokens']/last['time_prefill'],
               'engine_generate_seconds': last['time_generate'], 'generate_tps': last['new_tokens']/last['time_generate'],
               'observed_first_token_seconds': first, 'wall_seconds': elapsed,
               'torch_peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
               'torch_peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
               'prompt_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
               'eos_reason': last.get('eos_reason')}
        if last.get('accepted_draft_tokens') is not None:
            accepted = last['accepted_draft_tokens']
            rejected = last['rejected_draft_tokens']
            row.update(accepted_draft_tokens=accepted, rejected_draft_tokens=rejected,
                       draft_acceptance=accepted/max(1, accepted+rejected))
        rows.append(row)
        (out / 'results.json').write_text(json.dumps({'environment': info, 'runs': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
        print('BENCH_RESULT', json.dumps(row), flush=True)
        gen.close()
        if profiler is not None:
            profiler.active = False
            if measured:
                profiler.save(out / 'decode-profile.json')

    run('warmup', prompt('warmup', 1024), 512, False)
    for i in range(args.runs):
        run(f'cold30k-{i+1}', prompt(f'cold30k-{i+1}', args.input_tokens), args.output_tokens, True)
    measured = [r for r in rows if r['measured']]
    summary = {key: statistics.median(r[key] for r in measured) for key in
               ('prefill_tps', 'prefill_seconds', 'generate_tps', 'observed_first_token_seconds', 'wall_seconds')}
    (out / 'results.json').write_text(json.dumps({'environment': info, 'runs': rows, 'median': summary}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('BENCH_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
