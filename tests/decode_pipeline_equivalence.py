"""Paired full-model checks: minimum-length boundaries, RNG, metadata and forced tokens."""
import argparse
import json
from pathlib import Path

import torch
from exllamav3 import Generator, Job, Filter, model_init
from exllamav3.generator.sampler import ArgmaxSampler, CategoricalSampler, CustomSampler, SS_RepP, SS_Argmax
import exllamav3.generator.generator as gen_mod
import exllamav3.generator.job as job_mod

parser = argparse.ArgumentParser()
model_init.add_args(parser, cache=True, add_sampling_args=True, add_draft_model_args=True,
                    default_cache_size=32768, default_recurrent_cache_size=0, default_chunk_size=1024)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
(ROOT/'logs/decode-pipeline-equivalence.json').unlink(missing_ok=True)
with torch.inference_mode():
    model, config, cache, tok, draft, _, dcache = model_init.init(args)
    prefix = '<|im_start|>user\nPlease repeat the sentence "The fox ran." many times.'
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    pad = 1024 - tok.encode(prefix+suffix, add_bos=False, encode_special_tokens=True).shape[-1]
    ids = tok.encode(prefix+' x'*pad+suffix, add_bos=False, encode_special_tokens=True)

    class AllowSetFilter(Filter):
        def __init__(self):
            super().__init__(tok, None, None, False)
            self.allowed = tok.encode(' red blue', add_bos=False).flatten().tolist()
        def reset(self): pass
        def accept_token(self, token): pass
        def is_completed(self): return False
        def use_background_worker(self): return False
        def get_next_logit_mask(self):
            mask = torch.full((1, self.vocab_size), -float('inf'), dtype=torch.half)
            mask[:, self.allowed] = 0
            return mask

    def run(enabled, minimum, kind='greedy', stop=None, metadata=False, force=False,
            healing=False, filtering=False, banned=False):
        gen_mod.decode_pipeline_enable = job_mod.decode_pipeline_enable = enabled
        gen_mod.head_window_enable = enabled
        sampler = (CategoricalSampler(0.8) if kind == 'sample' else
                   CustomSampler([SS_RepP(1.05), SS_Argmax()]) if kind == 'penalty' else ArgmaxSampler())
        widths = []
        original = sampler.forward
        def record(logits, *a, **kw):
            widths.append(logits.shape[-2])
            return original(logits, *a, **kw)
        sampler.forward = record
        gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                        num_draft_tokens=3, max_batch_size=1, max_chunk_size=1024,
                        cpu_cache_size=0, recurrent_cache_size=0)
        job = Job(input_ids=ids, max_new_tokens=64, min_new_tokens=minimum, sampler=sampler, seed=1234,
                  stop_conditions=stop if stop is not None else config.eos_token_id_list or [],
                  return_probs=metadata, return_top_tokens=3 if metadata else 0,
                  token_healing=healing, filters=[AllowSetFilter()] if filtering else None,
                  banned_strings=['fox'] if banned else None)
        gen.enqueue(job)
        if force:
            job.constrain_output_now(' TEST')
        while gen.num_remaining_jobs():
            gen.iterate()
        result = (job.sequences[0].sequence_ids.torch()[0].tolist(), job.rng.getstate(),
                  job.accepted_draft_tokens, job.rejected_draft_tokens, widths)
        assert job.new_tokens >= minimum
        gen.close()
        return result

    probe = run(False, 0, stop=[])
    # Explicitly check selector prefixes across head row tiles and batch layouts.
    torch.manual_seed(42)
    for batch in (1, 2):
        state = torch.randn((batch, draft.config.block_size, draft.config.hidden_size),
                            dtype=torch.half, device=draft.output_device)
        anchors = torch.full((batch, 1), 42, dtype=torch.long, pin_memory=True)
        full = draft.sample_from_state(state, {'dflash2_anchor_ids': anchors})
        for rows in (2, 4, 8):
            cropped = draft.sample_from_state(state, {'dflash2_anchor_ids': anchors, 'dflash2_head_rows': rows})
            assert torch.equal(cropped, full[:, :rows]), ('selector prefix', batch, rows)
    first_id = probe[0][ids.shape[-1]]
    cases = [(n, 'greedy', None, False, False) for n in (0, 1, 5, 32, 64)]
    cases += [(n, 'greedy', [first_id], False, False) for n in (5, 32)]
    cases += [(n, 'sample', None, False, False) for n in (0, 5, 32)]
    cases += [(5, 'penalty', None, False, False), (5, 'greedy', None, True, False),
              (5, 'greedy', None, False, True)]
    report = []
    for minimum, kind, stop, metadata, force in cases:
        before = run(False, minimum, kind, stop, metadata, force)
        after = run(True, minimum, kind, stop, metadata, force)
        assert before[:4] == after[:4], (minimum, kind, stop, metadata, force, 'sequence/RNG/draft counters differ')
        row = dict(minimum=minimum, sampler=kind, custom_stop=stop is not None,
                   metadata=metadata, forced=force, output_tokens=len(after[0])-ids.shape[-1],
                   vector_sampler_calls=sum(n>1 for n in after[4]))
        report.append(row)
        print(json.dumps(row), flush=True)
    for name in ('healing', 'filtering', 'banned'):
        before = run(False, 5, **{name: True})
        after = run(True, 5, **{name: True})
        assert before[:4] == after[:4], (name, 'sequence/RNG/draft counters differ')
        report.append({'constraint': name, 'status': 'PASS'})
        print('PASS', name, flush=True)
    assert sum(r.get('vector_sampler_calls', 0) for r in report) > 0
(ROOT/'logs/decode-pipeline-equivalence.json').write_text(
    json.dumps({'status':'PASS','selector_prefix_cases':6,'cases':report},indent=2),encoding='utf-8')
print('PASS', len(report), flush=True)
