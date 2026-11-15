"""CPU-only policy and real EXL3 Filter lifecycle checks, without loading the runtime."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from sentence_end import create_sentence_end_guard, sentence_end_ready


def main():
    # Importing the policy alone must not import torch or the model runtime.
    assert 'torch' not in sys.modules and 'exllamav3' not in sys.modules
    cases = {
        '': True, ' \n': True, '一句话。': True, '好了！': True, '真的吗？': True,
        '停在这里…': True, 'Maybe...': True, 'Done.': True,
        '**“一句话。”**\n': True, '*（动作结束。）*': True,
        '"Done."': True, '`Done.`': True,
        '<时间>晚上</时间>': True,
        '半句话': False, '半句话，': False, '接下来：': False, '还有、': False,
        '数值3.14': False, '数值3.': False, '数值3.14"': False,
        '完成。接着半句': False, '**': False,
        '<status>状态</status>': True, '<div>状态</div>\n': True,
        '**<state>状态</state>**': True, '<status>未关闭': False,
    }
    for text, expected in cases.items():
        assert sentence_end_ready(text) is expected, repr(text)

    import numpy as np
    import torch

    # Load the actual base class with only its annotation dependency stubbed.
    names = ('exllamav3', 'exllamav3.generator', 'exllamav3.generator.filter',
             'exllamav3.generator.sampler',
             'exllamav3.tokenizer')
    modules = {name: ModuleType(name) for name in names}
    modules['exllamav3.tokenizer'].Tokenizer = object
    specs = []
    for name, path in (('exllamav3.generator.filter.filter', 'filter/filter.py'),
                       ('exllamav3.generator.sampler.sampler', 'sampler/sampler.py')):
        spec = importlib.util.spec_from_file_location(name,
            ROOT / 'app/exllamav3/generator' / path)
        module = importlib.util.module_from_spec(spec)
        modules[name] = module
        specs.append((spec, module))
    with patch.dict(sys.modules, modules):
        for spec, module in specs:
            spec.loader.exec_module(module)

        class FakeSampler:
            reqs_past_ids = True
            reqs_torch_seed = True

            def __init__(self):
                self.calls = []
                self.preferred = 2

            def forward(self, logits, sequence_ids, rand_u32, tokenizer,
                        logit_mask=None, return_state=False):
                self.calls.append((rand_u32, logit_mask.clone()))
                words = logit_mask.numpy().view(np.uint32).reshape(-1)
                token = next(i for i in (self.preferred, 70, 71, 2)
                    if int(words[i >> 5]) & (1 << (i & 31)))
                result = torch.tensor([[token]])
                return SimpleNamespace(sample=result) if return_state else result

        base_sampler = FakeSampler()
        guard, sampler = create_sentence_end_guard(object(), [2, 31, 32, 63], base_sampler)
        job = SimpleNamespace(generator=SimpleNamespace(padded_vocab_size=96),
            full_completion='', held_text='')
        guard.attach(job)
        guard.reset()
        assert guard.is_active and not guard.eos_after_completed
        assert not guard.use_background_worker()
        assert sampler.reqs_past_ids and sampler.reqs_torch_seed
        assert isinstance(sampler, modules['exllamav3.generator.sampler.sampler'].Sampler)
        allow = guard.get_next_logit_mask()
        assert allow.dtype == torch.int32 and tuple(allow.shape) == (1, 3)
        assert allow.eq(-1).all()

        job.full_completion = '已经结束。'
        job.held_text = '下一句还没结束'
        assert guard.get_next_logit_mask() is allow
        logits = torch.zeros((1, 1, 96))
        # Ordinary samples pass through before the first incomplete EOS.
        base_sampler.preferred = 70
        assert sampler.forward(logits, rand_u32=17, logit_mask=allow).item() == 70
        assert len(base_sampler.calls) == 1 and not guard.pending_stop
        base_sampler.preferred = 2
        # Reuse the same random seed exactly once, retaining other token constraints.
        combined = allow.clone()
        combined[0, 2] &= ~(1 << (70 & 31))
        assert sampler.forward(logits, rand_u32=19, logit_mask=combined).item() == 71
        assert [seed for seed, _ in base_sampler.calls[-2:]] == [19, 19]
        assert guard.pending_stop
        assert not guard.feed(70)
        blocked = guard.get_next_logit_mask()
        words = blocked.numpy().view(np.uint32).reshape(-1)
        assert {i for i in range(96) if not int(words[i >> 5]) & (1 << (i & 31))} == {2, 31, 32, 63}
        assert sampler.forward(logits, rand_u32=23, logit_mask=blocked).item() == 70
        assert len(base_sampler.calls) == 4  # No further retry while pending.
        # Completion requires EOS immediately after the accepted current sentence.
        job.held_text += '。”**'
        only_eos = guard.get_next_logit_mask()
        words = only_eos.numpy().view(np.uint32).reshape(-1)
        assert {i for i in range(96) if int(words[i >> 5]) & (1 << (i & 31))} == {2, 31, 32, 63}
        assert sampler.forward(logits, rand_u32=29, logit_mask=only_eos).item() == 2
        job.held_text = '下一句还没结束'
        guard.rewind(1)
        assert guard.get_next_logit_mask() is blocked
        assert not guard.is_completed()

        # A naturally complete EOS (including empty output) remains one original call.
        for text in ('一句完整的话。', '', '<status>状态</status>'):
            original = FakeSampler()
            complete, wrapped = create_sentence_end_guard(object(), [2], original)
            complete_job = SimpleNamespace(generator=job.generator, full_completion=text, held_text='')
            complete.attach(complete_job)
            mask = complete.get_next_logit_mask()
            state = wrapped.forward(logits, rand_u32=31, logit_mask=mask, return_state=True)
            assert state.sample.item() == 2 and len(original.calls) == 1
            assert not complete.pending_stop
        assert not torch.cuda.is_initialized()
    print(json.dumps({'status': 'PASS', 'policy_cases': len(cases),
        'packed_mask_and_filter_lifecycle': True, 'cuda_initialized': False}))


if __name__ == '__main__':
    main()
