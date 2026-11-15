"""Opt-in RP decoding constraint; imports no model runtime at module load."""

import re


_CLOSERS = frozenset('\"\'”’」』）》】〕〉）)]}*_`~')
_TERMINALS = frozenset('。！？!?…')


def sentence_end_ready(text: str) -> bool:
    """Allow empty output or terminal punctuation followed by quote/Markdown closers.

    A bare period after a digit remains ambiguous with an unfinished decimal.
    This punctuation policy is intended for prose, not arbitrary structured output.
    """
    text = text.rstrip()
    if not text:
        return True
    while text and text[-1] in _CLOSERS:
        text = text[:-1].rstrip()
    if not text:
        return False
    if re.search(r'</[^\W\d][\w:.-]*\s*>$', text):
        return True
    if text[-1] in _TERMINALS or text.endswith('...'):
        return True
    return text.endswith('.') and (len(text) == 1 or not text[-2].isdigit())


def create_sentence_end_guard(tokenizer, eos_token_ids, base_sampler):
    """Delay an incomplete natural EOS until the current sentence ends.

    Before the model first chooses an incomplete EOS, its sampling is unchanged.
    Afterwards only the current sentence is completed, then natural EOS is required.
    """
    import numpy as np
    import torch
    from exllamav3.generator.filter.filter import Filter
    from exllamav3.generator.sampler.sampler import Sampler

    eos_token_ids = tuple(sorted(set(eos_token_ids)))

    class SentenceEndFilter(Filter):
        def __init__(self):
            super().__init__(tokenizer, None, None, False)
            self.pending_stop = False

        def attach(self, job):
            super().attach(job)
            self._allow = torch.full((1, (self.vocab_size + 31) // 32), -1, dtype=torch.int32)
            self._block = self._allow.clone()
            bits = self._block.numpy().view(np.uint32).reshape(-1)
            for token in eos_token_ids:
                bits[token >> 5] &= ~np.uint32(1 << (token & 31))
            self._only = ~self._block

        def reset(self):
            pass

        def accept_token(self, token):
            pass

        def is_completed(self):
            return False

        def use_background_worker(self):
            return False

        def sentence_ready(self):
            # Job owns decoding, held Unicode fragments, rewind and requeue state.
            text = self.job.full_completion[-256:] + self.job.held_text
            return sentence_end_ready(text)

        def get_next_logit_mask(self):
            if not self.pending_stop:
                return self._allow
            return self._only if self.sentence_ready() else self._block

    guard = SentenceEndFilter()

    class SentenceEndSampler(Sampler):
        def __init__(self):
            super().__init__()
            self.reqs_past_ids = base_sampler.reqs_past_ids
            self.reqs_torch_seed = base_sampler.reqs_torch_seed

        def forward(self, logits, sequence_ids=None, rand_u32=None, tokenizer=None,
                    logit_mask=None, return_state=False):
            result = base_sampler.forward(logits, sequence_ids, rand_u32, tokenizer,
                logit_mask=logit_mask, return_state=return_state)
            if guard.pending_stop:
                return result
            sampled = result.sample if return_state else result
            if sampled.item() not in eos_token_ids or guard.sentence_ready():
                return result
            guard.pending_stop = True
            # Retain the current Job mask, including any other active constraints.
            if logit_mask is None:
                retry_mask = guard._block.to(logits.device)
            elif logit_mask.dtype == torch.int32:
                retry_mask = logit_mask & guard._block.to(logit_mask.device)
            else:
                retry_mask = logit_mask.clone()
                retry_mask[..., list(eos_token_ids)] = float('-inf')
            return base_sampler.forward(logits, sequence_ids, rand_u32, tokenizer,
                logit_mask=retry_mask, return_state=return_state)

    return guard, SentenceEndSampler()
