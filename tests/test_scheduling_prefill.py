"""Forced prefill must retain the actual LMGen preparation and main KV step."""
from types import SimpleNamespace
import numpy as np
import pytest


class Generator:
    def __init__(self, missing=False, report=False):
        self.report_loss, self.return_logits = report, False
        self.lm_model = SimpleNamespace(dep_q=2)
        self.cache = np.arange(12).reshape(1, 3, 4)
        self.provided = np.ones((1, 3, 4), dtype=bool)
        self.provided[0, 1, 2] = not missing
        self.main_calls, self.depth_calls, self.prepare_calls = 0, 0, 0
        self._streaming_state = SimpleNamespace(cache=self.cache, provided=self.provided, offset=6, graphed_main=self.main)
    def prepare_step_input(self, **kwargs):
        self.prepare_calls += 1
        assert kwargs == dict(input_tokens='sine', moshi_tokens='zero', text_token=10)
        return self.cache[:, :, 1:2], self.provided[:, :, 2:3], self.cache[:, :, 2:3], 1, 2
    def main(self, inputs):
        self.main_calls += 1
        return 'kv', 'logits'
    def process_transformer_output(self, *args):
        self.depth_calls += 1
        self.provided[:, :, 1] = False
        self._streaming_state.offset += 1


def test_fully_forced_prefill_keeps_main_kv_and_cache_without_sampling():
    from speaker_lab.scheduling.prefill import forced_prefill_step
    gen = Generator()
    before = gen.cache.copy()
    assert forced_prefill_step(gen, input_tokens='sine', moshi_tokens='zero', text_token=10)
    assert gen.prepare_calls == gen.main_calls == 1 and gen.depth_calls == 0
    np.testing.assert_array_equal(gen.cache, before)
    assert not gen.provided[:, :, 1].any() and gen.provided[:, :, 2].all()
    assert gen._streaming_state.offset == 7


@pytest.mark.parametrize('missing,report', [(True, False), (False, True)])
def test_partial_targets_or_requested_logits_use_original_depth_path(missing, report):
    from speaker_lab.scheduling.prefill import forced_prefill_step
    gen = Generator(missing, report)
    assert not forced_prefill_step(gen, input_tokens='sine', moshi_tokens='zero', text_token=10)
    assert gen.prepare_calls == gen.main_calls == gen.depth_calls == 1
    assert gen._streaming_state.offset == 7


def test_offset_zero_uses_upstream_initialization_without_a_main_step():
    from speaker_lab.scheduling.prefill import forced_prefill_step
    gen = Generator()
    gen.prepare_step_input = lambda **kwargs: None
    assert not forced_prefill_step(gen, input_tokens='sine', moshi_tokens='zero', text_token=10)
    assert gen.main_calls == gen.depth_calls == 0


def test_text_prefill_checks_cancellation_before_every_token():
    import asyncio
    from speaker_lab.scheduling.prefill import prefill_text
    gen = Generator()
    gen.text_prompt_tokens = [10, 10, 10]
    gen._encode_zero_frame = lambda: 'zero'
    gen._encode_sine_frame = lambda: 'sine'
    async def current():
        return gen.prepare_calls < 1
    assert asyncio.run(prefill_text(gen, current)) == 1
    assert gen.prepare_calls == gen.main_calls == 1


def test_rng_advance_only_skips_the_discarded_sampling_path():
    from speaker_lab.scheduling.prefill import forced_prefill_step
    advances = []
    for missing in (False, True):
        gen = Generator(missing=missing)
        forced_prefill_step(gen, input_tokens='sine', moshi_tokens='zero', text_token=10,
                            advance_rng=lambda: advances.append(1))
    assert advances == [1]
