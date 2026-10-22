"""Skip discarded sampling only when every generated target is already supplied.

This is for non-emitting system/tool prefill, never live audio generation. Keep
LMGen's delayed-input preparation, main transformer KV update, cache mask and
offset transitions. Partially forced steps retain the upstream depth path.
A backend may supply a measured RNG-advance callback to preserve the original
CUDA sampling state as well as the conditioned tokens.
"""


def forced_prefill_step(generator, *, input_tokens, moshi_tokens, text_token, advance_rng=None):
    prepared = generator.prepare_step_input(
        input_tokens=input_tokens, moshi_tokens=moshi_tokens, text_token=text_token,
    )
    if prepared is None:
        return False
    inputs, provided, targets, previous, current = prepared
    state = generator._streaming_state
    transformer, logits = state.graphed_main(inputs)
    fully_forced = (
        not generator.report_loss and not generator.return_logits
        and bool(provided[:, :generator.lm_model.dep_q + 1].all())
    )
    if not fully_forced:
        generator.process_transformer_output(
            transformer, logits, provided, targets, previous, current,
        )
        return False
    # Upstream's generated-cache writes are no-ops when these masks are true.
    # The returned delayed output is discarded by every system prefill caller.
    state.provided[:, :, previous] = False
    state.offset += 1
    if advance_rng is not None:
        advance_rng()
    return True


async def prefill_text(generator, is_current, advance_rng=None):
    zero, sine = generator._encode_zero_frame(), generator._encode_sine_frame()
    skipped = 0
    for token in generator.text_prompt_tokens:
        if not await is_current():
            break
        skipped += forced_prefill_step(
            generator, input_tokens=sine, moshi_tokens=zero, text_token=token, advance_rng=advance_rng,
        )
    return skipped
