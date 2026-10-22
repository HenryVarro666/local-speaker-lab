"""Control payloads are bounded; forced text preserves the exact token order."""

import json
import pytest


def test_latest_control_replaces_pending_and_stale_command_is_rejected():
    from speaker_lab.scheduling.native_control import NativeControl

    control = NativeControl()
    assert control.submit(
        json.dumps(
            {"mode": "context", "epoch": 2, "text": "Friday at four thirty."}
        ).encode()
    )
    assert not control.submit(
        json.dumps({"mode": "context", "epoch": 1, "text": "Thursday."}).encode()
    )
    assert control.submit(json.dumps({"mode": "cancel", "epoch": 3}).encode())
    assert control.take() == {"mode": "cancel", "epoch": 3}
    assert control.take() is None


@pytest.mark.parametrize(
    "command",
    [
        {"mode": "context", "epoch": True, "text": "Friday"},
        {"mode": "context", "epoch": 1, "text": "x" * 601},
        {"mode": "context", "epoch": 1, "text": ""},
        {"mode": "unknown", "epoch": 1, "text": "Friday"},
        {"mode": "context", "epoch": 1, "text": "Friday", "pace": 100},
        [],
    ],
)
def test_bad_control_cannot_mutate_queue(command):
    from speaker_lab.scheduling.native_control import NativeControl

    control = NativeControl()
    with pytest.raises(ValueError):
        control.submit(json.dumps(command).encode())
    assert control.epoch == 0 and control.take() is None


def test_response_prefix_is_bounded_and_kept_with_its_control_epoch():
    from speaker_lab.scheduling.native_control import NativeControl

    c = NativeControl()
    assert c.submit(
        json.dumps(
            {
                "mode": "context",
                "epoch": 2,
                "text": "Friday is available",
                "prefix": "The",
            }
        ).encode()
    )
    assert c.take()["prefix"] == "The"
    with pytest.raises(ValueError):
        c.submit(
            json.dumps(
                {
                    "mode": "context",
                    "epoch": 3,
                    "text": "Friday is available",
                    "prefix": "x" * 601,
                }
            ).encode()
        )
    assert c.epoch == 2


def test_fresh_model_context_control_preserves_epoch_and_payload():
    from speaker_lab.scheduling.native_control import NativeControl

    c = NativeControl()
    assert c.submit(
        json.dumps(
            {
                "mode": "reset",
                "epoch": 4,
                "text": "Current confirmed Friday slot",
                "prefix": "Your",
            }
        ).encode()
    )
    assert c.take()["mode"] == "reset"


def test_response_prefix_keeps_subwords_then_padding_and_word_boundary():
    from speaker_lab.scheduling.native_control import response_prefix_tokens

    class Tokenizer:
        def encode(self, text):
            return [11, 12, 13]

        def id_to_piece(self, token):
            return {11: "▁Your", 12: "▁appoint", 13: "ment"}[token]

    assert response_prefix_tokens(Tokenizer(), "Your appointment") == [
        11,
        3,
        3,
        0,
        12,
        13,
        3,
        0,
    ]


def test_persona_cache_prefix_matches_real_tokens_without_trailing_space_token():
    from speaker_lab.scheduling.native_control import TOOL_PERSONA, tool_persona_tokens

    class Tokenizer:
        def encode(self, text):
            assert text == '<system> ' + TOOL_PERSONA.rstrip()
            return [10, 11]

    assert tool_persona_tokens(Tokenizer(), [10, 11, 12]) == [10, 11]
    assert tool_persona_tokens(Tokenizer(), [10, 99, 12]) == []


def test_identifier_pacing_holds_each_letter_and_zero_without_repeating_text():
    from speaker_lab.scheduling.native_control import response_prefix_tokens
    class Tokenizer:
        def encode(self, text): return [11, 12, 13, 13, 13, 14]
        def id_to_piece(self, token): return {11: "▁B", 12: "▁K", 13: "▁zero", 14: "▁one"}[token]
    tokens = response_prefix_tokens(Tokenizer(), 'B K zero zero zero one', pace=5)
    assert [t for t in tokens if t not in (0, 3)] == [11, 12, 13, 13, 13, 14]
    assert tokens == [11,3,3,3,3,0,12,3,3,3,3,0]+[13,3,3,3,3,0]*3+[14,3,3,0]


def test_date_pacing_keeps_tokens_and_extends_only_calendar_words():
    from speaker_lab.scheduling.native_control import response_prefix_tokens
    class Tokenizer:
        def encode(self, text): return [11,16,12,13,14,15,17]
        def id_to_piece(self, token):
            return {11:'▁Tuesday',16:'▁October',12:'▁twent',13:'ieth,',14:'▁at',15:'▁four',17:'▁thirty'}[token]
    slow = response_prefix_tokens(Tokenizer(), 'Tuesday October twentieth, at four thirty', date_pace=5)
    assert slow == [11,3,3,0,16,3,3,0,12,13,3,3,3,0,14,3,3,0,15,3,3,0,17,3,3,0]
    assert [t for t in slow if t not in (0,3)] == [11,16,12,13,14,15,17]


@pytest.mark.parametrize('date_pace', [True, 2, 8, '5'])
def test_invalid_date_pacing_rejected_before_mutating_epoch(date_pace):
    from speaker_lab.scheduling.native_control import NativeControl
    c = NativeControl()
    with pytest.raises(ValueError):
        c.submit(json.dumps(dict(mode='read_cached', epoch=1,text='Tuesday',date_pace=date_pace)).encode())
    assert c.epoch == 0 and c.take() is None


def test_weekday_extension_preserves_all_following_timing():
    from speaker_lab.scheduling.native_control import response_prefix_tokens
    class Tokenizer:
        def encode(self, text): return [11,12]
        def id_to_piece(self, token): return {11:'▁Tuesday',12:'▁morning'}[token]
    original = response_prefix_tokens(Tokenizer(), 'Tuesday morning')
    longer = response_prefix_tokens(Tokenizer(), 'Tuesday morning', weekday_pace=5)
    assert original[:4] == [11,3,3,0]
    assert longer[:6] == [11,3,3,3,3,0] and longer[6:] == original[4:]


@pytest.mark.parametrize('weekday_pace', [True,2,8,'5'])
def test_invalid_weekday_pacing_does_not_accept_epoch(weekday_pace):
    from speaker_lab.scheduling.native_control import NativeControl
    c = NativeControl()
    with pytest.raises(ValueError):
        c.submit(json.dumps(dict(mode='read_cached',epoch=1,text='Tuesday',weekday_pace=weekday_pace)).encode())
    assert c.epoch == 0
