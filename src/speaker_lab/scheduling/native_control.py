"""Bounded experimental controls for the locally extended PersonaPlex backend."""

from __future__ import annotations
import json

TOOL_PERSONA = "You work for Lakeside Appointment Desk and your name is Alex. Information: "

TOOL_CONTEXT_PREFIX = TOOL_PERSONA + "The conversation is already in progress. Tool result:"

def tool_persona_tokens(tokenizer, full_prompt_tokens):
    # SentencePiece emits an extra standalone space token for a trailing space.
    # Reuse only a token-identical prefix; arbitrary personas use voice-only cache.
    prefix = tokenizer.encode("<system> " + TOOL_PERSONA.rstrip())
    return prefix if full_prompt_tokens[: len(prefix)] == prefix else []


class NativeControl:
    def __init__(self):
        self.epoch = 0
        self.pending = None

    def submit(self, payload: bytes) -> bool:
        if len(payload) > 4096:
            raise ValueError("Native control exceeds limit")
        try:
            command = json.loads(payload)
        except (ValueError, UnicodeError) as error:
            raise ValueError("Invalid native control JSON") from error
        if not isinstance(command, dict):
            raise ValueError("Expected native control object")
        mode, epoch = command.get("mode"), command.get("epoch")
        if (
            mode not in ("context", "reset", "read", "reset_cached", "read_cached", "cancel")
            or type(epoch) is not int
            or epoch < 1
        ):
            raise ValueError("Invalid native mode or epoch")
        if mode != "cancel" and (
            not isinstance(command.get("text"), str)
            or not 1 <= len(command["text"]) <= 600
        ):
            raise ValueError("Expected 1-600 characters of tool text")
        pace = command.get("pace", 3)
        if type(pace) is not int or not 1 <= pace <= 5:
            raise ValueError("Invalid native text pacing")
        date_pace = command.get("date_pace", 3)
        if type(date_pace) is not int or not 3 <= date_pace <= 7:
            raise ValueError("Invalid native date pacing")
        weekday_pace = command.get("weekday_pace", 3)
        if type(weekday_pace) is not int or not 3 <= weekday_pace <= 7:
            raise ValueError("Invalid native weekday pacing")
        prefix = command.get("prefix", "")
        if not isinstance(prefix, str) or len(prefix) > 600:
            raise ValueError("Native response prefix exceeds limit")
        if epoch <= self.epoch:
            return False
        self.epoch, self.pending = epoch, command
        return True

    def take(self):
        command, self.pending = self.pending, None
        return command


WEEKDAYS = frozenset("monday tuesday wednesday thursday friday saturday sunday".split())
MONTHS = frozenset("january february march april may june july august september october november december".split())
DATE_WORDS = frozenset("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth twenty thirty".split())


def response_prefix_tokens(tokenizer, text, pace=3, date_pace=3, weekday_pace=3):
    """Short reply opening only: PAD sustains a word; EPAD closes its gap."""
    words = []
    for token in tokenizer.encode(text):
        if not words or tokenizer.id_to_piece(token).startswith("▁"):
            words.append([])
        words[-1].append(token)
    output = []
    date_region = False
    for word in words:
        spelling = "".join(tokenizer.id_to_piece(t) for t in word).replace("▁", "").lower()
        lexical = spelling.strip(".,!?;:")
        if lexical in MONTHS:
            date_region = True
        elif lexical == "at":
            date_region = False
        if spelling in ("b", "k", "zero"):
            width = pace
        elif lexical in WEEKDAYS:
            width = weekday_pace
        elif date_region and lexical in DATE_WORDS:
            width = date_pace
        else:
            width = 3
        output.extend(word + [3] * max(0, width - len(word)) + [0])
    return output
