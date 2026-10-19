"""Remove model credentials before diagnostic text reaches disk or the console."""
from __future__ import annotations

import os
import re
import threading


class Redactor:
    def __init__(self, options=None, *, environment=None):
        environment = os.environ if environment is None else environment
        names = {"HF_TOKEN"}

        def token_names(value):
            if isinstance(value, dict):
                name = value.get("token_env")
                if isinstance(name, str):
                    names.add(name)
                for item in value.values():
                    token_names(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    token_names(item)

        token_names(options)
        self.secrets = tuple(sorted({environment[name] for name in names if environment.get(name)},
                                    key=len, reverse=True))

    def text(self, value):
        text = str(value)
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return re.sub(r"hf_[A-Za-z0-9]{8,}", "[REDACTED]", text)

    def value(self, value):
        """Redact Python values before JSON escaping, preserving valid JSON."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.text(key) if isinstance(key, str) else key: self.value(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.value(item) for item in value)
        return value


class RedactingStream:
    """Line-buffer a text stream so split writes cannot expose a partial token.

    ``flush`` leaves an incomplete line buffered; ``finish`` emits its redacted
    tail. Carriage returns also delimit progress-bar updates. The parent runner
    separately filters native-library output before writing its worker log.
    """

    def __init__(self, stream, redactor):
        self.stream, self.redactor = stream, redactor
        self.pending = ""
        self.lock = threading.RLock()

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text):
        with self.lock:
            self.pending += text
            last = max(self.pending.rfind("\n"), self.pending.rfind("\r"))
            if last >= 0:
                self.stream.write(self.redactor.text(self.pending[:last + 1]))
                self.pending = self.pending[last + 1:]
        return len(text)

    def flush(self):
        with self.lock:
            self.stream.flush()

    def finish(self):
        with self.lock:
            if self.pending:
                self.stream.write(self.redactor.text(self.pending))
                self.pending = ""
            self.stream.flush()
