"""Client for the official PersonaPlex WebSocket protocol, with explicit refresh."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from urllib.parse import urlsplit
import numpy as np


class PersonaPlexClient:
    def __init__(self, url, on_audio, on_event, *, voice="NATF2.pt"):
        parts = urlsplit(url)
        if (
            parts.scheme not in ("http", "https")
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError(
                "Expected an HTTP backend URL without credentials, query or fragment"
            )
        self.url = url.rstrip("/")
        self.on_audio = on_audio
        self.on_event = on_event
        self.voice = voice
        self.ready = False
        self._session = None
        self._ws = None
        self._reader = None
        self._writer = None
        self._lock = asyncio.Lock()
        self.capabilities = {}
        self._output_epoch = 0

    async def _disconnect(self):
        self.ready = False
        if self._reader:
            self._reader.cancel()
            with suppress(asyncio.CancelledError, ConnectionError):
                await self._reader
            self._reader = None
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def connect(self, prompt: str, epoch: int, *, seed=None):
        import aiohttp
        import sphn

        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("Seed must be a uint32 or None")
        params = {"text_prompt": prompt, "voice_prompt": self.voice}
        if seed is not None:
            params["seed"] = str(seed)
        async with self._lock:
            await self._disconnect()
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=10)
                )
            try:
                self._ws = await self._session.ws_connect(
                    self.url + "/api/chat",
                    params=params,
                    max_msg_size=2**20,
                )
                first = await self._ws.receive(timeout=45)
                if first.type != aiohttp.WSMsgType.BINARY or first.data != b"\x00":
                    raise ConnectionError("PersonaPlex handshake was not received")
                self._writer = sphn.OpusStreamWriter(24000)
                self.capabilities = {}
                self._output_epoch = epoch
                self.ready = True
                self._reader = asyncio.create_task(self._read(self._ws, epoch))
                await self.on_event(
                    {
                        "kind": "model_ready",
                        "epoch": epoch,
                        "model": "PersonaPlex",
                        "context": "reset_with_tool_facts",
                    }
                )
            except BaseException:
                await self._disconnect()
                raise

    async def _read(self, ws, epoch):
        import aiohttp
        import sphn

        reader = sphn.OpusStreamReader(24000)
        try:
            while not ws.closed:
                try:
                    message = await ws.receive(timeout=0.01)
                except asyncio.TimeoutError:
                    message = None
                if message is None:
                    pcm = reader.read_pcm()
                    if pcm is not None and len(pcm):
                        await self.on_audio(pcm, self._output_epoch)
                    continue
                if message.type == aiohttp.WSMsgType.BINARY:
                    kind, payload = message.data[:1], message.data[1:]
                    if kind == b"\x01":
                        reader.append_bytes(payload)
                        pcm = reader.read_pcm()
                        if pcm is not None and len(pcm):
                            await self.on_audio(pcm, self._output_epoch)
                    elif kind == b"\x02":
                        await self.on_event(
                            {
                                "kind": "model_text",
                                "text": payload.decode("utf-8"),
                                "epoch": self._output_epoch,
                            }
                        )
                    elif kind == b"\x03":
                        event = json.loads(payload)
                        if event.get("kind") == "native_capabilities":
                            self.capabilities = event
                            event["epoch"] = self._output_epoch
                        elif event.get("phase") == "applied":
                            self._output_epoch = event["epoch"]
                            reader = sphn.OpusStreamReader(24000)
                        else:
                            event["epoch"] = self._output_epoch
                        await self.on_event(event)
                elif message.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError("PersonaPlex stream failed")
                elif message.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.on_event(
                {
                    "kind": "model_error",
                    "error": type(error).__name__,
                    "message": str(error),
                    "epoch": self._output_epoch,
                }
            )
        finally:
            if ws is self._ws:
                self.ready = False
                await self.on_event(
                    {"kind": "model_disconnected", "epoch": self._output_epoch}
                )

    async def feed(self, pcm: np.ndarray):
        if not self.ready or self._ws is None:
            return False
        samples = np.asarray(pcm, dtype=np.float32)
        frames = (
            [samples]
            if len(samples) in (120, 240, 480, 960, 1920, 2880)
            else [
                np.pad(
                    samples[i : i + 480], (0, max(0, 480 - len(samples[i : i + 480])))
                )
                for i in range(0, len(samples), 480)
            ]
        )
        for frame in frames:
            self._writer.append_pcm(frame)
            encoded = self._writer.read_bytes()
            if encoded:
                try:
                    await self._ws.send_bytes(b"\x01" + encoded)
                except (ConnectionError, RuntimeError):
                    self.ready = False
                    return False
        return True

    async def control(self, mode, text, epoch, *, pace=3, prefix=None, date_pace=None, weekday_pace=None):
        if not self.ready or mode not in self.capabilities.get("modes", []):
            raise ConnectionError("Backend does not support native tool controls")
        command = {"mode": mode, "epoch": epoch, "pace": pace}
        if date_pace is not None:
            command["date_pace"] = date_pace
        if weekday_pace is not None:
            command["weekday_pace"] = weekday_pace
        if mode != "cancel":
            command["text"] = text
        if prefix is not None:
            command["prefix"] = prefix
        from .native_control import NativeControl

        NativeControl().submit(json.dumps(command).encode())
        await self._ws.send_bytes(b"\x03" + json.dumps(command).encode())

    async def close(self):
        async with self._lock:
            await self._disconnect()
            if self._session:
                await self._session.close()
                self._session = None
