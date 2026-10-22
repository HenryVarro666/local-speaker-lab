"""Loopback-only Web Audio gateway and versioned simulated booking controller."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from datetime import date
import json
import logging
from pathlib import Path
import re
import struct
import time
from urllib.parse import urlsplit

from aiohttp import web, WSMsgType

from .audio import VoskObserver, UtteranceBuffer, validate_pcm
from .domain import BookingStore, SchedulingSession
from .personaplex import PersonaPlexClient
from .native_control import TOOL_PERSONA

ACTIVE = web.AppKey("active", object)
MODEL = web.AppKey("model", object)
SETTINGS = web.AppKey("settings", dict)


class Gateway:
    def __init__(self, ws, settings, observer_model):
        self.ws = ws
        self.settings = settings
        self.domain = SchedulingSession(BookingStore(), today=settings["today"])
        self.observer = VoskObserver(observer_model) if observer_model else None
        self.epoch = 0
        self.suppressed = True
        self.correction = False
        self.closed = False
        self.query_tasks = set()
        self.refresh_task = None
        self.sent_events = 0
        self.started = time.monotonic()
        self.pending = deque()
        self.pending_samples = 0
        self.dropped_samples = 0
        self.model_text = ""
        self.capture = UtteranceBuffer()
        self.last_utterance = None
        self.replaying = False
        self.voiced_epoch = None
        self.native_playback = False
        self.client = (
            PersonaPlexClient(settings["model_url"], self.on_audio, self.on_model_event)
            if settings["model_url"]
            else None
        )
        self.logfile = None
        if settings["event_dir"]:
            folder = settings["event_dir"] / self.domain.id
            folder.mkdir(parents=True, exist_ok=False)
            self.logfile = (folder / "events.jsonl").open("w", encoding="utf-8")

    async def emit(self, kind, **data):
        event = {
            "kind": kind,
            "elapsed_ms": round((time.monotonic() - self.started) * 1000, 3),
            **data,
        }
        if self.logfile:
            self.logfile.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.logfile.flush()
        if not self.ws.closed:
            await self.ws.send_json(event)

    async def publish(self):
        for event in self.domain.events[self.sent_events :]:
            await self.emit("tool_event", event=event)
        self.sent_events = len(self.domain.events)
        await self.emit("state", state=self.domain.snapshot())

    async def invalidate(self, reason):
        self.epoch += 1
        self.suppressed = True
        await self.emit("playback_reset", epoch=self.epoch, reason=reason)

    async def on_audio(self, pcm, epoch):
        if not self.suppressed and epoch == self.epoch and not self.ws.closed:
            if float((pcm**2).mean()) > 0.008**2 and self.voiced_epoch != epoch:
                self.voiced_epoch = epoch
                await self.emit(
                    "native_voice_started",
                    epoch=epoch,
                    boundary="gateway voiced PCM; excludes browser/hardware",
                )
            await self.ws.send_bytes(
                struct.pack("<I", epoch) + pcm.astype("<f4").tobytes()
            )

    async def on_model_event(self, event):
        if event.get("epoch") != self.epoch:
            return
        kind = event.pop("kind")
        if kind == "native_control" and event.get("phase") == "applied":
            self.suppressed = False
        if kind == "model_text":
            self.model_text = (self.model_text + event["text"])[-500:]
        await self.emit(kind, **event)

    async def refresh(self, reason):
        if not self.client or self.closed:
            self.suppressed = False
            return
        if self.refresh_task and not self.refresh_task.done():
            self.refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.refresh_task
        await self.invalidate(reason)
        epoch = self.epoch
        if (
            reason != "session_start"
            and self.settings["tool_voice"] != "device"
            and "cancel" in self.client.capabilities.get("modes", [])
        ):
            await self.client.control("cancel", "", epoch)
            await self.emit(
                "context_update", epoch=epoch, reason=reason, strategy="in_place_cancel"
            )
            return
        prompt = self.domain.prompt()
        replay = self.last_utterance

        async def run():
            try:
                await self.client.connect(prompt, epoch)
                if epoch == self.epoch and not self.closed:
                    self.model_text = ""
                    self.suppressed = False
                    if replay is not None and len(replay):
                        self.replaying = True
                        await self.emit(
                            "acoustic_context_replay",
                            epoch=epoch,
                            samples=len(replay),
                            sample_rate=24000,
                        )
                        for i in range(0, len(replay), 1920):
                            await self.client.feed(replay[i : i + 1920])
                            await asyncio.sleep(0.005)
                        self.replaying = False
                    pending = list(self.pending)
                    self.pending.clear()
                    self.pending_samples = 0
                    await self.emit(
                        "context_refresh",
                        epoch=epoch,
                        reason=reason,
                        buffered_samples=sum(len(p) for p in pending),
                        dropped_samples=self.dropped_samples,
                    )
                    self.dropped_samples = 0
                    for pcm in pending:
                        await self.client.feed(pcm)
            except asyncio.CancelledError:
                self.replaying = False
                raise
            except Exception as error:
                self.replaying = False
                logging.getLogger(__name__).exception("Voice context refresh failed")
                await self.emit(
                    "model_error",
                    error=type(error).__name__,
                    message="PersonaPlex unavailable. Tools remain diagnostic; reconnect after checking the GPU service.",
                    epoch=epoch,
                )

        self.refresh_task = asyncio.create_task(run())

    async def speak_tool_fact(self, purpose):
        text = (
            self.domain.proposal_text()
            if purpose == "proposal"
            else self.domain.receipt_text()
        )
        if not text:
            return
        fact = dict(
            self.domain.proposal if purpose == "proposal" else self.domain.booking
        )
        version = self.domain.version

        async def device_voice():
            await self.emit(
                purpose + "_readback",
                text=text,
                **{purpose: fact},
                version=version,
                source="tool_facts_for_browser_device_voice",
            )

        if (
            self.client
            and self.settings["tool_voice"] != "device"
            and any(
                mode in self.client.capabilities.get("modes", [])
                for mode in ("read_cached", "reset_cached", "read", "reset", "context")
            )
        ):
            modes = (
                ("read_cached", "read", "reset_cached", "reset", "context")
                if self.settings["tool_voice"] == "native-controlled"
                else ("reset_cached", "reset", "read_cached", "read", "context")
            )
            native_mode = next(
                mode
                for mode in modes
                if mode in self.client.capabilities.get("modes", [])
            )
            opening = (
                self.domain.tool_speech(purpose)
                if native_mode in ("read", "read_cached")
                else self.domain.tool_opening(purpose)
            )
            self.native_playback = True
            await self.invalidate("native_" + purpose)
            epoch = self.epoch
            await self.emit(
                "native_readback",
                purpose=purpose,
                text=text,
                **{purpose: fact},
                version=version,
                epoch=epoch,
                source="PersonaPlex_controlled_native_readback"
                if native_mode in ("read", "read_cached")
                else "PersonaPlex_guided_native_reply",
                response_prefix=opening,
                model_context=native_mode,
                language_generation="tool_text_conditioned_native_audio"
                if native_mode in ("read", "read_cached")
                else "sampled_after_fixed_opening",
            )
            try:
                context = self.domain.tool_context(purpose)
                mode = native_mode
                if mode in ("reset", "read", "reset_cached", "read_cached"):
                    context = (
                        TOOL_PERSONA
                        + context
                    )
                await self.client.control(
                    mode,
                    context,
                    epoch,
                    pace=5 if purpose == "receipt" else 3,
                    prefix=opening,
                )
            except (ConnectionError, RuntimeError):
                await device_voice()
                return

            async def watchdog():
                await asyncio.sleep(self.settings["native_timeout"])
                if (
                    not self.closed
                    and epoch == self.epoch
                    and self.voiced_epoch != epoch
                ):
                    self.suppressed = True
                    await self.emit(
                        "native_fallback",
                        epoch=epoch,
                        purpose=purpose,
                        reason="no_voiced_pcm",
                    )
                    await device_voice()

            task = asyncio.create_task(watchdog())
            self.query_tasks.add(task)
            task.add_done_callback(self.query_tasks.discard)
        else:
            await self.refresh(purpose + "_device_fallback")
            await device_voice()

    async def input_text(self, text, source):
        before = (self.domain.version, self.domain.status)
        receipts_before = len(self.domain.store.operations)
        self.domain.hear(text, defer_query=True)
        await self.emit("transcript", text=text, source=source)
        await self.publish()
        after = (self.domain.version, self.domain.status)
        if before != after:
            await self.invalidate("business_state_changed")
        if self.domain.status == "checking" and before[0] != after[0]:
            version = self.domain.version
            start = self.domain.proposal["start"]

            async def query():
                await asyncio.sleep(self.settings["query_delay"])
                available = self.domain.store.available(
                    start, self.domain.booking["id"] if self.domain.booking else None
                )
                if self.domain.finish_query(version, available):
                    await self.publish()
                    if self.domain.proposal_text():
                        await self.speak_tool_fact("proposal")
                    else:
                        await self.refresh("availability_result")
                else:
                    await self.publish()

            task = asyncio.create_task(query())
            self.query_tasks.add(task)
            task.add_done_callback(self.query_tasks.discard)
        elif (
            self.domain.status != "checking"
            and (before != after or self.correction)
            and len(self.domain.store.operations) == receipts_before
        ):
            await self.refresh("tool_receipt_or_withdrawal")
        self.correction = False
        if len(self.domain.store.operations) > receipts_before:
            await self.speak_tool_fact("receipt")

    async def input_audio(self, payload):
        pcm = validate_pcm(payload)
        self.capture.feed(pcm)
        if self.client:
            if self.client.ready and not self.replaying:
                await self.client.feed(pcm)
            else:
                self.pending.append(pcm.copy())
                self.pending_samples += len(pcm)
                while self.pending_samples > 24000 and self.pending:
                    removed = self.pending.popleft()
                    self.pending_samples -= len(removed)
                    self.dropped_samples += len(removed)
        if self.observer:
            events = await asyncio.to_thread(self.observer.feed, pcm)
            for event in events:
                if event["kind"] == "observer_final":
                    self.last_utterance = self.capture.finish()
                    await self.input_text(event["text"], "vosk")
                else:
                    await self.emit("partial", text=event["text"], source="vosk")
                    if self.native_playback and event["text"].lower() not in (
                        "uh huh",
                        "yes",
                        "okay",
                        "ok",
                        "mhm",
                        "hmm",
                        "right",
                    ):
                        self.native_playback = False
                        await self.refresh("user_speech")
                    if not self.correction and re.search(
                        r"\b(actually|wait|instead|change|move|never mind)\b",
                        event["text"],
                    ):
                        self.correction = True
                        self.domain.interrupt_correction()
                        await self.publish()
                        await self.invalidate("correction_partial")

    async def start(self):
        await self.emit(
            "session",
            id=self.domain.id,
            mode="live" if self.client else "diagnostic",
            observer_ready=bool(self.observer),
            sample_rate=24000,
            context_strategy="in_place_native_tools_when_supported; reconnect_fallback",
        )
        await self.publish()
        await self.refresh("session_start")
        try:
            async for message in self.ws:
                try:
                    if message.type == WSMsgType.BINARY:
                        await self.input_audio(message.data)
                    elif message.type == WSMsgType.TEXT:
                        command = json.loads(message.data)
                        if not isinstance(command, dict):
                            raise ValueError("Expected command object")
                        kind = command.get("kind")
                        if kind == "text":
                            text = command.get("text")
                            if not isinstance(text, str) or not 1 <= len(text) <= 500:
                                raise ValueError(
                                    "Expected 1–500 characters of diagnostic text"
                                )
                            await self.input_text(text, "diagnostic_text")
                        elif kind == "confirm":
                            version = command.get("version")
                            if type(version) is not int:
                                raise ValueError("Expected integer version")
                            receipts_before = len(self.domain.store.operations)
                            self.domain.confirm(version)
                            await self.publish()
                            if len(self.domain.store.operations) > receipts_before:
                                await self.speak_tool_fact("receipt")
                        elif kind == "readback_status":
                            if command.get("phase") not in (
                                "started",
                                "ended",
                                "error",
                                "unavailable",
                            ):
                                raise ValueError("Invalid readback phase")
                            await self.emit(
                                "readback_status",
                                phase=command["phase"],
                                version=command.get("version"),
                                purpose=command.get("purpose", "receipt"),
                                source="browser_speech_synthesis",
                            )
                        elif kind == "native_playback_status":
                            if (
                                command.get("phase")
                                not in (
                                    "first_voiced_chunk_scheduled",
                                    "first_voiced_chunk_ended",
                                )
                                or type(command.get("epoch")) is not int
                            ):
                                raise ValueError("Invalid native playback status")
                            if self.client and command["epoch"] == self.epoch:
                                await self.emit(
                                    "native_playback_status",
                                    phase=command["phase"],
                                    epoch=command["epoch"],
                                    source="browser_audio_context",
                                )
                        elif kind == "snapshot":
                            await self.publish()
                        elif kind == "refresh":
                            await self.refresh("manual_refresh")
                        else:
                            raise ValueError("Unsupported command")
                    elif message.type == WSMsgType.ERROR:
                        break
                except (ValueError, KeyError) as error:
                    await self.emit("error", message=str(error))
        finally:
            self.closed = True
            tasks = list(self.query_tasks)
            if self.refresh_task:
                tasks.append(self.refresh_task)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.client:
                await self.client.close()
            if self.logfile:
                self.logfile.close()


def create_app(
    *,
    model_url="http://127.0.0.1:8998",
    vosk_path=None,
    today: date | None = None,
    query_delay=0.6,
    event_dir: Path | None = None,
    tool_voice="native-controlled",
    native_timeout=8.0,
):
    app = web.Application(client_max_size=200000)
    app[SETTINGS] = {
        "model_url": model_url,
        "today": today,
        "query_delay": query_delay,
        "event_dir": event_dir,
        "tool_voice": tool_voice,
        "native_timeout": native_timeout,
    }
    app[ACTIVE] = {"gateway": None}
    app[MODEL] = VoskObserver.load(Path(vosk_path)) if vosk_path else None

    async def websocket(request):
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            raise web.HTTPForbidden(text="Origin is not allowed")
        if app[ACTIVE]["gateway"] is not None:
            raise web.HTTPConflict(
                text="This single-user Demo already has an active session"
            )
        ws = web.WebSocketResponse(max_msg_size=200000, heartbeat=20)
        gateway = Gateway(ws, app[SETTINGS], app[MODEL])
        app[ACTIVE]["gateway"] = gateway
        try:
            await ws.prepare(request)
            await gateway.start()
        finally:
            app[ACTIVE]["gateway"] = None
        return ws

    async def health(request):
        active = app[ACTIVE]["gateway"]
        return web.json_response(
            {
                "mode": "live" if model_url else "diagnostic",
                "model_ready": bool(active and active.client and active.client.ready),
                "observer_ready": app[MODEL] is not None,
                "simulated_tools": True,
            }
        )

    async def page(request):
        return web.Response(
            text=Path(__file__).with_name("page.html").read_text(encoding="utf-8"),
            content_type="text/html",
        )

    async def cleanup(application):
        active = application[ACTIVE]["gateway"]
        if active:
            await active.ws.close()

    app.router.add_get("/", page)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/health", health)
    app.on_shutdown.append(cleanup)
    return app
