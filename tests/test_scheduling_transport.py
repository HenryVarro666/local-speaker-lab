"""The upstream fixture verifies the actual wire codec, not model quality."""

import asyncio
import numpy as np
import pytest

aiohttp = pytest.importorskip("aiohttp")
sphn = pytest.importorskip("sphn")


def test_real_opus_transport_refresh_and_stale_audio_epoch():
    async def run():
        from aiohttp import web
        from speaker_lab.scheduling.personaplex import PersonaPlexClient

        prompts, audio, messages, params_seen = [], [], [], []

        async def handler(request):
            prompts.append(request.query["text_prompt"])
            params_seen.append(dict(request.query))
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_bytes(b"\x00")
            reader = sphn.OpusStreamReader(24000)
            writer = sphn.OpusStreamWriter(24000)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY and msg.data[:1] == b"\x01":
                    reader.append_bytes(msg.data[1:])
                    pcm = reader.read_pcm()
                    if len(pcm):
                        writer.append_pcm(pcm)
                        encoded = writer.read_bytes()
                        if encoded:
                            await ws.send_bytes(b"\x01" + encoded)
                        await ws.send_bytes(b"\x02Hello")
            return ws

        app = web.Application()
        app.router.add_get("/api/chat", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        async def on_audio(pcm, epoch):
            audio.append((pcm, epoch))

        async def on_event(event):
            messages.append(event)

        client = PersonaPlexClient(f"http://127.0.0.1:{port}", on_audio, on_event)
        try:
            await client.connect("first prompt", 1)
            for _ in range(12):
                await client.feed(np.full(1920, 0.05, dtype=np.float32))
                await asyncio.sleep(
                    0.01
                )  # Opus uses a worker thread; input is streamed.
            await client.feed(
                np.full(1440, 0.05, dtype=np.float32)
            )  # Replay tail between codec frame sizes.
            await asyncio.sleep(0.1)
            assert any(epoch == 1 and len(pcm) > 0 for pcm, epoch in audio)
            assert any(e.get("text") == "Hello" for e in messages)
            await client.connect("updated tool facts", 2, seed=42)
            before = len(audio)
            for _ in range(12):
                await client.feed(np.full(1920, 0.05, dtype=np.float32))
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)
            assert prompts == ["first prompt", "updated tool facts"]
            assert "seed" not in params_seen[0] and params_seen[1]["seed"] == "42"
            assert audio[before:] and all(epoch == 2 for pcm, epoch in audio[before:])
        finally:
            await client.close()
            await runner.cleanup()

    asyncio.run(run())


def test_handshake_failure_reports_disconnected_and_leaves_no_reader():
    async def run():
        from aiohttp import web
        from speaker_lab.scheduling.personaplex import PersonaPlexClient

        async def wrong(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_bytes(b"\x02Not a handshake")
            return ws

        app = web.Application()
        app.router.add_get("/api/chat", wrong)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        async def ignore(*args):
            pass

        client = PersonaPlexClient(f"http://127.0.0.1:{port}", ignore, ignore)
        try:
            with pytest.raises(ConnectionError, match="handshake"):
                await client.connect("prompt", 1)
            assert not client.ready
            await client.feed(np.zeros(1920, dtype=np.float32))
        finally:
            await client.close()
            await runner.cleanup()

    asyncio.run(run())


def test_pcm_validation_prevents_nan_and_oversized_input():
    from speaker_lab.scheduling.audio import validate_pcm

    with pytest.raises(ValueError):
        validate_pcm(b"123")
    with pytest.raises(ValueError):
        validate_pcm(np.array([float("nan")], dtype="<f4").tobytes())
    with pytest.raises(ValueError):
        validate_pcm(np.zeros(48001, dtype="<f4").tobytes())
    assert validate_pcm(np.zeros(480, dtype="<f4").tobytes()).shape == (480,)


def test_utterance_replay_retains_speech_but_is_bounded_and_cleared():
    from speaker_lab.scheduling.audio import UtteranceBuffer

    buffer = UtteranceBuffer(max_samples=2400)
    buffer.feed(np.zeros(480, dtype=np.float32))
    buffer.feed(np.full(480, 0.1, dtype=np.float32))
    buffer.feed(np.zeros(480, dtype=np.float32))
    clip = buffer.finish()
    assert len(clip) == 1440 and np.max(clip) > 0.09
    assert len(buffer.finish()) == 0
    for _ in range(100):
        buffer.feed(np.full(480, 0.1, dtype=np.float32))
    assert len(buffer.finish()) <= 2400


def test_native_control_ack_changes_epoch_and_resets_codec_without_reconnecting(
    monkeypatch,
):
    real_reader = sphn.OpusStreamReader

    class NullableReader:
        # Linux sphn 0.1.12 returns None until a new Ogg stream has data.
        def __init__(self, rate):
            self.inner = real_reader(rate)

        def append_bytes(self, payload):
            return self.inner.append_bytes(payload)

        def read_pcm(self):
            pcm = self.inner.read_pcm()
            return pcm if pcm is not None and len(pcm) else None

    monkeypatch.setattr(sphn, "OpusStreamReader", NullableReader)

    async def run():
        import json
        from aiohttp import web
        from speaker_lab.scheduling.personaplex import PersonaPlexClient

        connections = []
        events = []
        audio = []
        controls = []

        async def backend(request):
            connections.append(1)
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_bytes(b"\x00")
            await ws.send_bytes(
                b"\x03"
                + json.dumps(
                    {
                        "kind": "native_capabilities",
                        "protocol": 1,
                        "modes": ["context", "cancel"],
                    }
                ).encode()
            )
            epoch = 1
            writer = sphn.OpusStreamWriter(24000)
            reader = sphn.OpusStreamReader(24000)
            async for msg in ws:
                if msg.data[:1] == b"\x03":
                    control = json.loads(msg.data[1:])
                    controls.append(control)
                    epoch = control["epoch"]
                    writer = sphn.OpusStreamWriter(24000)
                    await ws.send_bytes(
                        b"\x03"
                        + json.dumps(
                            {
                                "kind": "native_control",
                                "phase": "applied",
                                "epoch": epoch,
                                "mode": control["mode"],
                            }
                        ).encode()
                    )
                    await asyncio.sleep(
                        0.08
                    )  # A new codec may return None before its first packet.
                elif msg.data[:1] == b"\x01":
                    reader.append_bytes(msg.data[1:])
                    pcm = reader.read_pcm()
                    if pcm is not None and len(pcm):
                        writer.append_pcm(pcm)
                        encoded = writer.read_bytes()
                        if encoded:
                            await ws.send_bytes(b"\x01" + encoded)
            return ws

        app = web.Application()
        app.router.add_get("/api/chat", backend)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()

        async def on_audio(pcm, epoch):
            audio.append(epoch)

        async def on_event(event):
            events.append(dict(event))

        client = PersonaPlexClient(
            f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
            on_audio,
            on_event,
        )
        try:
            await client.connect("ongoing appointment call", 1)
            for _ in range(15):
                await client.feed(np.full(480, 0.05, dtype=np.float32))
                await asyncio.sleep(0.01)
            assert any(e["kind"] == "native_capabilities" for e in events)
            await client.control("context", "Friday at four thirty.", 2, date_pace=5, weekday_pace=5)
            for _ in range(30):
                await client.feed(np.full(480, 0.05, dtype=np.float32))
                await asyncio.sleep(0.01)
            assert len(connections) == 1
            assert controls[0]["text"] == "Friday at four thirty."
            assert controls[0]["date_pace"] == controls[0]["weekday_pace"] == 5
            assert any(e.get("phase") == "applied" and e["epoch"] == 2 for e in events)
            assert audio[-1] == 2
        finally:
            await client.close()
            await runner.cleanup()
        assert any(
            e["kind"] == "model_disconnected" and e["epoch"] == 2 for e in events
        )

    asyncio.run(run())
