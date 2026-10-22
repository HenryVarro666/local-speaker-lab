"""Actual WebSocket controller flow; diagnostic text is never called speech evidence."""

import asyncio
from datetime import date
import pytest

aiohttp = pytest.importorskip("aiohttp")


async def start():
    from aiohttp import web
    from speaker_lab.scheduling.server import create_app

    app = create_app(model_url=None, today=date(2026, 10, 20), query_delay=0.05)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


async def until(ws, predicate):
    for _ in range(60):
        message = await ws.receive_json(timeout=3)
        if predicate(message):
            return message
    raise AssertionError("Expected event was not emitted")


def test_complete_create_correct_confirm_reschedule_flow():
    async def run():
        runner, url = await start()
        try:
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(url + "/ws") as ws:
                    await until(ws, lambda e: e["kind"] == "state")
                    await ws.send_json(
                        {"kind": "text", "text": "Book Thursday at three pm"}
                    )
                    await until(
                        ws,
                        lambda e: (
                            e["kind"] == "state" and e["state"]["status"] == "checking"
                        ),
                    )
                    await ws.send_json(
                        {"kind": "text", "text": "Actually Friday at four thirty"}
                    )
                    proposed = await until(
                        ws,
                        lambda e: (
                            e["kind"] == "state"
                            and e["state"]["status"] == "awaiting_confirmation"
                        ),
                    )
                    assert (
                        proposed["state"]["proposal"]["start"]
                        == "2026-10-23T16:30:00-05:00"
                    )
                    readback = await until(
                        ws, lambda e: e["kind"] == "proposal_readback"
                    )
                    assert (
                        "Friday" in readback["text"] and "4:30 PM" in readback["text"]
                    )
                    version = proposed["state"]["version"]
                    await ws.send_json({"kind": "confirm", "version": version - 1})
                    await until(ws, lambda e: e["kind"] == "error")
                    await ws.send_json({"kind": "confirm", "version": version})
                    booked = await until(
                        ws,
                        lambda e: (
                            e["kind"] == "state" and e["state"]["status"] == "booked"
                        ),
                    )
                    assert booked["state"]["booking"]["id"] == "BK-0001"
                    readback = await until(
                        ws, lambda e: e["kind"] == "receipt_readback"
                    )
                    assert (
                        "Friday" in readback["text"] and "4:30 PM" in readback["text"]
                    )
                    await ws.send_json(
                        {"kind": "text", "text": "Move it to Thursday at two pm"}
                    )
                    proposed = await until(
                        ws,
                        lambda e: (
                            e["kind"] == "state"
                            and e["state"]["status"] == "awaiting_confirmation"
                        ),
                    )
                    await ws.send_json(
                        {"kind": "confirm", "version": proposed["state"]["version"]}
                    )
                    booked = await until(
                        ws,
                        lambda e: (
                            e["kind"] == "state" and e["state"]["status"] == "booked"
                        ),
                    )
                    assert booked["state"]["booking"]["id"] == "BK-0001"
                    assert (
                        booked["state"]["booking"]["start"]
                        == "2026-10-22T14:00:00-05:00"
                    )
                    readback = await until(
                        ws, lambda e: e["kind"] == "receipt_readback"
                    )
                    assert (
                        "Thursday" in readback["text"] and "2:00 PM" in readback["text"]
                    )
                health = await (await client.get(url + "/api/health")).json()
                assert health["model_ready"] is False
                assert health["mode"] == "diagnostic"
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_foreign_origin_cannot_open_session_or_send_commands():
    async def run():
        runner, url = await start()
        try:
            async with aiohttp.ClientSession() as client:
                with pytest.raises(aiohttp.WSServerHandshakeError) as caught:
                    await client.ws_connect(
                        url + "/ws", headers={"Origin": "https://untrusted.example"}
                    )
                assert caught.value.status == 403
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_bad_messages_do_not_kill_session_or_create_booking():
    async def run():
        runner, url = await start()
        try:
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(url + "/ws") as ws:
                    await until(ws, lambda e: e["kind"] == "state")
                    for bad in [
                        "not json",
                        "[]",
                        '{"kind":"text","text":12}',
                        '{"kind":"confirm","version":true}',
                    ]:
                        await ws.send_str(bad)
                        await until(ws, lambda e: e["kind"] == "error")
                    await ws.send_json({"kind": "snapshot"})
                    state = await until(ws, lambda e: e["kind"] == "state")
                    assert state["state"]["booking"] is None
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_only_one_live_session_owns_model_and_disconnect_releases_it():
    async def run():
        runner, url = await start()
        try:
            async with aiohttp.ClientSession() as client:
                ws = await client.ws_connect(url + "/ws")
                await until(ws, lambda e: e["kind"] == "state")
                with pytest.raises(aiohttp.WSServerHandshakeError) as caught:
                    await client.ws_connect(url + "/ws")
                assert caught.value.status == 409
                await ws.close()
                await asyncio.sleep(0.05)
                ws2 = await client.ws_connect(url + "/ws")
                await ws2.close()
        finally:
            await runner.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["context", "reset", "read", "reset_cached", "read_cached"])
def test_native_tool_voice_uses_live_connection_and_silence_falls_back(mode):
    async def run():
        from aiohttp import web
        from speaker_lab.scheduling.server import create_app
        import json

        controls = []
        connections = []

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
                        "modes": ["reset_cached", "reset", "read", mode, "cancel"] if mode == "read_cached" else [mode, "cancel"],
                    }
                ).encode()
            )
            async for message in ws:
                if message.data[:1] == b"\x03":
                    command = json.loads(message.data[1:])
                    controls.append(command)
                    await ws.send_bytes(
                        b"\x03"
                        + json.dumps(
                            {"kind": "native_control", "phase": "applied", **command}
                        ).encode()
                    )
            return ws

        app = web.Application()
        app.router.add_get("/api/chat", backend)
        upstream = web.AppRunner(app)
        await upstream.setup()
        site = web.TCPSite(upstream, "127.0.0.1", 0)
        await site.start()
        gateway = web.AppRunner(
            create_app(
                model_url=f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
                today=date(2026, 10, 20),
                query_delay=0.01,
                native_timeout=0.05,
            )
        )
        await gateway.setup()
        site = web.TCPSite(gateway, "127.0.0.1", 0)
        await site.start()
        try:
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(
                    f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/ws"
                ) as ws:
                    await until(ws, lambda e: e["kind"] == "native_capabilities")
                    await ws.send_json(
                        {"kind": "text", "text": "Book Friday at four thirty"}
                    )
                    native = await until(ws, lambda e: e["kind"] == "native_readback")
                    assert "Friday" in native["text"] and "4:30" in native["text"]
                    await until(ws, lambda e: e["kind"] == "proposal_readback")
                    assert len(connections) == 1 and controls[-1]["mode"] == mode
                    await ws.send_json({"kind": "confirm", "version": 1})
                    native = await until(ws, lambda e: e["kind"] == "native_readback")
                    assert native["purpose"] == "receipt" and "B K" in native["text"]
                    fallback = await until(
                        ws, lambda e: e["kind"] == "receipt_readback"
                    )
                    assert fallback["receipt"]["id"] == "BK-0001"
                    assert controls[-1]["pace"] == 5
                    assert len(connections) == 1
        finally:
            await gateway.cleanup()
            await upstream.cleanup()

    asyncio.run(run())
