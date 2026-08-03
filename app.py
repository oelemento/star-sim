"""FastAPI web server for the Hamilton STAR digital twin.

Serves an interactive browser UI: the user types a natural-language experiment
prompt, the agent proposes actions one turn at a time, and the user confirms
each turn before the robot executes it.

Usage:
    python app.py
    python app.py --port 8080
    python app.py --reload        # auto-restart on code changes (dev mode)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import AgentSession, _preview, build_client
from star_sim import RobotEnv
from render import LiveCapture, geometry, render_html


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class _State:
    def __init__(self) -> None:
        # Computed once at startup, never change
        self.geometry: dict | None = None

        # Live session — reset at the start of each run
        self.session: AgentSession | None = None
        self.env: RobotEnv | None = None
        self.capture: LiveCapture | None = None
        self.frames: list[dict] = []

        # SSE subscribers: one asyncio.Queue per open browser tab
        self._queues: list[asyncio.Queue] = []

    def broadcast(self, event: dict) -> None:
        payload = json.dumps(event)
        for q in self._queues:
            q.put_nowait(payload)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    async def teardown(self) -> None:
        """Detaches listener from env, then also tears down RobotEnv and disconnects
        from the liquid handler, virtual or real"""
        if self.capture:
            self.capture.close()
            self.capture = None
        if self.env:
            await self.env.teardown()
            self.env = None
        self.session = None
        self.frames = []


_state = _State()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    env = RobotEnv(use_hardware=False)
    await env.setup()
    _state.geometry = geometry(env)
    await env.teardown()
    yield
    # On shutdown: unblock every waiting SSE generator so connections close
    # before uvicorn's drain timeout fires.
    for q in list(_state._queues):
        q.put_nowait(None)  # sentinel — generator exits on None
    await asyncio.sleep(0.1)


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html = render_html(
        geometry=_state.geometry,
        frames=[],
        title="live run",
    )
    return HTMLResponse(html)


class RunRequest(BaseModel):
    goal: str


@app.post("/run")
async def start_run(req: RunRequest):
    """Start a new session: set up the env and run the first think()."""
    if _state.session is not None:
        await _state.teardown()

    _state.env = RobotEnv(use_hardware=False)
    await _state.env.setup()

    _state.capture = LiveCapture(_state.env, _state.geometry)
    _state.frames = _state.capture.frames

    _state.session = AgentSession(env=_state.env, client=build_client("groq", req.goal))

    asyncio.create_task(_think())
    return JSONResponse({"ok": True})


@app.post("/confirm")
async def confirm():
    """User approved the proposed tool uses: execute them, then think() again."""
    if _state.session is None:
        return JSONResponse({"error": "no active session"}, status_code=400)

    asyncio.create_task(_act_then_think())
    return JSONResponse({"ok": True})


@app.post("/stop")
async def stop():
    await _state.teardown()
    _state.broadcast({"type": "stopped"})
    return JSONResponse({"ok": True})


@app.post("/quit")
async def quit_server():
    import os, signal
    asyncio.get_event_loop().call_later(0.1, lambda: os.kill(os.getpid(), signal.SIGINT))
    return JSONResponse({"ok": True})


@app.get("/events")
async def sse():
    q = _state.subscribe()

    async def generate() -> AsyncGenerator[str, None]:
        yield ": connected\n\n"
        try:
            while True:
                payload = await q.get()
                if payload is None:  # shutdown sentinel
                    return
                yield f"data: {payload}\n\n"
        finally:
            _state.unsubscribe(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Internal: think and act
# ---------------------------------------------------------------------------

async def _think() -> None:
    """Run one LLM completion and broadcast the result."""
    try:
        _state.broadcast({"type": "thinking"})
        # time.sleep(15)
        response = await _state.session.think()

        if response.text:
            _state.broadcast({"type": "agent_text", "text": response.text})

        if response.done:
            await _state.teardown()
            _state.broadcast({"type": "done"})
        else:
            proposals = [
                {"name": name, "preview": _preview(name, args)}
                for _, name, args in response.tool_uses
            ]
            _state.broadcast({"type": "proposals", "tools": proposals})

    except Exception as exc:
        _state.broadcast({"type": "error", "message": str(exc)})
        await _state.teardown()


async def _act_then_think() -> None:
    """Execute the pending tool uses, update the visualization, then think() again."""
    async def on_step(name: str, args: dict, message: str | None, result: dict) -> None:
        _state.capture.record_step(name, args, message, result.get("error"))
        _state.frames = _state.capture.frames
        _state.broadcast({
            "type": "tool_result",
            "name": name,
            "ok": "error" not in result,
            "error": result.get("error"),
        })
        _state.broadcast({"type": "frame", "data": _state.capture.frames[-1]})

    try:
        _state.broadcast({"type": "acting"})
        await _state.session.act(step_delay=0.3, on_step=on_step)
    except Exception as exc:
        _state.broadcast({"type": "error", "message": str(exc)})
        await _state.teardown()
        return

    await _think()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="reload on code changes (dev mode)")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    threading.Thread(
        target=lambda: (time.sleep(1.2), webbrowser.open(url)),
        daemon=True,
    ).start()

    uvicorn.run("app:app", host="127.0.0.1", port=args.port, reload=args.reload)
