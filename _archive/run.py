#!/usr/bin/env python3
"""Run an AI-directed experiment on the Hamilton STAR digital twin (or real robot).

The agent receives a natural-language goal and issues liquid-handling commands
autonomously. Set ANTHROPIC_API_KEY (default), or GROQ_API_KEY with --provider groq.
"""

import argparse
import asyncio
import json
import os
import webbrowser

from agent import (
    SERIAL_DILUTION_SCRIPT, ToolCall,
    AnthropicClient, OpenAICompatClient,
    run_agent, run_scripted,
)
from star_sim import RobotEnv
from _archive.visualize_run import LiveCapture, _fetch_js, _geometry, render_html


_DEFAULT_GOAL = (
    "Perform a 2-fold serial dilution of dye across 6 columns of destination plate. "
    "Use 100 µL transfers. "
    "Start with the pre-requisite dye and buffer on source plate. "
    "Then, seed column 1 of the destination plate. Then additionally "
    "seed any necessary buffer in the next columns. Then perform a serial transfer. "
    "Afterwards, columns 1-6 of the destination plate should hold dye at concentrations of 200, 100, 50, 25, 12.5, 6.25 µM."
)


# def _save_record(
#     goal: str, final_text: str, record: list[ToolCall], messages: list[str | None],
# ) -> str:
#     os.makedirs("runs", exist_ok=True)
#     ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
#     path = f"runs/{ts}.json"
#     with open(path, "w") as f:
#         json.dump({"goal": goal, "final_text": final_text,
#                    "record": [[name, args] for name, args in record],
#                    "messages": messages}, f, indent=2)
#     return path


# def _load_record(path: str) -> tuple[str, str, list[ToolCall]]:
#     with open(path) as f:
#         data = json.load(f)
#     return data["goal"], data["final_text"], [(name, args) for name, args in data["record"]]


async def main(
    scripted: bool,
    provider: str,
    model: str | None,
    base_url: str | None,
    replay: str | None,
) -> None:
    live = not scripted and not replay
    if not live:
        raise NotImplementedError
    
    env = RobotEnv(use_hardware=False)
    await env.setup()
    
    try:
        geometry = _geometry(env)
        live_capture = LiveCapture(env, geometry)
        
        async def on_step(name: str, args: dict, message: str | None, result: dict) -> None:
            live_capture.record_step(name, args, message, result.get("error"))
            
        
    finally:
        await env.teardown()
        return

    live_server = None
    live_capture: LiveCapture | None = None
    if visualize:
        if use_hardware:
            raise SystemExit("--visualize is for simulation only; drop --hardware.")
        os.makedirs("runs", exist_ok=True)
        frames_name = "_live.frames.json"
        html_path = os.path.join("runs", "_live.viz.html")
        frames_path = os.path.join("runs", frames_name)

        # Listens to the real env's own movement events as they actually happen —
        # no second env, no re-dispatching, no duplicate backend chatter. The
        # state needed for each frame was already computed once by the real run.
        geometry = _geometry(env)
        live_capture = LiveCapture(env, geometry)

        def write_frames_now() -> None:
            with open(frames_path, "w") as f:
                json.dump(live_capture.frames, f)

        # The HTML (with the bundled Three.js payload) is written once; only the
        # small frames.json is rewritten per step, and the open tab polls that —
        # no page reload, so the user can freely scrub through history without
        # ever getting yanked back to the live edge.
        html = render_html(goal, geometry, live_capture.frames, "live run", _fetch_js(),
                            live=True, frames_url=frames_name)
        with open(html_path, "w") as f:
            f.write(html)
        write_frames_now()

        serve_url, live_server = _serve_dir("runs")
        webbrowser.open(f"{serve_url}/_live.viz.html")
        print(f"\n[live]  Visualizer at {serve_url}/_live.viz.html")
        if delay == 0.0:
            delay = 0.5

    async def on_step(name: str, args: dict, message: str | None, result: dict) -> None:
        if live_capture is not None:
            live_capture.record_step(name, args, message, result.get("error"))
            write_frames_now()

    live_hook = on_step if live_capture is not None else None

    try:
        run_record: list[ToolCall] | None = None

        if replay is not None:
            goal, final_text, script = _load_record(replay)
            print(f"\n[replay]  Loaded {len(script)} steps from {replay}")
            print(f"\n[replay]  Goal: {goal}")
            await run_scripted(env, script, step_delay=delay, confirm=confirm, on_step=live_hook)
            print("\n[replay]  Replay completed, see agent message:")
            print(f"\n[agent] {final_text}")
            if live_capture is not None:
                live_capture.finish(final_text)
                write_frames_now()
        elif scripted:
            await run_scripted(env, SERIAL_DILUTION_SCRIPT, step_delay=delay, confirm=confirm, on_step=live_hook)
            print("\n[script]  Script completed")
        else:
            print(f"\n[user]  Goal: {goal}")
            client = _build_client(provider, goal, model, base_url)

            final_text, run_record, run_messages = await run_agent(
                client, env, step_delay=delay, confirm=confirm, on_step=live_hook,
            )
            if live_capture is not None:
                live_capture.finish(final_text)
                write_frames_now()
            if record and run_record is not None:
                path = _save_record(goal, final_text, run_record, run_messages)
                print(f"\n[record]  Saved {len(run_record)} steps to {path}")

        print("\n" + "─" * 60)
        print("\n[user]  Final plate contents:")
        print(json.dumps(env.plate_map.to_dict(), indent=2))

        if live_server is not None:
            input("\n[live]  Run complete. The visualizer stays up — press Enter to stop it and exit...")

    finally:
        if live_server is not None:
            live_server.shutdown()
        if live_capture is not None:
            live_capture.close()
        await env.teardown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["anthropic", "groq", "ollama", "openai-compat"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model", type=str, default=None,
                        help="model name override (each provider has a sensible default)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL (required for openai-compat; overrides provider default)")
    parser.add_argument("--replay", type=str, default=None, metavar="PATH",
                        help="load a recorded JSON file and run it as a scripted replay")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scripted", action="store_true",
                      help="replay the hardcoded serial-dilution script (no agent)")
    args = parser.parse_args()

    asyncio.run(main(
        scripted=args.scripted,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        replay=args.replay,
    ))
