#!/usr/bin/env python3
"""Run an AI-directed experiment on the Hamilton STAR digital twin (or real robot).

The agent receives a natural-language goal and issues liquid-handling commands
autonomously. Set ANTHROPIC_API_KEY (default), or GROQ_API_KEY with --provider groq.

Examples:
    # Headless simulation with Claude (default):
    python run.py

    # Use Groq (openai/gpt-oss-120b by default):
    # Local Ollama interop is also an available provider option
    python run.py --provider groq

    # Arbitrary OpenAI-compatible endpoint:
    python run.py --provider openai-compat --model my-model --base-url http://host/v1

    # Simulation with the live 3D browser visualizer:
    python run.py --visualize

    # Drive the physical Hamilton STAR over USB:
    python run.py --hardware

    # Save the agent's steps as a JSON replay file:
    python run.py --record

    # Replay a previously recorded run:
    python run.py --replay runs/2026-06-11T12-00-00.json

    # Step through each tool call for review:
    python run.py --confirm
"""

import argparse
import asyncio
import json
import os
from datetime import datetime

from agent import (
    SERIAL_DILUTION_SCRIPT, ToolCall,
    AnthropicClient, OpenAICompatClient,
    run_agent, run_scripted,
)
from star_sim import RobotEnv
from pylabrobot.visualizer import Visualizer

_DEFAULT_GOAL = (
    "Perform a 2-fold serial dilution of dye across 6 columns of destination plate. "
    "Use 100 µL transfers. "
    "Start with the pre-requisite dye and buffer on source plate. "
    "Then, seed column 1 of the destination plate. Then additionally "
    "seed any necessary buffer in the next columns. Then perform a serial transfer. "
    "Afterwards, columns 1-6 of the destination plate should hold dye at concentrations of 200, 100, 50, 25, 12.5, 6.25 µM."
)


def _resolve_goal(goal: str | None) -> str:
    """Return the goal string. If goal looks like a file path, read it; else use as-is.
    Falls back to _DEFAULT_GOAL when goal is None."""
    if goal is None:
        return _DEFAULT_GOAL
    if os.path.exists(goal):
        with open(goal) as f:
            return f.read().strip()
    return goal

_PROVIDER_DEFAULTS: dict[str, dict] = {
    "anthropic": {"model": "claude-opus-4-8",     "base_url": None},
    "groq":      {"model": "openai/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1"},
    "ollama":    {"model": "llama3.1:8b",          "base_url": "http://localhost:11434/v1"},
}

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic":    "ANTHROPIC_API_KEY",
    "groq":         "GROQ_API_KEY",
    "ollama":       "",           # Ollama doesn't need a key
    "openai-compat": "OPENAI_API_KEY",
}


def _build_client(
    provider: str,
    goal: str,
    model: str | None,
    base_url: str | None,
) -> AnthropicClient | OpenAICompatClient:
    defaults = _PROVIDER_DEFAULTS.get(provider, {"model": None, "base_url": None})
    resolved_model = model or defaults["model"]
    resolved_url = base_url or defaults["base_url"]

    if provider == "anthropic":
        if not resolved_model:
            resolved_model = "claude-opus-4-8"
        return AnthropicClient(goal=goal, model=resolved_model)

    # All other providers use the OpenAI-compatible client
    env_key = _PROVIDER_ENV_KEYS.get(provider, "OPENAI_API_KEY")
    api_key = os.environ.get(env_key, "none") if env_key else "ollama"
    if not resolved_url:
        raise SystemExit(f"--base-url is required for provider '{provider}'")
    if not resolved_model:
        raise SystemExit(f"--model is required for provider '{provider}'")
    return OpenAICompatClient(
        goal=goal,
        model=resolved_model,
        base_url=resolved_url,
        api_key=api_key,
    )


def _save_record(goal: str, final_text: str, record: list[ToolCall]) -> str:
    os.makedirs("runs", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = f"runs/{ts}.json"
    with open(path, "w") as f:
        json.dump({"goal": goal, "final_text": final_text,
                   "record": [[name, args] for name, args in record]}, f, indent=2)
    return path


def _load_record(path: str) -> tuple[str, str, list[ToolCall]]:
    with open(path) as f:
        data = json.load(f)
    return data["goal"], data["final_text"], [(name, args) for name, args in data["record"]]


async def _wait_for_browser(vis: Visualizer, timeout: float = 30.0) -> bool:
    print("Waiting for the visualizer browser tab to connect...")
    for _ in range(int(timeout / 0.1)):
        if vis.has_connection():
            print("Browser connected. Starting protocol.")
            return True
        await asyncio.sleep(0.1)
    print("Warning: no browser connected within timeout; running anyway.")
    return False


async def main(
    use_hardware: bool,
    visualize: bool,
    goal: str,
    delay: float,
    scripted: bool,
    provider: str,
    model: str | None,
    base_url: str | None,
    confirm: bool,
    record: bool,
    replay: str | None,
) -> None:
    goal = _resolve_goal(goal)
    env = RobotEnv(use_hardware=use_hardware)
    await env.setup()

    vis: Visualizer | None = None
    if visualize:
        if use_hardware:
            raise SystemExit("--visualize is for simulation only; drop --hardware.")
        vis = Visualizer(resource=env.lh.deck)
        await vis.setup()

    try:
        if vis is not None:
            await _wait_for_browser(vis)
            if delay == 0.0:
                delay = 0.5

        run_record: list[ToolCall] | None = None

        if replay is not None:
            goal, final_text, script = _load_record(replay)
            print(f"\n[replay]  Loaded {len(script)} steps from {replay}")
            print(f"\n[replay]  Goal: {goal}")
            await run_scripted(env, script, step_delay=delay, confirm=confirm)
            print("\n[replay]  Replay completed, see agent message:")
            print(f"\n[agent] {final_text}")
        elif scripted:
            await run_scripted(env, SERIAL_DILUTION_SCRIPT, step_delay=delay, confirm=confirm)
            print("\n[script]  Script completed")
        else:
            print(f"\n[user]  Goal: {goal}")
            client = _build_client(provider, goal, model, base_url)
            final_text, run_record = await run_agent(client, env, step_delay=delay, confirm=confirm)
            if record and run_record is not None:
                path = _save_record(goal, final_text, run_record)
                print(f"\n[record]  Saved {len(run_record)} steps to {path}")

        print("\n" + "─" * 60)
        print("\n[user]  Final plate contents:")
        print(json.dumps(env.plate_map.to_dict(), indent=2))

        if vis is not None:
            input("\nVisualizer live at the URL above. Press Enter to close...")

    finally:
        if vis is not None:
            await vis.stop()
        await env.teardown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hardware", action="store_true",
                        help="drive the physical Hamilton STAR (default: simulation)")
    parser.add_argument("--visualize", action="store_true",
                        help="open the live browser visualizer (simulation only)")
    parser.add_argument("--goal", type=str, default=None,
                        help="experiment goal as a natural-language string or path to a .txt file")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds between operations (auto-set to 0.5 with --visualize)")
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["anthropic", "groq", "ollama", "openai-compat"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model", type=str, default=None,
                        help="model name override (each provider has a sensible default)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL (required for openai-compat; overrides provider default)")
    parser.add_argument("--confirm", action="store_true", default=False,
                        help="pause before each tool call for step-through debugging")
    parser.add_argument("--record", action="store_true",
                        help="write agent steps to runs/<timestamp>.json for later replay")
    parser.add_argument("--replay", type=str, default=None, metavar="PATH",
                        help="load a recorded JSON file and run it as a scripted replay")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scripted", action="store_true",
                      help="replay the hardcoded serial-dilution script (no agent)")
    args = parser.parse_args()

    asyncio.run(main(
        use_hardware=args.hardware,
        visualize=args.visualize,
        goal=args.goal,
        delay=args.delay,
        scripted=args.scripted,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        confirm=args.confirm,
        record=args.record,
        replay=args.replay,
    ))
