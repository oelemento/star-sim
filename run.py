#!/usr/bin/env python3
"""Run an AI-directed experiment on the Hamilton STAR digital twin (or real robot).

The agent receives a natural-language goal and issues liquid-handling commands
autonomously using the Claude API. Set ANTHROPIC_API_KEY before running.

Examples:
    # Headless simulation (prints every tool call and result):
    python run.py

    # Simulation with the live 3D browser visualizer:
    python run.py --visualize

    # Custom experiment goal:
    python run.py --goal "dispense 50 uL from source column 1 into dest columns 1 through 4"

    # Drive the physical Hamilton STAR over USB (no visualizer):
    python run.py --hardware
"""

import argparse
import asyncio
import json

from agent import SERIAL_DILUTION_SCRIPT, run_agent, run_agent_ollama, run_scripted
from star_sim import RobotEnv
from star_sim.deck import DeckLayout
from pylabrobot.visualizer import Visualizer

_DEFAULT_GOAL = (
    "Perform a 2-fold serial dilution of dye across 6 columns of dest_plate. "
    "Use 100 µL transfers. "
    "Seed dest_plate column 1 with 200 µL of dye (not 100 µL) so that after the serial "
    "transfer takes 100 µL away, column 1 still retains 100 µL at 200 µM. "
    "Pre-fill dest_plate columns 2-6 with 100 µL of buffer each before the serial transfer. "
    "After the serial transfer, dest columns 1-6 should hold 200, 100, 50, 25, 12.5, 6.25 µM."
)


def prime_simulation(layout: DeckLayout, wells: dict[str, dict[str, float]]) -> None:
    """Seed the simulator's volume tracker from a {plate_name: {well_id: volume_ul}} mapping.

    For agent/scripted runs this is called automatically by the propose_prime tool.
    Exposed here for manual use in one-off scripts or tests.
    """
    for plate_name, vol_map in wells.items():
        plate = getattr(layout, plate_name)
        for well_id, vol in vol_map.items():
            plate.get_item(well_id).tracker.set_volume(vol)


async def _wait_for_browser(vis: Visualizer, timeout: float = 30.0) -> bool:
    """Block until the browser tab connects to the visualizer websocket."""
    print("Waiting for the visualizer browser tab to connect...")
    deadline = int(timeout / 0.1)
    for _ in range(deadline):
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
    ollama: bool,
    ollama_model: str,
    confirm: bool,
) -> None:
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

        if scripted:
            await run_scripted(env, SERIAL_DILUTION_SCRIPT, step_delay=delay, confirm=confirm)
            print("[script]  Script completed, now printing plate data")
        elif ollama:
            await run_agent_ollama(
                env, goal, model=ollama_model, step_delay=delay, confirm=confirm
            )
        else:
            await run_agent(env, goal, step_delay=delay, confirm=confirm)

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
    parser.add_argument("--goal", type=str, default=_DEFAULT_GOAL,
                        help="experiment goal as a natural-language string")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds between robot operations (auto-set to 0.5 with --visualize)")
    parser.add_argument("--confirm", action="store_true", default=False,
                        help="pause before each tool call for step-through debugging")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scripted", action="store_true",
                      help="replay a hardcoded script (no agent)")
    mode.add_argument("--ollama", action="store_true",
                      help="use a local Ollama model (requires ollama)")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b",
                        help="Ollama model name (default: llama3.1:8b)")
    args = parser.parse_args()

    asyncio.run(main(
        use_hardware=args.hardware,
        visualize=args.visualize,
        goal=args.goal,
        delay=args.delay,
        scripted=args.scripted,
        ollama=args.ollama,
        ollama_model=args.ollama_model,
        confirm=args.confirm,
    ))
