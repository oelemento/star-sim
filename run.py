#!/usr/bin/env python3.11
"""Run a protocol on the Hamilton STAR digital twin (or the real robot).

Examples:
    # Simulation, headless (prints every robot action + a volume report):
    python3.11 run.py

    # Simulation with the live 3D browser visualizer:
    python3.11 run.py --visualize

    # Drive the physical Hamilton STAR over USB (no visualizer):
    python3.11 run.py --hardware

The same protocol code runs in every mode; only the backend changes.
"""

import argparse
import asyncio

from star_sim import make_liquid_handler
from protocols import serial_dilution


async def _wait_for_browser(vis, timeout: float = 30.0) -> bool:
    """Block until the browser tab connects to the visualizer websocket.

    The visualizer only streams state changes to a *connected* browser and does
    not replay missed events, so the protocol must not start until the tab is
    listening. Returns True if a connection was established within `timeout`.
    """
    print("Waiting for the visualizer browser tab to connect...")
    deadline = timeout / 0.1
    waited = 0
    while waited < deadline:
        if vis.has_connection():
            print("Browser connected. Starting protocol.")
            return True
        await asyncio.sleep(0.1)
        waited += 1
    print("Warning: no browser connected within timeout; running anyway.")
    return False


async def main(use_hardware: bool, visualize: bool, n_columns: int, delay: float) -> None:
    lh, layout = make_liquid_handler(use_hardware=use_hardware)

    vis = None
    if visualize:
        if use_hardware:
            raise SystemExit("--visualize is for simulation only; drop --hardware.")
        from pylabrobot.visualizer import Visualizer

        vis = Visualizer(resource=lh.deck)
        await vis.setup()

    await lh.setup()
    try:
        if vis is not None:
            # Pace operations so each state change is visible, and don't start
            # until the browser is actually watching.
            await _wait_for_browser(vis)
            if delay == 0.0:
                delay = 0.5
        await serial_dilution.run(lh, layout, n_columns=n_columns, step_delay=delay)
        print()
        print(serial_dilution.report(layout, n_columns=n_columns))
        if vis is not None:
            input("\nVisualizer live at the URL above. Press Enter to close...")
    finally:
        await lh.stop()
        if vis is not None:
            await vis.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", action="store_true",
                        help="drive the physical Hamilton STAR (default: simulation)")
    parser.add_argument("--visualize", action="store_true",
                        help="open the live browser visualizer (simulation only)")
    parser.add_argument("--columns", type=int, default=6,
                        help="number of dilution-series columns (default: 6)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds between operations (default: 0 headless, "
                             "0.5 with --visualize so movement is visible)")
    args = parser.parse_args()

    asyncio.run(main(use_hardware=args.hardware,
                     visualize=args.visualize,
                     n_columns=args.columns,
                     delay=args.delay))
