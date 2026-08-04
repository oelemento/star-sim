"""RobotEnv: the single entry point for AI-directed experiments.

Wraps a LiquidHandler + DeckLayout + TipManager + PlateMap and exposes:
  - observe()  → structured dict of current deck state
  - reset()    → tear down and rebuild to a clean initial state
  - .lh        → the underlying LiquidHandler for protocol calls
  - .layout    → DeckLayout for resource references
  - .tips      → TipManager for safe tip acquisition
  - .plate_map → contents of each well in each plate

Use as an async context manager::

    async with RobotEnv() as env:
        state = env.observe()
        tips = env.tips.next_column()
        await env.lh.pick_up_tips(tips)
        ...
"""

from __future__ import annotations

from typing import Callable

from .deck import DeckLayout
from .lab import make_liquid_handler
from .plate_map import PlateMap
from .tip_manager import TipManager
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.resources import Plate, TipRack


class RobotEnv:
    def __init__(self, use_hardware: bool = False, num_channels: int = 8) -> None:
        self._use_hardware = use_hardware
        self._num_channels = num_channels
        self.lh: LiquidHandler
        self.layout: DeckLayout
        self.tips: TipManager
        self.plate_map: PlateMap
        # Hooks protocol primitives call after each physical sub-step (tip
        # pickup/drop, aspirate, dispense). Empty during normal operation —
        # no-op unless something is listening.
        self._movement_listeners: list[Callable[[str, dict], None]] = []

    def add_movement_listener(self, fn: Callable[[str, dict], None]) -> None:
        self._movement_listeners.append(fn)

    def remove_movement_listener(self, fn: Callable[[str, dict], None]) -> None:
        self._movement_listeners.remove(fn)

    def record_movement(self, kind: str, **info) -> None:
        # Most-recently-registered listener runs first. This lets a narrowly
        # scoped listener (e.g. a dispatch case updating concentrations for
        # the call it wraps) finish its update before a longer-lived listener
        # (e.g. a replay capturing a snapshot) observes the result.
        for fn in reversed(self._movement_listeners):
            fn(kind, info)

    async def setup(self) -> None:
        self.lh, self.layout = make_liquid_handler(
            use_hardware=self._use_hardware,
            num_channels=self._num_channels,
        )
        self.tips = TipManager(self.layout.tip_rack)
        self.plate_map = PlateMap()
        await self.lh.setup()

    async def teardown(self) -> None:
        await self.lh.stop()

    async def reset(self) -> None:
        """Rebuild the environment from scratch with a clean deck and full tip rack."""
        await self.teardown()
        await self.setup()

    async def __aenter__(self) -> RobotEnv:
        await self.setup()
        return self

    async def __aexit__(self, *_) -> None:
        await self.teardown()

    def observe(self) -> dict:
        """Return a complete snapshot of current deck state.

        Suitable for passing directly to an AI agent as tool output or context.
        All volumes in microlitres. Plate wells map well_id → volume_ul (float).
        Tip rack maps position → has_tip (bool).
        """
        return {
            "source_plate": _plate_state(self.layout.source_plate),
            "dest_plate": _plate_state(self.layout.dest_plate),
            "tip_rack": _rack_state(self.layout.tip_rack),
            "tips_remaining_columns": self.tips.remaining,
            "plate_map": self.plate_map.to_dict(),
        }


# --- helpers -----------------------------------------------------------------

def _plate_state(plate: Plate) -> dict[str, float]:
    """Map each *non-empty* well id to its current volume in µL.

    Wells at 0 µL are omitted rather than listed — on a 96-well plate the
    vast majority are empty for most of an experiment, and including them
    was most of observe()'s token cost with zero informational value (an
    LLM tool caller only cares where liquid actually is).
    """
    state = {}
    for row in "ABCDEFGH":
        for col in range(1, plate.num_items_x + 1):
            well_id = f"{row}{col}"
            volume = plate.get_item(well_id).tracker.get_used_volume()
            if volume > 0:
                state[well_id] = volume
    return state


def _rack_state(rack: TipRack) -> dict[int, bool]:
    """Map each tip COLUMN (not individual position) to whether it still has
    a full set of tips. The 8-channel head only ever picks up (or has picked
    up) a whole column at once — per-position detail is dead weight the
    model can't act on, not useful precision.
    """
    rows = "ABCDEFGH"[: rack.num_items_y]
    state = {}
    for col in range(1, rack.num_items_x + 1):
        state[col] = all(rack.get_item(f"{row}{col}").tracker.has_tip for row in rows)
    return state
