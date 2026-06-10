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
    """Map each well id to its current volume in µL."""
    state = {}
    for row in "ABCDEFGH":
        for col in range(1, plate.num_items_x + 1):
            well_id = f"{row}{col}"
            state[well_id] = plate.get_item(well_id).tracker.get_used_volume()
    return state


def _rack_state(rack: TipRack) -> dict[str, bool]:
    """Map each tip position to True (tip present) / False (spent or missing)."""
    rows = "ABCDEFGH"[: rack.num_items_y]
    state = {}
    for row in rows:
        for col in range(1, rack.num_items_x + 1):
            pos = f"{row}{col}"
            state[pos] = rack.get_item(pos).tracker.has_tip
    return state
