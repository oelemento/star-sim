"""Hand-written demo protocol: 8-channel 2-fold serial dilution.

This is the "digital twin" reference protocol. It exercises the core liquid
handling verbs (pick up tips, aspirate, dispense, drop tips) using the 8-channel
head, and produces a serial dilution across several columns of the destination
plate. It is deterministic and parameterized so it can later be called by an AI
agent.

Layout assumptions (see star_sim/deck.py):
  - source_plate column 1 (A1:H1): concentrated dye
  - source_plate column 2 (A2:H2): buffer / diluent
  - dest_plate columns 1..n_columns receive the dilution series

All volumes stay within the 2 mL well capacity.
"""

import asyncio

from star_sim.deck import DeckLayout
from pylabrobot.liquid_handling import LiquidHandler

# A column is 8 wells (rows A-H) addressed for the 8-channel head.
ROWS = "ABCDEFGH"


def _column(plate, col: int):
    """Return the 8 wells of a 1-indexed plate column as an A..H ordered list."""
    return [plate.get_item(f"{row}{col}") for row in ROWS]


def prime_source(layout: DeckLayout, dye_volume: float, buffer_volume: float) -> None:
    """Set starting liquids in the source plate so the simulator can track them.

    Column 1 gets dye, column 2 gets buffer. On the physical robot this step is
    done by the operator filling the plate; here it seeds the volume tracker.
    """
    for well in _column(layout.source_plate, 1):
        well.tracker.set_liquids([("dye", dye_volume)])
    for well in _column(layout.source_plate, 2):
        well.tracker.set_liquids([("buffer", buffer_volume)])


async def run(
    lh: LiquidHandler,
    layout: DeckLayout,
    n_columns: int = 6,
    transfer_volume: float = 100.0,
    step_delay: float = 0.0,
) -> None:
    """Execute the serial dilution.

    Steps:
      1. Pre-fill destination columns 2..n with `transfer_volume` of buffer.
      2. Seed destination column 1 with 2x `transfer_volume` of dye.
      3. Serially transfer `transfer_volume` from column k to k+1 (k=1..n-1),
         halving concentration at each step.

    Args:
        lh: a set-up LiquidHandler (sim or hardware).
        layout: the deck layout from build_starlet_deck().
        n_columns: number of columns in the dilution series (>= 2).
        transfer_volume: per-channel transfer volume in uL.
        step_delay: seconds to pause after each aspirate/dispense. Use a nonzero
            value (e.g. 0.5) so deck state changes are visible in the visualizer;
            leave at 0 for fast headless/hardware runs.
    """

    async def pause():
        if step_delay:
            await asyncio.sleep(step_delay)
    if n_columns < 2:
        raise ValueError(f"n_columns must be >= 2, got {n_columns}")

    src = layout.source_plate
    dst = layout.dest_plate

    # Validate against the plate's real column count before any movement, so a
    # bad parameter fails fast instead of crashing mid-run with tips picked up
    # and the deck in a partial state (critical on physical hardware).
    n_dst_cols = dst.num_items_x
    if n_columns > n_dst_cols:
        raise ValueError(
            f"n_columns ({n_columns}) exceeds destination plate columns ({n_dst_cols})"
        )
    rack = layout.tip_rack
    vols = [transfer_volume] * len(ROWS)

    # Buffer needed per source well: one transfer per pre-filled column (2..n)
    # plus the dye column needs 2x. Validate against well capacity up front so a
    # bad parameter fails before any movement.
    buffer_needed = (n_columns - 1) * transfer_volume
    dye_needed = 2 * transfer_volume
    prime_source(layout, dye_volume=dye_needed + transfer_volume,
                 buffer_volume=buffer_needed + transfer_volume)

    tips = rack["A1:H1"]

    # --- 1. Pre-fill destination columns 2..n with buffer ---
    await lh.pick_up_tips(tips)
    for col in range(2, n_columns + 1):
        await lh.aspirate(_column(src, 2), vols=vols)
        await pause()
        await lh.dispense(_column(dst, col), vols=vols)
        await pause()
    await lh.drop_tips(tips)

    # --- 2. Seed destination column 1 with 2x dye ---
    await lh.pick_up_tips(tips)
    double = [2 * transfer_volume] * len(ROWS)
    await lh.aspirate(_column(src, 1), vols=double)
    await pause()
    await lh.dispense(_column(dst, 1), vols=double)
    await pause()
    await lh.drop_tips(tips)

    # --- 3. Serial transfer down the row, fresh tips per step to avoid carryover ---
    for k in range(1, n_columns):
        await lh.pick_up_tips(tips)
        await lh.aspirate(_column(dst, k), vols=vols)
        await pause()
        await lh.dispense(_column(dst, k + 1), vols=vols)
        await pause()
        await lh.drop_tips(tips)


def report(layout: DeckLayout, n_columns: int = 6) -> str:
    """Summarize destination-column volumes after a run (uses the volume tracker)."""
    lines = ["Destination plate volumes (well A of each column):"]
    for col in range(1, n_columns + 1):
        well = layout.dest_plate.get_item(f"A{col}")
        lines.append(f"  column {col:>2}: {well.tracker.get_used_volume():6.1f} uL")
    return "\n".join(lines)
