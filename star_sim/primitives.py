"""Well-level liquid-handling primitives for AI-directed experiments.

Two operations cover everything the agent needs:

  transfer(...)  move liquid from a set of source wells to a set of destinations
  mix(...)       aspirate and re-dispense in place to resuspend or homogenise

Both take *selectors* rather than column numbers — a single well ("B3"), a range
("A1:H1"), or an explicit list — so the same primitive serves a one-well spot
check and a full-plate fill. `star_sim.channels.schedule()` breaks the requested
wells into the sequence of movements the machine would physically make; see that
module for the head-geometry constraints.

Plates are referenced by name matching the DeckLayout fields:
  "source_plate" -> env.layout.source_plate
  "dest_plate"   -> env.layout.dest_plate

Volumes are always in microlitres.
"""

from __future__ import annotations

import asyncio

from star_sim.channels import (
    ROWS,
    Move,
    describe,
    pair,
    parse_selector,
    schedule,
)
from star_sim.env import RobotEnv


def _plate(env: RobotEnv, name: str):
    p = getattr(env.layout, name, None)
    if p is None:
        raise ValueError(
            f"Unknown plate '{name}'. Available: 'source_plate', 'dest_plate'."
        )
    return p


def _rows_of(wells) -> list[str]:
    return [w[0] for w in wells]


async def transfer(
    env: RobotEnv,
    src_plate: str,
    src,
    dst_plate: str,
    dst,
    volume,
    tips: str = "new",
    transfer_cells: bool = False,
    step_delay: float = 0.0,
) -> dict:
    """Move liquid from `src` wells to `dst` wells.

    Args:
        src, dst: selectors. Equal-length selections pair elementwise; a single
            source well broadcasts across every destination.
        volume: µL, either one number for every pair or one per destination.
        tips: "new" takes fresh tips for each movement — the safe default.
            "reuse" holds one set of tips for the whole call, which is what makes
            a serial dilution or a one-reagent-into-many-wells distribution cheap.
        transfer_cells: pass True only after `mix` has resuspended the source.

    Returns a summary dict including any contamination warnings.
    """
    if tips not in ("new", "reuse"):
        raise ValueError(f"tips must be 'new' or 'reuse', got {tips!r}.")

    src_res = _plate(env, src_plate)
    dst_res = _plate(env, dst_plate)
    moves = schedule(pair(src, dst, volume))

    async def pause():
        if step_delay:
            await asyncio.sleep(step_delay)

    warnings: list[str] = []
    # Tracks where the held tips last left liquid, so reuse that doubles back
    # into an unrelated well can be reported. A serial dilution always aspirates
    # from the well it just filled, which is why it is safe to reuse tips there.
    last_dispensed: set[str] | None = None
    held = False

    try:
        for move in moves:
            if not held:
                spots = env.tips.acquire(_rows_of(move.src_wells))
                await env.lh.pick_up_tips(spots)
                env.record_movement(
                    "pick_up_tips",
                    rack=env.layout.tip_rack.name,
                    wells=list(env.tips.last_spots),
                    channels=list(move.channels),
                )
                held = True
                last_dispensed = None
            elif last_dispensed is not None and not set(move.src_wells) <= last_dispensed:
                outside = sorted(set(move.src_wells) - last_dispensed)
                warnings.append(
                    f"Reused tips aspirated from {describe(outside)}, which these "
                    f"tips did not previously dispense into — carryover from "
                    f"{describe(sorted(last_dispensed))} is possible. Use "
                    f"tips='new' if these wells must not cross-contaminate."
                )

            vols = list(move.volumes)
            await env.lh.aspirate(
                [src_res.get_item(w) for w in move.src_wells], vols=vols
            )
            env.record_movement(
                "aspirate",
                plate=src_plate,
                wells=list(move.src_wells),
                channels=list(move.channels),
                volumes=vols,
            )
            await pause()

            await env.lh.dispense(
                [dst_res.get_item(w) for w in move.dst_wells], vols=vols
            )
            env.record_movement(
                "dispense",
                plate=dst_plate,
                wells=list(move.dst_wells),
                channels=list(move.channels),
                volumes=vols,
                transfer_cells=transfer_cells,
            )
            await pause()

            last_dispensed = set(move.dst_wells)

            if tips == "new":
                await env.lh.discard_tips()
                env.record_movement("discard_tips", channels=list(move.channels))
                held = False
    finally:
        if held:
            await env.lh.discard_tips()
            env.record_movement(
                "discard_tips", channels=list(moves[-1].channels) if moves else []
            )

    return {
        "moves": len(moves),
        "wells_filled": sum(len(m) for m in moves),
        "tips_remaining": env.tips.remaining,
        **({"warnings": warnings} if warnings else {}),
    }


async def mix(
    env: RobotEnv,
    plate: str,
    wells,
    volume: float,
    repetitions: int = 3,
    tips: str = "new",
    step_delay: float = 0.0,
) -> dict:
    """Resuspend or homogenise `wells` in place by repeated aspirate/dispense.

    Mixing never moves liquid between wells, so one set of tips serves every
    repetition within a column group regardless of `tips`; the setting only
    controls whether fresh tips are taken for each column group.
    """
    if repetitions < 1:
        raise ValueError(f"repetitions must be at least 1, got {repetitions}.")

    selected = parse_selector(wells)
    # Mixing is a same-well operation, so pair each well with itself and let
    # schedule() apply the same column/channel grouping as a transfer.
    moves = schedule([(w, w, float(volume)) for w in selected])
    res = _plate(env, plate)

    async def pause():
        if step_delay:
            await asyncio.sleep(step_delay)

    held = False
    try:
        for move in moves:
            if not held:
                spots = env.tips.acquire(_rows_of(move.src_wells))
                await env.lh.pick_up_tips(spots)
                env.record_movement(
                    "pick_up_tips",
                    rack=env.layout.tip_rack.name,
                    wells=list(env.tips.last_spots),
                    channels=list(move.channels),
                )
                held = True

            items = [res.get_item(w) for w in move.src_wells]
            vols = list(move.volumes)
            for _ in range(repetitions):
                await env.lh.aspirate(items, vols=vols)
                env.record_movement(
                    "aspirate", plate=plate, wells=list(move.src_wells),
                    channels=list(move.channels), volumes=vols,
                )
                await pause()
                await env.lh.dispense(items, vols=vols)
                env.record_movement(
                    "dispense", plate=plate, wells=list(move.src_wells),
                    channels=list(move.channels), volumes=vols, mixing=True,
                )
                await pause()

            if tips == "new":
                await env.lh.discard_tips()
                env.record_movement("discard_tips", channels=list(move.channels))
                held = False
    finally:
        if held:
            await env.lh.discard_tips()
            env.record_movement(
                "discard_tips", channels=list(moves[-1].channels) if moves else []
            )

    return {
        "moves": len(moves),
        "wells_mixed": len(selected),
        "repetitions": repetitions,
        "tips_remaining": env.tips.remaining,
    }
