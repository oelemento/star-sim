"""Atomic 8-channel column-level operations for AI-directed experiments.

Each function is self-contained: it acquires fresh tips, performs one logical
liquid-handling action, and drops tips before returning. They are the building
blocks an AI agent composes into protocols.

All functions accept a RobotEnv so the agent has one object to pass. Plates
are referenced by name matching the DeckLayout fields:
  "source_plate" → env.layout.source_plate
  "dest_plate"   → env.layout.dest_plate

Volumes are always in microlitres.
"""

from __future__ import annotations

import asyncio

from star_sim.env import RobotEnv

_ROWS = "ABCDEFGH"


def _col(plate, col: int) -> list:
    return [plate.get_item(f"{row}{col}") for row in _ROWS]


def _plate(env: RobotEnv, name: str):
    p = getattr(env.layout, name, None)
    if p is None:
        raise ValueError(
            f"Unknown plate '{name}'. Available: 'source_plate', 'dest_plate'."
        )
    return p


async def column_transfer(
    env: RobotEnv,
    src_plate: str,
    src_col: int,
    dst_plate: str,
    dst_col: int,
    volume: float,
    step_delay: float = 0.0,
) -> None:
    """Move `volume` µL from src_plate column src_col to dst_plate column dst_col.

    Uses one tip column. Source and destination may be the same plate.
    """
    async def pause():
        if step_delay:
            await asyncio.sleep(step_delay)

    vols = [volume] * len(_ROWS)
    tip_col = env.tips.next_column()
    await env.lh.pick_up_tips(tip_col)
    try:
        await env.lh.aspirate(_col(_plate(env, src_plate), src_col), vols=vols)
        await pause()
        await env.lh.dispense(_col(_plate(env, dst_plate), dst_col), vols=vols)
        await pause()
    finally:
        await env.lh.drop_tips(tip_col)


async def multi_dispense(
    env: RobotEnv,
    src_plate: str,
    src_col: int,
    dst_plate: str,
    dst_cols: list[int],
    volume: float,
    step_delay: float = 0.0,
) -> None:
    """Dispense `volume` µL from one source column into each of several destination columns.

    Reuses one tip set for all dispenses — safe because tips only contact one
    source. Re-aspirates from the source before each destination column.
    The source wells must each hold at least volume * len(dst_cols) µL total.
    """
    async def pause():
        if step_delay:
            await asyncio.sleep(step_delay)

    src = _plate(env, src_plate)
    dst = _plate(env, dst_plate)
    vols = [volume] * len(_ROWS)

    tip_col = env.tips.next_column()
    await env.lh.pick_up_tips(tip_col)
    try:
        for col in dst_cols:
            await env.lh.aspirate(_col(src, src_col), vols=vols)
            await pause()
            await env.lh.dispense(_col(dst, col), vols=vols)
            await pause()
    finally:
        await env.lh.drop_tips(tip_col)


async def mix_column(
    env: RobotEnv,
    plate: str,
    col: int,
    volume: float,
    repetitions: int = 3,
    step_delay: float = 0.0,
) -> None:
    """Mix a column in place by aspirating and re-dispensing `repetitions` times.

    Uses one tip column.
    """
    async def pause():
        if step_delay:
            await asyncio.sleep(step_delay)

    p = _plate(env, plate)
    vols = [volume] * len(_ROWS)

    tip_col = env.tips.next_column()
    await env.lh.pick_up_tips(tip_col)
    try:
        for _ in range(repetitions):
            await env.lh.aspirate(_col(p, col), vols=vols)
            await pause()
            await env.lh.dispense(_col(p, col), vols=vols)
            await pause()
    finally:
        await env.lh.drop_tips(tip_col)


async def serial_transfer(
    env: RobotEnv,
    plate: str,
    start_col: int,
    end_col: int,
    volume: float,
    step_delay: float = 0.0,
) -> None:
    """Transfer `volume` µL from column k to column k+1 across [start_col, end_col].

    Uses one tip set for the whole sequence. start_col and end_col are both
    inclusive: end_col receives the final transfer.

    This is contamination-safe because each aspirate draws from a well that was
    just dispensed into — the tips never go "backwards" in concentration.
    """
    async def pause():
        if step_delay:
            await asyncio.sleep(step_delay)

    p = _plate(env, plate)
    vols = [volume] * len(_ROWS)

    tip_col = env.tips.next_column()
    await env.lh.pick_up_tips(tip_col)
    try:
        for k in range(start_col, end_col):
            await env.lh.aspirate(_col(p, k), vols=vols)
            await pause()
            await env.lh.dispense(_col(p, k + 1), vols=vols)
            await pause()
    finally:
        await env.lh.drop_tips(tip_col)
