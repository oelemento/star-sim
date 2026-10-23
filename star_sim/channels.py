"""Well selection and channel scheduling for the 8-channel head.

This module is where the sim's one real machine constraint lives. The STAR's
pipetting channels sit on a fixed 9 mm pitch along the y axis, so:

  * channel *i* can only ever reach row *i* (A->0, B->1, ... H->7), and
  * one head movement addresses one plate column, because x is fixed per move.

Callers therefore hand in whatever set of wells they want and `schedule()`
breaks it into the sequence of moves the machine would actually make. Nothing
here touches PyLabRobot or performs I/O, so it is cheap to test directly.

Selector syntax
---------------
A selector is a single well id ("B3"), an explicit list (["A1", "C4"]), or a
range ("A1:H5"). A range denotes the rectangular block spanned by its two
corners, walked **column-major** — down each column, then on to the next:

    "A1:H1"  -> A1 B1 C1 D1 E1 F1 G1 H1
    "A1:D2"  -> A1 B1 C1 D1 A2 B2 C2 D2

Note this deliberately differs from PyLabRobot's `plate["A1:D2"]`, which walks
the same block row-major. Column-major is the order `ItemizedResource._ordering`
itself uses, it is the order an 8-channel head fills a plate, and it keeps
`schedule()` from interleaving columns — a serial dilution expressed as
src="A1:H5" / dst="A2:H6" comes out as the chain col1->col2, col2->col3, ...
rather than a scramble. (PyLabRobot's string-range form is marked for
deprecation in its own source, so there is nothing to stay compatible with.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ROWS = "ABCDEFGH"
NUM_CHANNELS = len(ROWS)

_WELL_RE = re.compile(r"^([A-H])([1-9][0-9]?)$")


class SelectionError(ValueError):
    """A selector was malformed, or named a well the head cannot reach."""


def parse_well(well: str) -> tuple[int, int]:
    """Split a well id into (row_index, column_number). Raises SelectionError."""
    m = _WELL_RE.match(well.strip().upper()) if isinstance(well, str) else None
    if not m:
        raise SelectionError(
            f"Malformed well id {well!r}. Expected a row letter A-H followed by "
            f"a column number, e.g. 'A1' or 'H12'."
        )
    return ROWS.index(m.group(1)), int(m.group(2))


def channel_index(well: str) -> int:
    """Which pipetting channel must service this well (0-7, by row)."""
    return parse_well(well)[0]


def column_of(well: str) -> int:
    return parse_well(well)[1]


def normalize(well: str) -> str:
    row, col = parse_well(well)
    return f"{ROWS[row]}{col}"


def parse_selector(sel: str | list[str] | tuple[str, ...]) -> list[str]:
    """Expand a selector into an ordered list of well ids.

    See the module docstring for the syntax. Duplicates are rejected: they would
    silently double a transfer's volume, which is never what a caller means.
    """
    if isinstance(sel, (list, tuple)):
        wells = [normalize(w) for w in sel]
    elif isinstance(sel, str):
        text = sel.strip()
        if ":" in text:
            start, _, end = text.partition(":")
            wells = _expand_range(start, end)
        else:
            wells = [normalize(text)]
    else:
        raise SelectionError(
            f"Selector must be a well id, a range like 'A1:H1', or a list of "
            f"well ids — got {type(sel).__name__}."
        )

    if not wells:
        raise SelectionError(f"Selector {sel!r} selected no wells.")
    seen = set()
    for w in wells:
        if w in seen:
            raise SelectionError(
                f"Selector {sel!r} names well {w} more than once."
            )
        seen.add(w)
    return wells


def _expand_range(start: str, end: str) -> list[str]:
    r0, c0 = parse_well(start)
    r1, c1 = parse_well(end)
    rows = range(min(r0, r1), max(r0, r1) + 1)
    cols = range(min(c0, c1), max(c0, c1) + 1)
    # Column-major: walk down each column before moving to the next.
    return [f"{ROWS[r]}{c}" for c in cols for r in rows]


@dataclass(frozen=True)
class Move:
    """One head movement: aspirate from `src_wells`, dispense to `dst_wells`.

    Every source well shares `src_col`, every destination well shares `dst_col`,
    and `channels` holds the channel index servicing each pair — never repeated,
    since one channel cannot be in two places at once. The three lists are
    parallel and in channel order.
    """

    src_col: int
    dst_col: int
    src_wells: tuple[str, ...]
    dst_wells: tuple[str, ...]
    volumes: tuple[float, ...]
    channels: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.channels)


def pair(
    src: str | list[str],
    dst: str | list[str],
    volume: float | list[float],
) -> list[tuple[str, str, float]]:
    """Expand selectors into (src_well, dst_well, volume) triples.

    Equal-length selections pair elementwise. A shorter source is **tiled** across
    the destinations, which is how one reagent feeds many wells: a source column
    ("A1:H1") repeats across each destination column, and a single source well
    repeats across a row of destinations. Tiling only requires that the
    destination count be a whole multiple of the source count — whether the
    resulting pairs are physically reachable is `schedule()`'s job to check.

    Volume is either a scalar applied to every pair, or one value per destination.
    """
    src_wells = parse_selector(src)
    dst_wells = parse_selector(dst)

    if len(src_wells) != len(dst_wells):
        if len(src_wells) > len(dst_wells) or len(dst_wells) % len(src_wells):
            raise SelectionError(
                f"Cannot pair {len(src_wells)} source wells with "
                f"{len(dst_wells)} destination wells. Either select the same "
                f"number of each, or select a source whose count divides the "
                f"destination count so it can be repeated (e.g. one source "
                f"column feeding several destination columns)."
            )
        src_wells = src_wells * (len(dst_wells) // len(src_wells))

    if isinstance(volume, (int, float)):
        volumes = [float(volume)] * len(dst_wells)
    else:
        volumes = [float(v) for v in volume]
        if len(volumes) != len(dst_wells):
            raise SelectionError(
                f"Got {len(volumes)} volumes for {len(dst_wells)} destination "
                f"wells. Pass one volume per destination, or a single number."
            )

    for v in volumes:
        if v <= 0:
            raise SelectionError(f"Transfer volumes must be positive, got {v}.")

    return list(zip(src_wells, dst_wells, volumes))


def schedule(triples: list[tuple[str, str, float]]) -> list[Move]:
    """Group (src_well, dst_well, volume) triples into physically valid moves.

    Triples are consumed in order and grouped greedily: a triple joins the
    current move if it shares both source and destination column and needs a
    channel that move has not already used, otherwise it starts a new one.
    Preserving input order matters — a serial dilution is only correct if
    col1->col2 completes before col2->col3 begins.
    """
    moves: list[Move] = []
    cur: list[tuple[str, str, float, int]] = []

    def flush() -> None:
        if not cur:
            return
        cur.sort(key=lambda t: t[3])  # channel order, so the payload reads top-to-bottom
        moves.append(
            Move(
                src_col=column_of(cur[0][0]),
                dst_col=column_of(cur[0][1]),
                src_wells=tuple(t[0] for t in cur),
                dst_wells=tuple(t[1] for t in cur),
                volumes=tuple(t[2] for t in cur),
                channels=tuple(t[3] for t in cur),
            )
        )
        cur.clear()

    for src, dst, vol in triples:
        src_row, src_col = parse_well(src)
        dst_row, dst_col = parse_well(dst)
        if src_row != dst_row:
            raise SelectionError(
                f"{src} -> {dst} crosses rows. A channel is fixed to its row, so "
                f"a transfer must stay in one row (A1->A5 is fine, A1->B5 is not). "
                f"To move liquid between rows, transfer via an intermediate well."
            )
        channel = src_row
        if cur and (
            column_of(cur[0][0]) != src_col
            or column_of(cur[0][1]) != dst_col
            or any(t[3] == channel for t in cur)
        ):
            flush()
        cur.append((src, dst, vol, channel))

    flush()
    return moves


def describe(wells: list[str]) -> str:
    """Compact human summary of a selection, for previews and log lines."""
    if not wells:
        return "(none)"
    if len(wells) == 1:
        return wells[0]
    if len(wells) <= 4:
        return " ".join(wells)
    return f"{wells[0]}..{wells[-1]} ({len(wells)} wells)"
