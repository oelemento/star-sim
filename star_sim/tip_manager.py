"""TipManager: per-tip acquisition for the 8-channel head.

The Hamilton 96-tip rack is 8 rows (A-H) x 12 columns. Because the channels sit
on a fixed pitch, a single pickup can only take tips from **one column**, one tip
per row — but it need not take all eight. A single-channel operation consumes one
tip, not a whole column, so a rack really does hold 96 single-channel operations
rather than 12.

Tip presence itself lives in PyLabRobot's tip tracker, not here; this class only
decides *which* spots to reach for. That means it stays correct even if tips are
consumed by something other than `acquire()`.
"""

from pylabrobot.resources import TipRack

from .channels import ROWS


class OutOfTipsError(Exception):
    pass


class TipManager:
    """Chooses tip spots for the rows a pipetting step needs.

    Usage::

        mgr = TipManager(layout.tip_rack)
        spots = mgr.acquire(["A", "B"])   # two tips from the leftmost column
        await lh.pick_up_tips(spots)      # that has both free
        ...
        await lh.discard_tips()
    """

    def __init__(self, rack: TipRack) -> None:
        self._rack = rack
        self.last_spots: list[str] = []  # well ids of the most recent acquire()

    @property
    def rows(self) -> str:
        return ROWS[: self._rack.num_items_y]

    @property
    def num_columns(self) -> int:
        return self._rack.num_items_x

    @property
    def remaining(self) -> int:
        """Individual tips still on the rack."""
        return sum(
            1
            for col in range(1, self.num_columns + 1)
            for row in self.rows
            if self._has_tip(row, col)
        )

    @property
    def free_rows_by_column(self) -> dict[int, str]:
        """{column: row letters still holding a tip}. Columns with none are omitted."""
        out = {}
        for col in range(1, self.num_columns + 1):
            free = "".join(r for r in self.rows if self._has_tip(r, col))
            if free:
                out[col] = free
        return out

    def acquire(self, rows: list[str]) -> list:
        """Return tip spots for `rows`, all from a single column.

        Scans columns left to right for the first one holding a tip at every
        requested row, so partially used columns get filled in before fresh ones
        are opened. Returns spots in row order.

        Raises:
            OutOfTipsError: if no single column can serve all of `rows`.
        """
        wanted = self._validate(rows)
        for col in range(1, self.num_columns + 1):
            if all(self._has_tip(r, col) for r in wanted):
                self.last_spots = [f"{r}{col}" for r in wanted]
                return [self._rack.get_item(w) for w in self.last_spots]

        raise OutOfTipsError(
            f"Tip rack '{self._rack.name}' has no remaining column with free tips "
            f"in rows {''.join(wanted)} ({self.remaining} tips left overall). "
            f"Free rows per column: {self.free_rows_by_column or 'none'}."
        )

    def _validate(self, rows: list[str]) -> list[str]:
        if not rows:
            raise ValueError("acquire() needs at least one row.")
        wanted = []
        for r in rows:
            up = str(r).strip().upper()
            if up not in self.rows:
                raise ValueError(
                    f"Unknown tip row {r!r}. This rack has rows {self.rows}."
                )
            if up in wanted:
                raise ValueError(f"Row {up} requested twice in one pickup.")
            wanted.append(up)
        # Row order, so the returned spots line up with channel order.
        return sorted(wanted, key=self.rows.index)

    def _has_tip(self, row: str, col: int) -> bool:
        return self._rack.get_item(f"{row}{col}").tracker.has_tip
