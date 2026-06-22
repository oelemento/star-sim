"""TipManager: safe, cursor-based tip acquisition for the 8-channel head.

The Hamilton 96-tip rack has 12 columns (A-H x 1-12). With an 8-channel head
you consume one full column per pickup. TipManager tracks which columns have
been used and advances a cursor so callers never reach for an empty position.
"""

from pylabrobot.resources import TipRack


class OutOfTipsError(Exception):
    pass


class TipManager:
    """Cursor into a TipRack that advances one column per pickup.

    Usage::

        mgr = TipManager(layout.tip_rack)
        tips = mgr.next_column()   # returns rack["A1:H1"] first call
        await lh.pick_up_tips(tips)
        ...
        await lh.drop_tips(tips)
        tips = mgr.next_column()   # returns rack["A2:H2"] next call
    """

    def __init__(self, rack: TipRack) -> None:
        self._rack = rack
        self._cursor = 0  # next column index (0-based)
        self.last_column: int | None = None  # column number (1-based) returned by the last next_column() call

    @property
    def remaining(self) -> int:
        """Columns with all tips still present, from the cursor onward."""
        n_cols = self._rack.num_items_x
        count = 0
        for col_idx in range(self._cursor, n_cols):
            col_num = col_idx + 1
            if self._column_full(col_num):
                count += 1
        return count

    def next_column(self):
        """Return the tip spots for the next unused column.

        Skips any column that is partially or fully spent (handles cases where
        the rack was pre-used before this manager was created).

        Raises:
            OutOfTipsError: if no full column remains.
        """
        n_cols = self._rack.num_items_x
        while self._cursor < n_cols:
            col_num = self._cursor + 1
            self._cursor += 1
            if self._column_full(col_num):
                self.last_column = col_num
                return self._rack[f"A{col_num}:H{col_num}"]
        raise OutOfTipsError(f"Tip rack '{self._rack.name}' is exhausted")

    def reset(self) -> None:
        """Reset the cursor to column 1 (does not restore physical tip state)."""
        self._cursor = 0

    def _column_full(self, col_num: int) -> bool:
        n_rows = self._rack.num_items_y
        rows = "ABCDEFGH"[:n_rows]
        return all(
            self._rack.get_item(f"{row}{col_num}").tracker.has_tip
            for row in rows
        )
