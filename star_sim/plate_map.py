"""PlateMap: semantic contents of the wells on the deck.

PyLabRobot's volume tracker records physical µL. PlateMap records chemical
identity and concentration — what compound(s) and cells are in each well.
It is updated by the agent via the update_column / update_well tools and lives
on RobotEnv so observe() can include it alongside the physical state.

A well can hold multiple compounds simultaneously (e.g. drug A + drug B in
a synergy assay) plus a cell line with density.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_ROWS = "ABCDEFGH"


@dataclass
class Compound:
    name: str
    concentration_um: float  # µM


@dataclass
class WellContents:
    compounds: list[Compound] = field(default_factory=list)
    cells: Optional[str] = None           # cell line name, e.g. "HeLa"
    cell_density_per_ml: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        d: dict = {}
        if self.compounds:
            d["compounds"] = [
                {"name": c.name, "concentration_um": c.concentration_um}
                for c in self.compounds
            ]
        if self.cells is not None:
            d["cells"] = self.cells
            if self.cell_density_per_ml is not None:
                d["cell_density_per_ml"] = self.cell_density_per_ml
        if self.notes:
            d["notes"] = self.notes
        return d


def mix_contents(
    src: "WellContents | None", src_vol: float,
    dst: "WellContents | None", dst_vol: float,
    transfer_cells: bool = False,
) -> "WellContents":
    """Return a new WellContents reflecting volume-weighted mixing of src into dst.

    Compounds always follow mass conservation: new_c = (v_src*c_src + v_dst*c_dst) / total.
    Cells propagate only when transfer_cells=True — i.e. the agent explicitly resuspended
    before aspirating. When False, dst cell identity is preserved unchanged.
    """
    total = src_vol + dst_vol
    if total == 0:
        return WellContents()

    names: set[str] = set()
    if src:
        names.update(c.name for c in src.compounds)
    if dst:
        names.update(c.name for c in dst.compounds)

    result: list[Compound] = []
    for name in sorted(names):
        c_src = next((c.concentration_um for c in (src.compounds if src else []) if c.name == name), 0.0)
        c_dst = next((c.concentration_um for c in (dst.compounds if dst else []) if c.name == name), 0.0)
        new_c = (src_vol * c_src + dst_vol * c_dst) / total
        if new_c > 1e-9:
            result.append(Compound(name=name, concentration_um=new_c))

    if transfer_cells:
        src_line = src.cells if src else None
        dst_line = dst.cells if dst else None
        src_density = (src.cell_density_per_ml or 0.0) if src else 0.0
        dst_density = (dst.cell_density_per_ml or 0.0) if dst else 0.0
        if src_line and dst_line and src_line != dst_line:
            new_line: Optional[str] = f"{src_line}+{dst_line}"
        else:
            new_line = src_line or dst_line
        new_density = (src_vol * src_density + dst_vol * dst_density) / total
        cells = new_line if (new_line and new_density > 0) else None
        cell_density: Optional[float] = new_density if new_density > 0 else None
    else:
        cells = dst.cells if dst else None
        cell_density = dst.cell_density_per_ml if dst else None

    return WellContents(
        compounds=result,
        cells=cells,
        cell_density_per_ml=cell_density,
        notes=dst.notes if dst else "",
    )


class PlateMap:
    """Semantic contents of every named plate on the deck.

    Keyed by plate name (matching DeckLayout field names) then well id ("A1" etc.).
    Wells with no recorded contents are absent from to_dict() output.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, WellContents]] = {}

    def set_well(self, plate: str, well: str, contents: WellContents) -> None:
        self._data.setdefault(plate, {})[well] = contents

    def set_column(self, plate: str, col: int, contents: WellContents) -> None:
        """Set the same contents for all 8 wells of a column at once."""
        for row in _ROWS:
            self.set_well(plate, f"{row}{col}", contents)

    def get_well(self, plate: str, well: str) -> Optional[WellContents]:
        return self._data.get(plate, {}).get(well)

    def to_dict(self) -> dict:
        result: dict = {}
        for plate, wells in self._data.items():
            nonempty = {w: c.to_dict() for w, c in wells.items() if c.to_dict()}
            if nonempty:
                result[plate] = nonempty
        return result
