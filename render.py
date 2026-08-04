import json
import os

import urllib

from agent import preview_tool
from star_sim.env import RobotEnv
from pylabrobot.resources import Resource


_ROWS = "ABCDEFGH"
_COLS = range(1, 13)


def geometry(env: RobotEnv) -> dict:
    """Walk the deck tree and collect geometry.  Every resource gets x/y/z/w/h/d."""
    boxes: list[dict] = []
    wells: list[dict] = []
    tip_spots: list[dict] = []

    def walk(resource: Resource, parent_name: str | None) -> None:
        """Recursively traverses the deck and adds to the box, well, and tip spot lists"""
        kind = type(resource).__name__
        loc = resource.get_absolute_location()
        w, h = resource.get_size_x(), resource.get_size_y()
        try:
            d = resource.get_size_z()
        except Exception:
            d = 0

        if kind == "Well":
            plate_name = resource.parent.name
            well_id = resource.name[len(f"{plate_name}_well_"):]
            wells.append({"plate": plate_name, "id": well_id,
                          "x": loc.x, "y": loc.y, "z": loc.z,
                          "w": w, "h": h, "d": d})
            return
        if kind == "TipSpot":
            rack_name = resource.parent.name
            spot_id = resource.name[len(f"{rack_name}_tipspot_"):]
            tip_spots.append({"rack": rack_name, "id": spot_id,
                              "x": loc.x, "y": loc.y, "z": loc.z,
                              "w": w, "h": h, "d": d})
            return

        if w > 0 and h > 0:
            boxes.append({
                "name": resource.name, "kind": kind, "parent": parent_name,
                "x": loc.x, "y": loc.y, "z": loc.z, "w": w, "h": h, "d": d,
            })
        for child in resource.children:
            walk(child, resource.name)

    walk(env.layout.deck, None)

    # A box's reported size_z is its full physical frame height, but its children
    # are drawn separately, so a naive solid render either hides them or floats
    # them. Two distinct cases, told apart by whether a child sits raised above
    # the box's own z (resting on an internal shelf, e.g. a rack on a carrier) or
    # flush at the same z (mounted to the box's face/side at its base, e.g. the
    # teaching tip rack on the waste block):
    #   - raised children: cap the box's height to just below the lowest one,
    #     leaving a hairline gap, so it sits visibly on top.
    #   - flush children: keep the box's real height, but carve their x-footprint
    #     out of the box's rendered width so they stand beside it, not inside it.
    _GAP = 1.0
    children_by_parent: dict[str, list[dict]] = {}
    for b in boxes:
        if b["parent"] is not None:
            children_by_parent.setdefault(b["parent"], []).append(b)

    for box in boxes:
        kids = children_by_parent.get(box["name"])
        if not kids:
            continue
        raised = [k for k in kids if k["z"] > box["z"]]
        flush = [k for k in kids if k["z"] <= box["z"]]

        if raised:
            top = min(k["z"] for k in raised)
            box["d"] = max(0.0, min(box["d"], top - box["z"] - _GAP))

        if flush:
            occupied = sorted((k["x"], k["x"] + k["w"]) for k in flush)
            box_left, box_right = box["x"], box["x"] + box["w"]
            free_spans, cursor = [], box_left
            for lo, hi in occupied:
                lo, hi = max(lo, box_left), min(hi, box_right)
                if lo > cursor:
                    free_spans.append((cursor, lo))
                cursor = max(cursor, hi)
            if cursor < box_right:
                free_spans.append((cursor, box_right))
            if free_spans:
                best = max(free_spans, key=lambda s: s[1] - s[0])
                box["x"], box["w"] = best[0], best[1] - best[0]

    # Calculate min/max xs and ys
    xs = [b["x"] for b in boxes] + [b["x"] + b["w"] for b in boxes]
    ys = [b["y"] for b in boxes] + [b["y"] + b["h"] for b in boxes]
    bounds = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}

    # Get all rail coordinates
    deck = env.layout.deck
    rails = [deck.rails_to_location(r).x for r in range(1, deck.num_rails + 1)]

    # Make sure containers know their contents
    for box in boxes:
        if box["kind"] == "Plate":
            box["well_count"] = sum(1 for w in wells if w["plate"] == box["name"])
        elif box["kind"] == "TipRack":
            box["tip_capacity"] = sum(1 for s in tip_spots if s["rack"] == box["name"])

    tip_rack_names = sorted({s["rack"] for s in tip_spots})

    # TipSpot's absolute z from PLR is recessed far below the rack's own box
    # (it's the kinematic pickup depth, not a visual surface height) — render
    # tips sitting on top of their rack's box instead.
    rack_top_z = {b["name"]: b["z"] + b["d"] for b in boxes if b["kind"] == "TipRack"}
    for spot in tip_spots:
        top = rack_top_z.get(spot["rack"])
        if top is not None:
            spot["z"], spot["d"] = top, 0

    # Row Y centres per plate (for aligning the 8 pipette channels to plate rows A-H).
    plate_row_ys: dict = {}
    for plate_name in {w["plate"] for w in wells}:
        row_ys: dict = {}
        for row in _ROWS:
            rw = [w for w in wells if w["plate"] == plate_name and w["id"][0] == row]
            if rw:
                row_ys[row] = sum(w["y"] + w["h"] / 2 for w in rw) / len(rw)
        plate_row_ys[plate_name] = row_ys

    # Find the teaching tip rack and start the heads there
    tip_rack_box = next((b for b in boxes if b["name"] == "teaching_tip_rack"), None)
    home_pos = (
        {"x": tip_rack_box["x"] + tip_rack_box["w"] / 2,
         "y": tip_rack_box["y"] + tip_rack_box["h"] / 2, "plate": None}
        if tip_rack_box else {"x": bounds["min_x"], "y": bounds["min_y"], "plate": None}
    )

    # The trash area has size_x=0 (a degenerate line, not a box) so it never makes
    # it into `boxes` — capture its centre separately for discard-tip animation.
    trash_res = env.layout.deck.get_trash_area()
    trash_loc = trash_res.get_absolute_location()
    trash_pos = {
        "x": trash_loc.x + trash_res.get_size_x() / 2,
        "y": trash_loc.y + trash_res.get_size_y() / 2,
        "plate": None,
    }

    return {
        "boxes": boxes, "wells": wells, "tip_spots": tip_spots,
        "bounds": bounds, "rails": rails, "tip_rack_names": tip_rack_names,
        "plate_row_ys": plate_row_ys, "home_pos": home_pos, "trash_pos": trash_pos,
    }


def _col_center(plate_name: str, col: int, geometry: dict) -> dict | None:
    col_str = str(col)
    pw = [w for w in geometry["wells"]
          if w["plate"] == plate_name and w["id"][1:] == col_str]
    if not pw:
        return None
    return {
        "x": sum(w["x"] + w["w"] / 2 for w in pw) / len(pw),
        "y": sum(w["y"] + w["h"] / 2 for w in pw) / len(pw),
        "plate": plate_name,
    }


def _tip_col_center(rack_name: str, col: int, geometry: dict) -> dict | None:
    col_str = str(col)
    ps = [s for s in geometry["tip_spots"]
          if s["rack"] == rack_name and s["id"][1:] == col_str]
    if not ps:
        return None
    return {
        "x": sum(s["x"] + s["w"] / 2 for s in ps) / len(ps),
        "y": sum(s["y"] + s["h"] / 2 for s in ps) / len(ps),
        "plate": None,
    }
    

# Whether tips become visible (True), get discarded (False), or are unaffected
# (omitted) on the pipette after each kind of recorded movement.
_SHOW_TIPS_AFTER = {"pick_up_tips": True, "discard_tips": False}


def _snapshot(env: RobotEnv, tip_rack_names: list[str]) -> dict:
    """Full well-contents + tip-rack state, for coloring wells/tips in the 3D view."""
    snap: dict = {}
    for plate_name in ("source_plate", "dest_plate"):
        plate = getattr(env.layout, plate_name)
        plate_wells: dict = {}
        for row in _ROWS:
            for col in _COLS:
                well_id = f"{row}{col}"
                volume = plate.get_item(well_id).tracker.get_used_volume()
                contents = env.plate_map.get_well(plate_name, well_id)
                entry = {"volume": round(volume, 1)}
                if contents and contents.compounds:
                    entry["compounds"] = [
                        {"name": c.name, "conc": round(c.concentration_um, 3)}
                        for c in contents.compounds
                    ]
                if contents and contents.cells:
                    entry["cells"] = contents.cells
                plate_wells[well_id] = entry
        snap[plate_name] = plate_wells

    tip_racks: dict = {}
    for rack_name in tip_rack_names:
        rack = env.layout.deck.get_resource(rack_name)
        tip_racks[rack_name] = {
            spot.name[len(f"{rack_name}_tipspot_"):]: spot.tracker.has_tip
            for spot in rack.children
        }
    snap["tip_racks"] = tip_racks
    return snap


def _movement_target(kind: str, info: dict, geometry: dict) -> dict | None:
    if kind == "pick_up_tips":
        return _tip_col_center(info["rack"], info["col"], geometry)
    if kind == "discard_tips":
        return geometry["trash_pos"]
    if kind in ("aspirate", "dispense"):
        return _col_center(info["plate"], info["col"], geometry)
    return None


def _initial_frame(geometry: dict, env: RobotEnv, tip_rack_names: list[str]) -> dict:
    return {"step": -1, "tool": "initial", "preview": "", "message": None,
            "pipette_pos": geometry["home_pos"], "movements": [],
            "snapshot": _snapshot(env, tip_rack_names)}


def _done_frame(record_len: int, env: RobotEnv, tip_rack_names: list[str], final_text: str) -> dict:
    return {"step": record_len, "tool": "done", "preview": "", "message": final_text,
            "pipette_pos": None, "movements": [],
            "snapshot": _snapshot(env, tip_rack_names)}


class LiveCapture:
    """Builds the frame structure by listening to a live
    RobotEnv's own movement events as they really happen — The state needed to
    render each step (positions, snapshots) was already computed once by the
    real dispatch; this just packages it instead of recomputing it.

    Usage: construct once right after the real env's setup(), then call
    record_step() after every real _dispatch() call, and finish() once at the
    end. close() detaches the listener (does not touch the env's lifecycle).
    """

    def __init__(self, env: RobotEnv, geometry: dict) -> None:
        self.env = env
        self.geometry = geometry
        self.tip_rack_names = geometry["tip_rack_names"]
        self.frames: list[dict] = [_initial_frame(geometry, env, self.tip_rack_names)]
        self._pending: list[dict] = []
        self._step_idx = 0
        env.add_movement_listener(self._on_movement)

    def _on_movement(self, kind: str, info: dict) -> None:
        self._pending.append({
            "kind": kind,
            "pipette_pos": _movement_target(kind, info, self.geometry),
            "show_tips": _SHOW_TIPS_AFTER.get(kind),
            "snapshot": _snapshot(self.env, self.tip_rack_names),
        })

    def record_step(self, name: str, args: dict, message: str | None, error: str | None) -> None:
        movements, self._pending = self._pending, []
        step_idx, self._step_idx = self._step_idx, self._step_idx + 1
        if name == "observe":
            return
        self.frames.append({
            "step": step_idx,
            "tool": name,
            "preview": preview_tool(name, args),
            "message": message,
            "pipette_pos": movements[-1]["pipette_pos"] if movements else None,
            "movements": movements,
            "error": error,
            "snapshot": _snapshot(self.env, self.tip_rack_names),
        })

    def finish(self, final_text: str | None) -> None:
        if final_text:
            self.frames.append(_done_frame(self._step_idx, self.env, self.tip_rack_names, final_text))

    def close(self) -> None:
        self.env.remove_movement_listener(self._on_movement)

    # async def teardown(self) -> None:
    #     await self.env.teardown()


def render_html(geometry: dict, frames: list[dict], title: str,
                live: bool = False,
                frames_url: str = "",
                use_hardware: bool = False) -> str:
    js = _fetch_js()
    three_js         = (js or {}).get("three.js", "")
    orbit_js         = (js or {}).get("OrbitControls.js", "")
    max_step = len(frames) - 1
    # Live mode: default to the newest known frame so a still-running
    # experiment's tab opens already caught up, rather than at frame 0.
    start_step = max_step if live else 0
    template = open(os.path.join(os.path.dirname(__file__), "static", "app.html")).read()
    return (
        template
        .replace("__TITLE__", title)
        .replace("__MAX_STEP__", str(max_step))
        .replace("__START_STEP__", str(start_step))
        .replace("__LIVE__", "true" if live else "false")
        .replace("__USE_HARDWARE__", "true" if use_hardware else "false")
        .replace("__FRAMES_URL__", frames_url)
        .replace("__GEOMETRY_JSON__", json.dumps(geometry))
        .replace("__FRAMES_JSON__", json.dumps(frames))
        .replace("__THREEJS__", three_js)
        .replace("__ORBITCONTROLS__", orbit_js)
    )
    
    
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".viz_cache")
_THREE_URLS = {
    "three.js":         "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
    "OrbitControls.js": "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
}

def _fetch_js() -> dict[str, str]:
    """Return {filename: js_content}, downloading from CDN and caching locally."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    result = {}
    for name, url in _THREE_URLS.items():
        cache_path = os.path.join(_CACHE_DIR, name)
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                result[name] = f.read()
        else:
            print(f"Downloading {name} …")
            with urllib.request.urlopen(url) as r:
                content = r.read().decode()
            with open(cache_path, "w") as f:
                f.write(content)
            result[name] = content
    return result