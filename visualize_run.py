#!/usr/bin/env python3
"""Render a recorded run as a step-through deck visualization.

Replays a runs/<timestamp>.json file against a fresh, silent RobotEnv (same
dispatch logic the agent uses), snapshotting every well's volume, compound,
and concentration (plus tip-rack occupancy) after each step. Writes a
self-contained HTML file with a slider to scrub through the experiment,
drawn as a top-down deck layout (carriers, tip rack, plates) in the spirit
of PyLabRobot's built-in visualizer, but colored by compound/concentration
instead of bare volume.

Usage:
    python visualize_run.py runs/2026-06-12T16-43-33.json
    python visualize_run.py runs/2026-06-12T16-43-33.json --open
"""

import argparse
import asyncio
import json
import os
import webbrowser

from agent import _dispatch, _preview
from star_sim import RobotEnv

_ROWS = "ABCDEFGH"
_COLS = range(1, 13)


def _geometry(env: RobotEnv) -> dict:
    """Walk the deck tree once and capture static layout (positions don't change
    during a run). Containers (carriers, holders, racks, plates) become drawable
    boxes; Wells and TipSpots become leaf cells keyed by their parent + row/col id."""
    boxes: list[dict] = []
    wells: list[dict] = []
    tip_spots: list[dict] = []

    def walk(resource, parent_name: str | None) -> None:
        kind = type(resource).__name__
        loc = resource.get_absolute_location()
        w, h = resource.get_size_x(), resource.get_size_y()

        if kind == "Well":
            plate_name = resource.parent.name
            well_id = resource.name[len(f"{plate_name}_well_"):]
            wells.append({"plate": plate_name, "id": well_id, "x": loc.x, "y": loc.y, "w": w, "h": h})
            return
        if kind == "TipSpot":
            rack_name = resource.parent.name
            spot_id = resource.name[len(f"{rack_name}_tipspot_"):]
            tip_spots.append({"rack": rack_name, "id": spot_id, "x": loc.x, "y": loc.y, "w": w, "h": h})
            return

        if w > 0 and h > 0:
            boxes.append({
                "name": resource.name, "kind": kind, "parent": parent_name,
                "x": loc.x, "y": loc.y, "w": w, "h": h,
            })
        for child in resource.children:
            walk(child, resource.name)

    walk(env.layout.deck, None)

    xs = [b["x"] for b in boxes] + [b["x"] + b["w"] for b in boxes]
    ys = [b["y"] for b in boxes] + [b["y"] + b["h"] for b in boxes]
    bounds = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}

    # The Hamilton deck is physically divided into numbered rail slots that
    # carriers click into; draw them as track lines, same as the real layout.
    deck = env.layout.deck
    rails = [deck.rails_to_location(r).x for r in range(1, deck.num_rails + 1)]

    # Static capacities for the tree's labels (e.g. "96 wells", "12/96 tips" — the
    # numerator is filled in per-frame from the snapshot).
    for box in boxes:
        if box["kind"] == "Plate":
            box["well_count"] = sum(1 for well in wells if well["plate"] == box["name"])
        elif box["kind"] == "TipRack":
            box["tip_capacity"] = sum(1 for spot in tip_spots if spot["rack"] == box["name"])

    tip_rack_names = sorted({spot["rack"] for spot in tip_spots})

    return {
        "boxes": boxes, "wells": wells, "tip_spots": tip_spots,
        "bounds": bounds, "rails": rails, "tip_rack_names": tip_rack_names,
    }


def _snapshot(env: RobotEnv, tip_rack_names: list[str]) -> dict:
    """Capture volume + compound/concentration for every well, plus tip occupancy
    for every tip rack on the deck (not just the one the protocol actively uses)."""
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


async def _replay(record: list[tuple[str, dict]]) -> tuple[dict, list[dict]]:
    """Run each recorded tool call through a fresh env, snapshotting after each step."""
    env = RobotEnv(use_hardware=False)
    await env.setup()
    try:
        geometry = _geometry(env)
        tip_rack_names = geometry["tip_rack_names"]
        frames = [{"step": -1, "tool": "initial", "preview": "", "snapshot": _snapshot(env, tip_rack_names)}]
        for i, (name, args) in enumerate(record):
            result = await _dispatch(name, args, env, step_delay=0.0, silent=True)
            frames.append({
                "step": i,
                "tool": name,
                "preview": _preview(name, args),
                "error": result.get("error"),
                "snapshot": _snapshot(env, tip_rack_names),
            })
    finally:
        await env.teardown()
    return geometry, frames


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Run replay: {title}</title>
<style>
  html, body {{ height: 100%; }}
  body {{ font-family: -apple-system, sans-serif; margin: 0; background: #fafafa; display: flex; flex-direction: row; height: 100vh; overflow: hidden; font-size: 13px; }}
  h1 {{ font-size: 1.1em; color: #333; margin: 0 10px 0 0; display: inline; }}
  #deck-wrap {{ flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; height: 100vh; }}
  #deck-svg {{ border: 1px solid #ddd; background: #fff; width: 100%; flex: 1 1 auto; min-height: 0; display: block; }}
  text.rail-label {{ font-size: 9px; fill: #888; font-family: monospace; text-anchor: middle; }}
  text.scale-label {{ font-size: 9px; fill: #444; font-family: monospace; text-anchor: middle; }}
  rect.box-rect.hl {{ stroke: #e6b800 !important; stroke-width: 2.5 !important; }}
  #right {{ flex: 0 0 280px; height: 100vh; overflow-y: auto; border-left: 1px solid #ddd; background: #fff; box-sizing: border-box; }}
  #header {{ padding: 8px 10px; border-bottom: 1px solid #ddd; position: sticky; top: 0; background: #fff; z-index: 1; }}
  #header-top {{ margin-bottom: 4px; }}
  #goal {{ font-size: 1em; color: #666; }}
  #controls {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; flex-wrap: wrap; }}
  #step-label {{ font-size: 1em; font-weight: 600; white-space: nowrap; }}
  #args {{ font-size: 0.85em; color: #555; font-family: monospace; margin: 0 0 4px; white-space: pre; overflow-x: auto; }}
  #slider {{ width: 100%; flex-shrink: 1; }}
  .error {{ color: #c00; font-weight: 600; }}
  #tree-panel {{ padding: 8px 6px; font-size: 1em; }}
  #tree-panel .tree {{ list-style: none; margin: 0; padding: 0; }}
  #tree-panel li {{ padding: 1px 4px; border-radius: 3px; cursor: default; white-space: nowrap; }}
  #tree-panel li.hl {{ background: #fff3b0; }}
  .tree-name {{ font-weight: 600; }}
  .tree-info {{ color: #888; font-size: 0.9em; }}
  .tree-empty {{ color: #aaa; font-style: italic; }}
  #legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.9em; color: #333; }}
  #legend .swatch {{ display: inline-block; width: 11px; height: 11px; margin-right: 4px; vertical-align: middle; border: 1px solid #999; border-radius: 2px; }}
</style>
</head>
<body>
  <div id="deck-wrap">
    <svg id="deck-svg" viewBox="{viewbox}" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
  <div id="right">
    <div id="header">
      <div id="header-top"><h1>Replay: {title}</h1></div>
      <div id="goal">{goal}</div>
      <div id="controls">
        <input id="slider" type="range" min="0" max="{max_step}" value="0">
        <span id="step-label"></span>
      </div>
      <pre id="args"></pre>
      <div id="legend"></div>
    </div>
    <div id="tree-panel"></div>
  </div>

<script>
const GEOMETRY = {geometry_json};
const FRAMES = {frames_json};

const BOX_STYLE = {{
  HamiltonSTARDeck: {{fill: "none", stroke: "#000", strokeWidth: 1.5}},
  TipCarrier:       {{fill: "#7c5b73", stroke: "#4a2f45", strokeWidth: 1}},
  PlateCarrier:     {{fill: "#5f7290", stroke: "#33415a", strokeWidth: 1}},
  ResourceHolder:   {{fill: "#cfd6e0", stroke: "#8893a1", strokeWidth: 0.5}},
  PlateHolder:      {{fill: "#cfd6e0", stroke: "#8893a1", strokeWidth: 0.5}},
  TipRack:          {{fill: "#23232b", stroke: "#000", strokeWidth: 1}},
  Plate:            {{fill: "#1c1c22", stroke: "#000", strokeWidth: 1}},
}};
const DEFAULT_STYLE = {{fill: "#b8a23a", stroke: "#8a7a28", strokeWidth: 1}};

function colorForWell(entry) {{
  if (!entry || entry.volume <= 0) return "#fff";
  if (entry.compounds && entry.compounds.length) {{
    const c = entry.compounds[0];
    let hash = 0;
    for (const ch of c.name) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
    const pct = Math.min(c.conc / 200, 1);
    const light = 80 - pct * 50;
    return `hsl(${{hash}}, 70%, ${{light}}%)`;
  }}
  if (entry.cells) return "#a8d4ff";
  return "#888"; // volume only, no identity (e.g. buffer)
}}

function wellTooltip(plate, id, entry) {{
  let s = `${{plate}} ${{id}}`;
  if (!entry) return s;
  s += `: ${{entry.volume}} µL`;
  if (entry.compounds) for (const c of entry.compounds) s += `  ${{c.name}} ${{c.conc}}µM`;
  if (entry.cells) s += `  ${{entry.cells}}`;
  return s;
}}

const DECK_H = GEOMETRY.bounds.max_y - GEOMETRY.bounds.min_y;
const OX = GEOMETRY.bounds.min_x, OY = GEOMETRY.bounds.min_y;
// Flip Y (PLR: origin bottom-left, Y up) into SVG space (origin top-left, Y down).
function fx(x) {{ return x - OX; }}
function fy(y, h) {{ return DECK_H - (y - OY) - h; }}

function svgEl(tag, attrs) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}}

function renderRails(svg) {{
  // Rail tracks the carriers click into, plus a rail-number ruler and a 100mm scale bar.
  for (let i = 0; i < GEOMETRY.rails.length; i++) {{
    const railX = fx(GEOMETRY.rails[i]);
    svg.appendChild(svgEl("line", {{
      x1: railX, y1: 0, x2: railX, y2: DECK_H,
      stroke: "#ddd", "stroke-width": 0.5,
    }}));
    const rail = i + 1;
    svg.appendChild(svgEl("line", {{
      x1: railX, y1: DECK_H + 2, x2: railX, y2: DECK_H + 8,
      stroke: "#888", "stroke-width": 0.5,
    }}));
    if (rail === 1 || rail % 5 === 0) {{
      const t = svgEl("text", {{x: railX, y: DECK_H + 20, class: "rail-label"}});
      t.textContent = rail;
      svg.appendChild(t);
    }}
  }}

  const barY = DECK_H + 40;
  const barX0 = fx(GEOMETRY.bounds.max_x) - 100;
  svg.appendChild(svgEl("line", {{
    x1: barX0, y1: barY, x2: barX0 + 100, y2: barY, stroke: "#444", "stroke-width": 1,
  }}));
  for (const x of [barX0, barX0 + 100]) {{
    svg.appendChild(svgEl("line", {{x1: x, y1: barY - 3, x2: x, y2: barY + 3, stroke: "#444", "stroke-width": 1}}));
  }}
  const label = svgEl("text", {{x: barX0 + 50, y: barY + 14, class: "scale-label"}});
  label.textContent = "100 mm";
  svg.appendChild(label);
}}

function highlightOn(name) {{
  const rect = document.querySelector(`#deck-svg rect[data-name="${{name}}"]`);
  if (rect) rect.classList.add("hl");
  const li = document.querySelector(`#tree-panel li[data-name="${{name}}"]`);
  if (li) li.classList.add("hl");
}}
function highlightOff(name) {{
  const rect = document.querySelector(`#deck-svg rect[data-name="${{name}}"]`);
  if (rect) rect.classList.remove("hl");
  const li = document.querySelector(`#tree-panel li[data-name="${{name}}"]`);
  if (li) li.classList.remove("hl");
}}

function renderDeck(frame) {{
  const svg = document.getElementById("deck-svg");
  svg.innerHTML = "";
  renderRails(svg);

  for (const box of GEOMETRY.boxes) {{
    const style = BOX_STYLE[box.kind] || DEFAULT_STYLE;
    const rect = svgEl("rect", {{
      x: fx(box.x), y: fy(box.y, box.h), width: box.w, height: box.h,
      fill: style.fill, stroke: style.stroke, "stroke-width": style.strokeWidth,
      class: "box-rect", "data-name": box.name,
    }});
    rect.addEventListener("mouseenter", () => highlightOn(box.name));
    rect.addEventListener("mouseleave", () => highlightOff(box.name));
    svg.appendChild(rect);
  }}

  const wellMap = {{}};
  for (const plate of ["source_plate", "dest_plate"]) {{
    for (const [id, entry] of Object.entries(frame.snapshot[plate])) {{
      wellMap[plate + ":" + id] = entry;
    }}
  }}
  for (const w of GEOMETRY.wells) {{
    const entry = wellMap[w.plate + ":" + w.id];
    const rect = svgEl("rect", {{
      x: fx(w.x), y: fy(w.y, w.h), width: w.w, height: w.h,
      fill: colorForWell(entry), stroke: "#555", "stroke-width": 0.3,
    }});
    const title = svgEl("title", {{}});
    title.textContent = wellTooltip(w.plate, w.id, entry);
    rect.appendChild(title);
    svg.appendChild(rect);
  }}

  const tipMap = {{}};
  for (const [rackName, posState] of Object.entries(frame.snapshot.tip_racks || {{}})) {{
    for (const [pos, present] of Object.entries(posState)) tipMap[rackName + ":" + pos] = present;
  }}
  for (const t of GEOMETRY.tip_spots) {{
    const present = tipMap[t.rack + ":" + t.id];
    const cx = fx(t.x) + t.w / 2, cy = fy(t.y, t.h) + t.h / 2;
    const circle = svgEl("circle", {{
      cx: cx, cy: cy, r: Math.min(t.w, t.h) / 2.4,
      fill: present ? "#1abc9c" : "#fff",
      stroke: "#555", "stroke-width": 0.3,
    }});
    const title = svgEl("title", {{}});
    title.textContent = `${{t.rack}} ${{t.id}}: ${{present ? "tip" : "empty"}}`;
    circle.appendChild(title);
    svg.appendChild(circle);
  }}
}}

const ROOT_KEY = "__root__";
const TREE_CHILDREN = {{}};
for (const box of GEOMETRY.boxes) {{
  const key = box.parent === null ? ROOT_KEY : box.parent;
  (TREE_CHILDREN[key] = TREE_CHILDREN[key] || []).push(box);
}}

function treeNodeHTML(box, frame, depth) {{
  // Sort children top-of-deck first (highest physical Y = visually top after Y-flip).
  const children = (TREE_CHILDREN[box.name] || []).slice().sort((a, b) => b.y - a.y);
  const isEmptyHolder = (box.kind === "ResourceHolder" || box.kind === "PlateHolder") && children.length === 0;

  let info = box.kind;
  if (box.kind === "Plate") info = `${{box.well_count}} wells`;
  if (box.kind === "TipRack") {{
    const state = (frame.snapshot.tip_racks && frame.snapshot.tip_racks[box.name]) || {{}};
    const occupied = Object.values(state).filter(Boolean).length;
    info = `${{occupied}}/${{box.tip_capacity}} tips`;
  }}

  const label = isEmptyHolder
    ? `<span class="tree-empty">&lt;empty&gt;</span>`
    : `<span class="tree-name">${{box.name}}</span> <span class="tree-info">${{info}}</span>`;
  let html = `<li data-name="${{box.name}}" style="padding-left:${{depth * 14}}px">${{label}}</li>`;
  for (const child of children) html += treeNodeHTML(child, frame, depth + 1);
  return html;
}}

function renderTree(frame) {{
  const roots = TREE_CHILDREN[ROOT_KEY] || [];
  const html = roots.map(r => treeNodeHTML(r, frame, 0)).join("");
  const panel = document.getElementById("tree-panel");
  panel.innerHTML = `<ul class="tree">${{html}}</ul>`;
  panel.querySelectorAll("li").forEach(li => {{
    const name = li.dataset.name;
    li.addEventListener("mouseenter", () => highlightOn(name));
    li.addEventListener("mouseleave", () => highlightOff(name));
  }});
}}

function renderLegend(frame) {{
  const names = new Set();
  for (const plate of ["source_plate", "dest_plate"]) {{
    for (const entry of Object.values(frame.snapshot[plate])) {{
      if (entry.compounds) for (const c of entry.compounds) names.add(c.name);
    }}
  }}
  const chips = [];
  for (const name of names) {{
    let hash = 0;
    for (const ch of name) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
    chips.push(`<span><span class="swatch" style="background:hsl(${{hash}},70%,55%)"></span>${{name}} (darker = higher conc.)</span>`);
  }}
  chips.push(`<span><span class="swatch" style="background:#a8d4ff"></span>cells</span>`);
  chips.push(`<span><span class="swatch" style="background:#888"></span>volume only (e.g. buffer)</span>`);
  chips.push(`<span><span class="swatch" style="background:#1abc9c;border-radius:50%"></span>tip present</span>`);
  document.getElementById("legend").innerHTML = chips.join("");
}}

function render(i) {{
  const frame = FRAMES[i];
  document.getElementById("step-label").innerHTML =
    `Step ${{frame.step}} / ${{FRAMES.length - 2}}: <b>${{frame.tool}}</b>` +
    (frame.error ? ` <span class="error">ERROR: ${{frame.error}}</span>` : "");
  document.getElementById("args").textContent = frame.preview || "";
  renderDeck(frame);
  renderTree(frame);
  renderLegend(frame);
}}

const slider = document.getElementById("slider");
slider.addEventListener("input", () => render(parseInt(slider.value, 10)));
render(0);
</script>
</body>
</html>
"""


def render_html(goal: str, geometry: dict, frames: list[dict], title: str) -> str:
    goal = " ".join(goal.split())  # collapse newlines/whitespace into a single line
    b = geometry["bounds"]
    margin = 10
    bottom_margin = 55  # room for the rail-number ruler + 100mm scale bar drawn below the deck
    viewbox = (
        f"{-margin} {-margin} "
        f"{b['max_x'] - b['min_x'] + 2*margin} "
        f"{b['max_y'] - b['min_y'] + margin + bottom_margin}"
    )
    return _HTML_TEMPLATE.format(
        title=title,
        goal=goal,
        max_step=len(frames) - 1,
        viewbox=viewbox,
        geometry_json=json.dumps(geometry),
        frames_json=json.dumps(frames),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", help="path to runs/<timestamp>.json")
    parser.add_argument("--open", action="store_true", help="open the HTML file in a browser when done")
    args = parser.parse_args()

    with open(args.run_path) as f:
        data = json.load(f)
    record = [(name, call_args) for name, call_args in data["record"]]

    geometry, frames = asyncio.run(_replay(record))

    title = os.path.basename(args.run_path)
    html = render_html(data.get("goal", ""), geometry, frames, title)
    out_path = os.path.splitext(args.run_path)[0] + ".viz.html"
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Wrote {out_path}")
    if args.open:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
