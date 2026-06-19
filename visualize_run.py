#!/usr/bin/env python3
"""Render a recorded run as a 3D step-through deck visualization.

Replays a runs/<timestamp>.json file, snapshotting compound/concentration and
tip occupancy after each step, and writes a self-contained HTML file with a
Three.js 3D scene.  A Play/Pause button animates the pipette arm across the
deck step by step; Back resets to the previous frame instantly.

Usage:
    python visualize_run.py runs/2026-06-12T16-43-33.json
    python visualize_run.py runs/2026-06-12T16-43-33.json --open
"""

import argparse
import asyncio
import json
import os
import urllib.request
import webbrowser

from agent import _dispatch, _preview
from star_sim import RobotEnv

_ROWS = "ABCDEFGH"
_COLS = range(1, 13)


def _geometry(env: RobotEnv) -> dict:
    """Walk the deck tree and collect geometry.  Every resource gets x/y/z/w/h/d."""
    boxes: list[dict] = []
    wells: list[dict] = []
    tip_spots: list[dict] = []

    def walk(resource, parent_name: str | None) -> None:
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
            # HamiltonSTARDeck reports size_z as its full work-envelope height (900 mm).
            # Draw it as a thin slab so the camera isn't enclosed inside it.
            visual_d = 10 if kind == "HamiltonSTARDeck" else d
            boxes.append({
                "name": resource.name, "kind": kind, "parent": parent_name,
                "x": loc.x, "y": loc.y, "z": loc.z, "w": w, "h": h, "d": visual_d,
            })
        for child in resource.children:
            walk(child, resource.name)

    walk(env.layout.deck, None)

    xs = [b["x"] for b in boxes] + [b["x"] + b["w"] for b in boxes]
    ys = [b["y"] for b in boxes] + [b["y"] + b["h"] for b in boxes]
    bounds = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}

    deck = env.layout.deck
    rails = [deck.rails_to_location(r).x for r in range(1, deck.num_rails + 1)]

    for box in boxes:
        if box["kind"] == "Plate":
            box["well_count"] = sum(1 for w in wells if w["plate"] == box["name"])
        elif box["kind"] == "TipRack":
            box["tip_capacity"] = sum(1 for s in tip_spots if s["rack"] == box["name"])

    tip_rack_names = sorted({s["rack"] for s in tip_spots})

    # Row Y centres per plate (for aligning the 8 pipette channels to plate rows A-H).
    plate_row_ys: dict = {}
    for plate_name in {w["plate"] for w in wells}:
        row_ys: dict = {}
        for row in _ROWS:
            rw = [w for w in wells if w["plate"] == plate_name and w["id"][0] == row]
            if rw:
                row_ys[row] = sum(w["y"] + w["h"] / 2 for w in rw) / len(rw)
        plate_row_ys[plate_name] = row_ys

    tip_rack_box = next((b for b in boxes if b["name"] == "tip_rack"), None)
    home_pos = (
        {"x": tip_rack_box["x"] + tip_rack_box["w"] / 2,
         "y": tip_rack_box["y"] + tip_rack_box["h"] / 2, "plate": None}
        if tip_rack_box else {"x": bounds["min_x"], "y": bounds["min_y"], "plate": None}
    )

    return {
        "boxes": boxes, "wells": wells, "tip_spots": tip_spots,
        "bounds": bounds, "rails": rails, "tip_rack_names": tip_rack_names,
        "plate_row_ys": plate_row_ys, "home_pos": home_pos,
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


def _pipette_pos(name: str, args: dict, geometry: dict) -> dict | None:
    match name:
        case "propose_prime":
            reagents = args.get("reagents", [])
            if reagents:
                r = reagents[0]
                return _col_center(r.get("plate", "source_plate"), r.get("col", 1), geometry)
        case "column_transfer":
            return _col_center(args["dst_plate"], args["dst_col"], geometry)
        case "multi_dispense":
            cols = args.get("dst_cols", [])
            if cols:
                return _col_center(args["dst_plate"], cols[-1], geometry)
        case "serial_transfer":
            return _col_center(args["plate"], args["end_col"], geometry)
        case "mix_column":
            return _col_center(args["plate"], args["col"], geometry)
    return None


def _snapshot(env: RobotEnv, tip_rack_names: list[str]) -> dict:
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
    env = RobotEnv(use_hardware=False)
    await env.setup()
    try:
        geometry = _geometry(env)
        tip_rack_names = geometry["tip_rack_names"]
        frames = [{"step": -1, "tool": "initial", "preview": "",
                   "pipette_pos": geometry["home_pos"],
                   "snapshot": _snapshot(env, tip_rack_names)}]
        for i, (name, args) in enumerate(record):
            result = await _dispatch(name, args, env, step_delay=0.0, silent=True)
            frames.append({
                "step": i,
                "tool": name,
                "preview": _preview(name, args),
                "pipette_pos": _pipette_pos(name, args, geometry),
                "error": result.get("error"),
                "snapshot": _snapshot(env, tip_rack_names),
            })
    finally:
        await env.teardown()
    return geometry, frames


# ---------------------------------------------------------------------------
# HTML template — uses __MARKER__ substitution (no Python .format() escaping)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Replay 3D: __TITLE__</title>
<style>
  html { height: 100%; }
  body { height: 100%; margin: 0; font-family: -apple-system, sans-serif;
         background: #f0f0f0; font-size: 13px; display: flex; flex-direction: row; overflow: hidden; }
  #canvas-container { flex: 1 0 0; min-width: 0; height: 100vh; position: relative; }
  #canvas-container canvas { display: block; width: 100% !important; height: 100% !important; }
  #right { flex: 0 0 280px; height: 100vh; overflow-y: auto; border-left: 1px solid #ddd;
           background: #fff; color: #333; box-sizing: border-box; }
  #header { padding: 8px 10px; border-bottom: 1px solid #ddd; position: sticky; top: 0;
            background: #fff; z-index: 1; }
  #header-top { margin-bottom: 4px; }
  h1 { font-size: 1.05em; color: #222; margin: 0 0 3px; }
  #goal { font-size: 0.9em; color: #666; margin-bottom: 6px; }
  #controls { display: flex; align-items: center; gap: 5px; margin-bottom: 5px; flex-wrap: wrap; }
  #slider { width: 100%; margin-top: 4px; }
  #step-label { font-size: 0.95em; font-weight: 600; color: #333; margin-bottom: 2px; }
  #args { font-size: 0.82em; color: #555; font-family: monospace; margin: 0 0 5px;
          white-space: pre; overflow-x: auto; }
  .error { color: #c00; font-weight: 600; }
  button { background: #f0f0f0; color: #333; border: 1px solid #bbb; border-radius: 3px;
           padding: 3px 8px; cursor: pointer; font-size: 0.9em; }
  button:hover { background: #e0e0e0; }
  button#btn-play { min-width: 72px; }
  #legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 0.85em; color: #444; }
  #legend .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 4px;
                    vertical-align: middle; border: 1px solid #999; border-radius: 2px; }
  #tree-panel { padding: 8px 6px; font-size: 0.95em; }
  #tree-panel ul { list-style: none; margin: 0; padding: 0; }
  #tree-panel li { padding: 1px 4px; border-radius: 3px; cursor: default; white-space: nowrap; }
  #tree-panel li.hl { background: #fff3b0; }
  .tree-name { font-weight: 600; color: #222; }
  .tree-info { color: #888; font-size: 0.88em; }
  .tree-empty { color: #bbb; font-style: italic; }
</style>
</head>
<body>
<div id="canvas-container"></div>
<div id="right">
  <div id="header">
    <div id="header-top"><h1>Replay: __TITLE__</h1></div>
    <div id="goal">__GOAL__</div>
    <div id="controls">
      <button id="btn-back">&#9664;</button>
      <button id="btn-play">&#9654; Play</button>
      <button id="btn-fwd">&#9654;|</button>
    </div>
    <input id="slider" type="range" min="0" max="__MAX_STEP__" value="0">
    <div id="step-label"></div>
    <pre id="args"></pre>
    <div id="legend"></div>
  </div>
  <div id="tree-panel"></div>
</div>

<script>
const GEOMETRY = __GEOMETRY_JSON__;
const FRAMES   = __FRAMES_JSON__;
</script>

<script>__THREEJS__</script>
<script>__ORBITCONTROLS__</script>

<script>

// ---------------------------------------------------------------------------
// PLR → Three.js coordinate mapping
//   PLR x (rails, left-right)  → Three.js  x
//   PLR y (deck depth, front)  → Three.js -z
//   PLR z (height, up)         → Three.js  y
// Box at PLR(px,py,pz) size(sw,sh,sd):
//   Three.js centre  (px+sw/2,  pz+sd/2,  -(py+sh/2))
//   BoxGeometry      (sw, sd, sh)
// ---------------------------------------------------------------------------

function p2t(px, py, pz) { return new THREE.Vector3(px, pz, -py); }

function makeBox(px, py, pz, sw, sh, sd, color, opacity) {
  const mat = new THREE.MeshLambertMaterial({
    color,
    transparent: opacity != null && opacity < 1,
    opacity: opacity ?? 1,
  });
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(sw, sd, sh), mat);
  mesh.position.set(px + sw / 2, pz + sd / 2, -(py + sh / 2));
  return mesh;
}

// ---------------------------------------------------------------------------
// Scene globals
// ---------------------------------------------------------------------------
let scene, camera, renderer, controls;
let wellMeshes   = {};   // "plate:id"   → THREE.Mesh
let tipMeshes    = {};   // "rack:pos"   → THREE.Mesh
let channelMeshes = [];  // [8] channel tube meshes
let tipOnPipette  = [];  // [8] tip-slot meshes on channels
let armMesh, channelGroup;

const ARM_Y      = 350;  // Three.js y (= PLR z mm) of pipette arm crossbar
const CHAN_H     = 90;   // channel tube height mm
const TIP_H      = 38;   // tip box height mm

const BOX_COLOR = {
  HamiltonSTARDeck: 0xe8e8e8,
  TipCarrier:       0x7c5b73,
  PlateCarrier:     0x5f7290,
  ResourceHolder:   0xcfd6e0,
  PlateHolder:      0xcfd6e0,
  TipRack:          0x23232b,
  Plate:            0x1c1c22,
};
const DEFAULT_BOX_COLOR = 0xb8a23a;
const BOX_OPACITY = {
  HamiltonSTARDeck: 0.5,
};

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------
function colorForWell(entry) {
  const c = new THREE.Color();
  if (!entry || entry.volume <= 0) { c.set(0xffffff); return c; }
  if (entry.compounds && entry.compounds.length) {
    const comp = entry.compounds[0];
    let hash = 0;
    for (const ch of comp.name) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
    const pct = Math.min(comp.conc / 200, 1);
    c.setHSL(hash / 360, 0.72, 0.8 - pct * 0.5);
    return c;
  }
  if (entry.cells) { c.set(0xa8d4ff); return c; }
  c.set(0x888888);  // buffer/volume-only
  return c;
}

// ---------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------
function buildDeck() {
  for (const box of GEOMETRY.boxes) {
    const color   = BOX_COLOR[box.kind] ?? DEFAULT_BOX_COLOR;
    const opacity = BOX_OPACITY[box.kind];
    const mesh = makeBox(box.x, box.y, box.z, box.w, box.h, box.d, color, opacity);
    mesh.userData.name = box.name;
    scene.add(mesh);
  }
}

function buildWells() {
  const WELL_LIFT = 1.5;  // mm above plate surface
  for (const w of GEOMETRY.wells) {
    const geo  = new THREE.BoxGeometry(w.w - 0.8, WELL_LIFT * 2, w.h - 0.8);
    const mat  = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(w.x + w.w / 2, w.z + w.d + WELL_LIFT, -(w.y + w.h / 2));
    scene.add(mesh);
    wellMeshes[w.plate + ':' + w.id] = mesh;
  }
}

function buildTipSpots() {
  for (const t of GEOMETRY.tip_spots) {
    const r    = Math.min(t.w, t.h) / 2.8;
    const geo  = new THREE.CylinderGeometry(r, r, 6, 8);
    const mat  = new THREE.MeshLambertMaterial({ color: 0x1abc9c });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(t.x + t.w / 2, t.z + t.d + 3, -(t.y + t.h / 2));
    scene.add(mesh);
    tipMeshes[t.rack + ':' + t.id] = mesh;
  }
}

function buildPipette() {
  const b = GEOMETRY.bounds;
  const deckW = b.max_x - b.min_x;

  // Crossbar spanning full deck width
  const armGeo = new THREE.BoxGeometry(deckW, 12, 28);
  const armMat = new THREE.MeshLambertMaterial({ color: 0x888888, transparent: true, opacity: 0.85 });
  armMesh = new THREE.Mesh(armGeo, armMat);
  armMesh.position.set(b.min_x + deckW / 2, ARM_Y, 0);
  scene.add(armMesh);

  // 8 channel tubes + tip slots hang below the crossbar
  channelGroup = new THREE.Group();
  channelGroup.position.set(GEOMETRY.home_pos.x, ARM_Y, -GEOMETRY.home_pos.y);
  scene.add(channelGroup);

  const chanMat = new THREE.MeshLambertMaterial({ color: 0xaaaaaa });
  const tipMat  = new THREE.MeshLambertMaterial({ color: 0x1abc9c, transparent: true, opacity: 0.9 });

  for (let i = 0; i < 8; i++) {
    const chan = new THREE.Mesh(new THREE.BoxGeometry(7, CHAN_H, 7), chanMat.clone());
    chan.position.set(0, -CHAN_H / 2, 0);
    channelMeshes.push(chan);
    channelGroup.add(chan);

    const tip = new THREE.Mesh(new THREE.BoxGeometry(5, TIP_H, 5), tipMat.clone());
    tip.position.set(0, -CHAN_H - TIP_H / 2, 0);
    tip.visible = false;
    tipOnPipette.push(tip);
    channelGroup.add(tip);
  }
}

// ---------------------------------------------------------------------------
// Per-frame state updates
// ---------------------------------------------------------------------------
function updateWells(frame) {
  for (const [key, mesh] of Object.entries(wellMeshes)) {
    const [plate, id] = key.split(':');
    const entry = frame.snapshot[plate]?.[id];
    mesh.material.color.copy(colorForWell(entry));
  }
}

function updateTipSpots(frame) {
  for (const [key, mesh] of Object.entries(tipMeshes)) {
    const [rack, pos] = key.split(':');
    mesh.visible = !!(frame.snapshot.tip_racks?.[rack]?.[pos]);
  }
}

// Position the 8 channels at the correct row-Y offsets for the given plate.
// Returns the PLR centre-Y of that plate (used to position the channelGroup).
function alignChannels(plateName) {
  const rowYs = GEOMETRY.plate_row_ys[plateName];
  if (!rowYs) return null;
  const rows = 'ABCDEFGH'.split('');
  const ys = rows.map(r => rowYs[r]).filter(v => v != null);
  if (!ys.length) return null;
  const centerY = ys.reduce((a, b) => a + b, 0) / ys.length;
  rows.forEach((row, i) => {
    const localZ = rowYs[row] != null ? -(rowYs[row] - centerY) : -(i - 3.5) * 9;
    channelMeshes[i].position.z = localZ;
    tipOnPipette[i].position.z  = localZ;
  });
  return centerY;
}

const TRANSFER_TOOLS = new Set(['column_transfer', 'multi_dispense', 'serial_transfer',
                                  'mix_column', 'propose_prime']);

function updatePipetteTips(frame, showTips) {
  const show = showTips ?? TRANSFER_TOOLS.has(frame.tool);
  tipOnPipette.forEach(m => { m.visible = show; });
}

function updateState(frame, showTips) {
  updateWells(frame);
  updateTipSpots(frame);
  updatePipetteTips(frame, showTips);
}

// ---------------------------------------------------------------------------
// Pipette animation
// ---------------------------------------------------------------------------
let animState = null;  // {fromX, fromZ, toX, toZ, startMs, durationMs, onDone, showTips}

function startAnim(toPos, onDone, showTips) {
  if (!toPos) { onDone?.(); return; }

  const plateName = toPos.plate;
  const centerY = plateName ? alignChannels(plateName) : null;
  const toX = toPos.x;
  const toZ = -(centerY ?? toPos.y);

  tipOnPipette.forEach(m => { m.visible = !!(showTips); });

  animState = {
    fromX: channelGroup.position.x, fromZ: channelGroup.position.z,
    toX, toZ,
    startMs: performance.now(), durationMs: 550,
    onDone, showTips,
  };
}

function tickAnim(now) {
  if (!animState) return;
  const raw = Math.min((now - animState.startMs) / animState.durationMs, 1);
  const t   = raw < 0.5 ? 2 * raw * raw : -1 + (4 - 2 * raw) * raw;  // ease-in-out
  const x   = animState.fromX + (animState.toX - animState.fromX) * t;
  const z   = animState.fromZ + (animState.toZ - animState.fromZ) * t;
  channelGroup.position.x = x;
  channelGroup.position.z = z;
  armMesh.position.z       = z;
  if (raw >= 1) {
    const cb = animState.onDone;
    animState = null;
    cb?.();
  }
}

// Snap pipette instantly to a position (no animation).
function snapPipette(pos) {
  if (!pos) return;
  const plateName = pos.plate;
  const centerY = plateName ? alignChannels(plateName) : null;
  const x = pos.x;
  const z = -(centerY ?? pos.y);
  channelGroup.position.x = x;
  channelGroup.position.z = z;
  armMesh.position.z       = z;
}

// ---------------------------------------------------------------------------
// Frame navigation
// ---------------------------------------------------------------------------
let currentFrameIdx = 0;
let isPlaying = false;
let playTimeout = null;

function goToFrame(idx, animate, onDone) {
  currentFrameIdx = idx;
  document.getElementById('slider').value = idx;
  const frame = FRAMES[idx];
  const showTips = TRANSFER_TOOLS.has(frame.tool);

  if (animate && frame.pipette_pos) {
    startAnim(frame.pipette_pos, () => {
      updateState(frame, false);  // tips now gone (step complete)
      updateUI(frame);
      onDone?.();
    }, showTips);
  } else {
    animState = null;
    snapPipette(frame.pipette_pos);
    updateState(frame, false);
    updateUI(frame);
    onDone?.();
  }
}

function stepPlay() {
  if (!isPlaying) return;
  const next = currentFrameIdx + 1;
  if (next >= FRAMES.length) {
    isPlaying = false;
    updatePlayBtn();
    return;
  }
  goToFrame(next, true, () => {
    if (isPlaying) playTimeout = setTimeout(stepPlay, 180);
  });
}

function togglePlay() {
  if (isPlaying) {
    isPlaying = false;
    if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }
  } else {
    if (currentFrameIdx >= FRAMES.length - 1) currentFrameIdx = -1;
    isPlaying = true;
    playTimeout = setTimeout(stepPlay, 0);
  }
  updatePlayBtn();
}

function stepBack() {
  isPlaying = false;
  animState  = null;
  if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }
  if (currentFrameIdx > 0) goToFrame(currentFrameIdx - 1, false);
  updatePlayBtn();
}

function stepForward() {
  isPlaying = false;
  animState  = null;
  if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }
  if (currentFrameIdx < FRAMES.length - 1) goToFrame(currentFrameIdx + 1, false);
  updatePlayBtn();
}

function updatePlayBtn() {
  document.getElementById('btn-play').textContent = isPlaying ? '⏸ Pause' : '▶ Play';
}

// ---------------------------------------------------------------------------
// Right-panel UI
// ---------------------------------------------------------------------------
const ROOT_KEY = '__root__';
const TREE_CHILDREN = {};
for (const box of GEOMETRY.boxes) {
  const key = box.parent == null ? ROOT_KEY : box.parent;
  (TREE_CHILDREN[key] = TREE_CHILDREN[key] || []).push(box);
}

function treeNodeHTML(box, frame, depth) {
  const children = (TREE_CHILDREN[box.name] || []).slice().sort((a, b) => b.y - a.y);
  const isEmpty = (box.kind === 'ResourceHolder' || box.kind === 'PlateHolder') && !children.length;

  let info = box.kind;
  if (box.kind === 'Plate') info = `${box.well_count} wells`;
  if (box.kind === 'TipRack') {
    const state    = frame.snapshot.tip_racks?.[box.name] || {};
    const occupied = Object.values(state).filter(Boolean).length;
    info = `${occupied}/${box.tip_capacity} tips`;
  }

  const label = isEmpty
    ? `<span class="tree-empty">&lt;empty&gt;</span>`
    : `<span class="tree-name">${box.name}</span> <span class="tree-info">${info}</span>`;
  let html = `<li data-name="${box.name}" style="padding-left:${depth * 14}px">${label}</li>`;
  for (const child of children) html += treeNodeHTML(child, frame, depth + 1);
  return html;
}

function renderTree(frame) {
  const roots = TREE_CHILDREN[ROOT_KEY] || [];
  const panel = document.getElementById('tree-panel');
  panel.innerHTML = `<ul>${roots.map(r => treeNodeHTML(r, frame, 0)).join('')}</ul>`;
  panel.querySelectorAll('li').forEach(li => {
    li.addEventListener('mouseenter', () => highlightOn(li.dataset.name));
    li.addEventListener('mouseleave', () => highlightOff(li.dataset.name));
  });
}

function highlightOn(name) {
  document.querySelectorAll(`[data-name="${name}"]`).forEach(el => el.classList.add('hl'));
}
function highlightOff(name) {
  document.querySelectorAll(`[data-name="${name}"]`).forEach(el => el.classList.remove('hl'));
}

function renderLegend(frame) {
  const names = new Set();
  for (const plate of ['source_plate', 'dest_plate'])
    for (const entry of Object.values(frame.snapshot[plate]))
      if (entry.compounds) for (const c of entry.compounds) names.add(c.name);

  const chips = [];
  for (const name of names) {
    let hash = 0;
    for (const ch of name) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
    chips.push(`<span><span class="swatch" style="background:hsl(${hash},72%,55%)"></span>${name}</span>`);
  }
  chips.push(`<span><span class="swatch" style="background:#3a6a9a"></span>cells</span>`);
  chips.push(`<span><span class="swatch" style="background:#444"></span>buffer</span>`);
  chips.push(`<span><span class="swatch" style="background:#1abc9c;border-radius:50%"></span>tip</span>`);
  document.getElementById('legend').innerHTML = chips.join('');
}

function updateUI(frame) {
  document.getElementById('step-label').innerHTML =
    `Step ${frame.step} / ${FRAMES.length - 2}: <b>${frame.tool}</b>` +
    (frame.error ? ` <span class="error">ERR: ${frame.error}</span>` : '');
  document.getElementById('args').textContent = frame.preview || '';
  renderTree(frame);
  renderLegend(frame);
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
function renderLoop(now) {
  requestAnimationFrame(renderLoop);
  tickAnim(now);
  controls.update();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function init() {
  const container = document.getElementById('canvas-container');

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f0f0);

  const b = GEOMETRY.bounds;
  const cx = (b.min_x + b.max_x) / 2;
  const cy = (b.min_y + b.max_y) / 2;

  camera = new THREE.PerspectiveCamera(42, container.clientWidth / container.clientHeight, 1, 5000);
  camera.position.set(cx, 820, -cy + 580);
  camera.lookAt(cx, 0, -cy);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(cx, 60, -cy);
  controls.update();

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const sun = new THREE.DirectionalLight(0xffffff, 0.9);
  sun.position.set(cx + 300, 700, -cy + 500);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xddeeff, 0.3);
  fill.position.set(cx - 400, 200, -cy - 300);
  scene.add(fill);

  buildDeck();
  buildWells();
  buildTipSpots();
  buildPipette();

  goToFrame(0, false);
  renderLoop(0);

  // Event listeners
  document.getElementById('btn-play').addEventListener('click', togglePlay);
  document.getElementById('btn-back').addEventListener('click', stepBack);
  document.getElementById('btn-fwd').addEventListener('click', stepForward);
  document.getElementById('slider').addEventListener('input', e => {
    isPlaying = false;
    animState  = null;
    if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }
    goToFrame(parseInt(e.target.value, 10), false);
    updatePlayBtn();
  });

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
}

init();
</script>
</body>
</html>
"""


def render_html(goal: str, geometry: dict, frames: list[dict], title: str,
                js: dict[str, str] | None = None) -> str:
    goal = " ".join(goal.split())
    three_js         = (js or {}).get("three.js", "")
    orbit_js         = (js or {}).get("OrbitControls.js", "")
    return (
        _HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__GOAL__", goal)
        .replace("__MAX_STEP__", str(len(frames) - 1))
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
    js    = _fetch_js()
    html  = render_html(data.get("goal", ""), geometry, frames, title, js)
    out_path = os.path.splitext(args.run_path)[0] + ".viz.html"
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Wrote {out_path}")
    if args.open:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
