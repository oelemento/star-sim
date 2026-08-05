// ── PLR → Three.js coordinate mapping ─────────────────────────────────
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

// ── Scene globals ──────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let wellMeshes = {};
let tipMeshes = {};
let channelMeshes = [];
let tipOnPipette = [];
let armMesh, channelGroup;

const ARM_Y = 350;
const CHAN_H = 90;
const TIP_H = 38;

const BOX_COLOR = {
  HamiltonSTARDeck: 0xe8e8e8,
  TipCarrier: 0x7c5b73,
  PlateCarrier: 0x5f7290,
  ResourceHolder: 0xcfd6e0,
  PlateHolder: 0xcfd6e0,
  TipRack: 0x23232b,
  Plate: 0x1c1c22,
};
const DEFAULT_BOX_COLOR = 0xb8a23a;
const BOX_OPACITY = {};
const DEFAULT_BOX_OPACITY = 1;
const HOVER_DIM_OPACITY = 0.4;

// ── Color helpers ──────────────────────────────────────────────────────
// A hash-derived hue (even a well-mixed one) only spreads any *given pair*
// of names apart on average — with just 2-6 compounds actually in play at
// once, it's common for two of them to land within a few degrees of each
// other by pure chance, which is exactly what looked wrong on screen.
// Instead, assign hues in first-seen order stepped by the golden angle
// (~137.5°) around the wheel: each new compound lands as far as possible
// from every hue already handed out, so concurrently-visible compounds stay
// visually distinct regardless of how many there are or what they're named.
const _GOLDEN_ANGLE = 137.50776;
const _compoundHues = new Map();
function hueForName(name) {
  if (!_compoundHues.has(name)) {
    _compoundHues.set(name, (_compoundHues.size * _GOLDEN_ANGLE) % 360);
  }
  return _compoundHues.get(name);
}

function colorForWell(entry) {
  const c = new THREE.Color();
  if (!entry || entry.volume <= 0) { c.set(0xffffff); return c; }
  if (entry.compounds && entry.compounds.length) {
    const hues = entry.compounds.map(comp => hueForName(comp.name));
    const hue = hues.reduce((a, b) => a + b, 0) / hues.length;
    const maxConc = Math.max(...entry.compounds.map(comp => comp.conc || 0));
    const pct = Math.min(maxConc / 200, 1);
    // A blended hue alone can't be told apart from some other single pure
    // compound that happens to land at the same average — desaturating a
    // mix instead marks it as visually distinct from any single-compound
    // well. Hover for the exact contents rather than reading it off color.
    const sat = hues.length > 1 ? 0.35 : 0.72;
    c.setHSL(hue / 360, sat, 0.8 - pct * 0.5);
    return c;
  }
  if (entry.cells) { c.set(0xa8d4ff); return c; }
  c.set(0x888888);
  return c;
}

// ── Scene building ─────────────────────────────────────────────────────
let boxMeshes = {};

function buildDeck() {
  for (const box of GEOMETRY.boxes) {
    const color = BOX_COLOR[box.kind] ?? DEFAULT_BOX_COLOR;
    const opacity = BOX_OPACITY[box.kind] ?? DEFAULT_BOX_OPACITY;
    const mesh = makeBox(box.x, box.y, box.z, box.w, box.h, box.d, color, opacity);
    mesh.userData.name = box.name;
    scene.add(mesh);
    boxMeshes[box.name] = { mesh, baseOpacity: opacity };
  }
}

function buildRails() {
  const b = GEOMETRY.bounds;
  const length = b.max_y - b.min_y;
  const deckBox = GEOMETRY.boxes.find(bx => bx.kind === 'HamiltonSTARDeck');
  const deckTop = deckBox ? deckBox.z + deckBox.d : 0;
  const railMat = new THREE.MeshLambertMaterial({ color: 0x999999 });
  for (const railX of GEOMETRY.rails) {
    const geo = new THREE.BoxGeometry(2, 1, length);
    const mesh = new THREE.Mesh(geo, railMat);
    mesh.position.set(railX, deckTop + 0.5, -(b.min_y + length / 2));
    scene.add(mesh);
  }
}

function buildWells() {
  const WELL_LIFT = 1.5;
  for (const w of GEOMETRY.wells) {
    const geo = new THREE.BoxGeometry(w.w - 0.8, WELL_LIFT * 2, w.h - 0.8);
    const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(w.x + w.w / 2, w.z + w.d + WELL_LIFT, -(w.y + w.h / 2));
    scene.add(mesh);
    wellMeshes[w.plate + ':' + w.id] = mesh;
  }
}

function buildTipSpots() {
  for (const t of GEOMETRY.tip_spots) {
    const r = Math.min(t.w, t.h) / 2.8;
    const geo = new THREE.CylinderGeometry(r, r, 6, 8);
    const mat = new THREE.MeshLambertMaterial({ color: 0x1abc9c });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(t.x + t.w / 2, t.z + t.d + 3, -(t.y + t.h / 2));
    scene.add(mesh);
    tipMeshes[t.rack + ':' + t.id] = mesh;
  }
}

function buildPipette() {
  const b = GEOMETRY.bounds;
  const deckW = b.max_x - b.min_x;
  const armGeo = new THREE.BoxGeometry(deckW, 12, 28);
  const armMat = new THREE.MeshLambertMaterial({ color: 0x888888, transparent: true, opacity: 0.85 });
  armMesh = new THREE.Mesh(armGeo, armMat);
  armMesh.position.set(b.min_x + deckW / 2, ARM_Y, -GEOMETRY.home_pos.y);
  scene.add(armMesh);

  channelGroup = new THREE.Group();
  channelGroup.position.set(GEOMETRY.home_pos.x, ARM_Y, -GEOMETRY.home_pos.y);
  scene.add(channelGroup);

  const chanMat = new THREE.MeshLambertMaterial({ color: 0xaaaaaa });
  const tipMat = new THREE.MeshLambertMaterial({ color: 0x1abc9c, transparent: true, opacity: 0.9 });

  for (let i = 0; i < 8; i++) {
    const z0 = (i - 3.5) * 9;
    const chan = new THREE.Mesh(new THREE.BoxGeometry(7, CHAN_H, 7), chanMat.clone());
    chan.position.set(0, -CHAN_H / 2, z0);
    channelMeshes.push(chan);
    channelGroup.add(chan);

    const tip = new THREE.Mesh(new THREE.BoxGeometry(5, TIP_H, 5), tipMat.clone());
    tip.position.set(0, -CHAN_H - TIP_H / 2, z0);
    tip.visible = false;
    tipOnPipette.push(tip);
    channelGroup.add(tip);
  }
}

// ── Hover tooltip ────────────────────────────────────────────────────────
// wellMeshes/tipMeshes/boxMeshes never change membership after init(), so
// the raycast target list is built once (see init()) rather than rebuilt on
// every mousemove.
let _hoverTargets = [];
const _raycaster = new THREE.Raycaster();
const _mouseNDC = new THREE.Vector2();

function tooltipText(kind, key) {
  if (kind === 'well') {
    const [plate, id] = key.split(':');
    const entry = currentSnapshot?.[plate]?.[id];
    if (!entry || entry.volume <= 0) return `${plate} ${id}\nempty`;
    const lines = [`${plate} ${id}`, `${entry.volume} µL`];
    for (const comp of entry.compounds || []) lines.push(`${comp.name} @ ${comp.conc} µM`);
    if (entry.cells) lines.push(`cells: ${entry.cells}`);
    return lines.join('\n');
  }
  if (kind === 'tip') {
    const [rack, pos] = key.split(':');
    const hasTip = !!currentSnapshot?.tip_racks?.[rack]?.[pos];
    return `${rack} ${pos}\n${hasTip ? 'tip loaded' : 'empty'}`;
  }
  return key; // box: just the resource name
}

function hideTooltip() {
  document.getElementById('hover-tooltip').style.display = 'none';
}

function onCanvasMouseMove(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  _mouseNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  _mouseNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  _raycaster.setFromCamera(_mouseNDC, camera);

  const hits = _raycaster.intersectObjects(_hoverTargets.map(t => t.mesh));
  if (hits.length === 0) { hideTooltip(); return; }

  const target = _hoverTargets.find(t => t.mesh === hits[0].object);
  if (!target) { hideTooltip(); return; }

  const tooltip = document.getElementById('hover-tooltip');
  tooltip.textContent = tooltipText(target.kind, target.key);
  tooltip.style.display = 'block';
  const margin = 14;
  const maxLeft = window.innerWidth - tooltip.offsetWidth - margin;
  const maxTop = window.innerHeight - tooltip.offsetHeight - margin;
  tooltip.style.left = `${Math.min(e.clientX + margin, Math.max(margin, maxLeft))}px`;
  tooltip.style.top = `${Math.min(e.clientY + margin, Math.max(margin, maxTop))}px`;
}

// ── Per-frame state updates ────────────────────────────────────────────
// Kept in sync with whatever's currently drawn so the hover tooltip can look
// up exact contents without needing its own separate "current frame" state.
let currentSnapshot = null;

function updateWells(snapshot) {
  currentSnapshot = snapshot;
  for (const [key, mesh] of Object.entries(wellMeshes)) {
    const [plate, id] = key.split(':');
    const entry = snapshot[plate]?.[id];
    mesh.material.color.copy(colorForWell(entry));
  }
}

function updateTipSpots(snapshot) {
  for (const [key, mesh] of Object.entries(tipMeshes)) {
    const [rack, pos] = key.split(':');
    mesh.visible = !!(snapshot.tip_racks?.[rack]?.[pos]);
  }
}

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
    tipOnPipette[i].position.z = localZ;
  });
  return centerY;
}

// ── Pipette animation ──────────────────────────────────────────────────
let animState = null;

function startAnim(toPos, onDone, showTips) {
  if (!toPos) { onDone?.(); return; }
  const centerY = toPos.plate ? alignChannels(toPos.plate) : null;
  const toX = toPos.x;
  const toZ = -(centerY ?? toPos.y);
  if (showTips != null) tipOnPipette.forEach(m => { m.visible = showTips; });
  animState = {
    fromX: channelGroup.position.x, fromZ: channelGroup.position.z,
    toX, toZ,
    startMs: performance.now(), durationMs: 550,
    onDone,
  };
}

function tickAnim(now) {
  if (!animState) return;
  const raw = Math.min((now - animState.startMs) / animState.durationMs, 1);
  const t = raw < 0.5 ? 2 * raw * raw : -1 + (4 - 2 * raw) * raw;
  const x = animState.fromX + (animState.toX - animState.fromX) * t;
  const z = animState.fromZ + (animState.toZ - animState.fromZ) * t;
  channelGroup.position.x = x;
  channelGroup.position.z = z;
  armMesh.position.z = z;
  if (raw >= 1) {
    const cb = animState.onDone;
    animState = null;
    cb?.();
  }
}

function snapPipette(pos) {
  if (!pos) return;
  const centerY = pos.plate ? alignChannels(pos.plate) : null;
  channelGroup.position.x = pos.x;
  channelGroup.position.z = -(centerY ?? pos.y);
  armMesh.position.z = -(centerY ?? pos.y);
}

// ── Frame navigation ───────────────────────────────────────────────────
// -1 means "nothing shown yet" — distinct from 0, which is a real first
// frame. FRAMES starts empty in live mode, so treating 0 as the initial
// value made advanceToLatest()'s "already caught up" check true the moment
// a single frame arrived, silently skipping the render of frame 0.
let currentFrameIdx = -1;
let isPlaying = false;
let playTimeout = null;

function playMovements(frame, onDone) {
  const moves = (frame.movements && frame.movements.length)
    ? frame.movements
    : [{ pipette_pos: frame.pipette_pos, show_tips: null, snapshot: frame.snapshot }];
  let mi = 0;
  function step() {
    if (mi >= moves.length) { onDone?.(); return; }
    const mv = moves[mi++];
    startAnim(mv.pipette_pos, () => {
      if (mv.show_tips != null) tipOnPipette.forEach(m => { m.visible = mv.show_tips; });
      updateWells(mv.snapshot);
      updateTipSpots(mv.snapshot);
      if (mi < moves.length) playTimeout = setTimeout(step, 120);
      else step();
    }, null);
  }
  step();
}

function goToFrame(idx, animate, onDone) {
  currentFrameIdx = idx;
  document.getElementById('slider').value = idx;
  const frame = FRAMES[idx];
  updateUI(frame);
  if (animate) {
    playMovements(frame, () => onDone?.());
  } else {
    animState = null;
    if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }
    snapPipette(frame.pipette_pos);
    updateWells(frame.snapshot);
    updateTipSpots(frame.snapshot);
    tipOnPipette.forEach(m => { m.visible = false; });
    onDone?.();
  }
}

const STEP_PAUSE_MS = 700;

function stepPlay() {
  if (!isPlaying) return;
  const next = currentFrameIdx + 1;
  if (next >= FRAMES.length) {
    isPlaying = false;
    updatePlayBtn();
    return;
  }
  goToFrame(next, true, () => {
    if (isPlaying) playTimeout = setTimeout(stepPlay, STEP_PAUSE_MS);
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

function playOneStep() {
  isPlaying = false;
  if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }
  updatePlayBtn();
  if (currentFrameIdx < FRAMES.length - 1) goToFrame(currentFrameIdx + 1, true);
}

function stepBack() {
  isPlaying = false; followLive = false; animState = null;
  if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }
  if (currentFrameIdx > 0) goToFrame(currentFrameIdx - 1, false);
  updatePlayBtn(); updateLiveBtn();
}

function stepForward() {
  isPlaying = false; followLive = false; animState = null;
  if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }
  if (currentFrameIdx < FRAMES.length - 1) goToFrame(currentFrameIdx + 1, false);
  updatePlayBtn(); updateLiveBtn();
}

// ── Live polling ───────────────────────────────────────────────────────
let followLive = true;
let liveAdvancing = false;

function updateLiveBtn() {
  const btn = document.getElementById('btn-live');
  return;
  // if (!LIVE) return;
  btn.style.background = followLive ? '#cdebd6' : '#f0f0f0';
  btn.textContent = followLive ? '● Live' : '○ Live';
}

function advanceToLatest() {
  if (liveAdvancing) return;
  liveAdvancing = true;
  const step = () => {
    if (!followLive || currentFrameIdx >= FRAMES.length - 1) { liveAdvancing = false; return; }
    goToFrame(currentFrameIdx + 1, true, step);
  };
  step();
}

function goLive() {
  followLive = true;
  updateLiveBtn();
  advanceToLatest();
}

function updatePlayBtn() {
  document.getElementById('btn-play').textContent = isPlaying ? '⏸ Pause' : '▶ Play';
}

// ── Right-panel tree ───────────────────────────────────────────────────
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
    const state = frame.snapshot.tip_racks?.[box.name] || {};
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

// A synthetic "nothing has run yet" snapshot — every tip present, no plate
// contents — so the tree/legend panels have something to show before the
// first real frame arrives (they normally only ever see frame.snapshot).
function initialSnapshot() {
  const tip_racks = {};
  for (const t of GEOMETRY.tip_spots) {
    (tip_racks[t.rack] = tip_racks[t.rack] || {})[t.id] = true;
  }
  return { tip_racks };
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

function namesUnder(name, acc) {
  acc = acc || new Set();
  acc.add(name);
  for (const child of (TREE_CHILDREN[name] || [])) namesUnder(child.name, acc);
  return acc;
}

function setMeshOpacity(mesh, opacity) {
  mesh.material.opacity = opacity;
  mesh.material.transparent = opacity < 1;
}

function highlightOn(name) {
  document.querySelectorAll(`[data-name="${name}"]`).forEach(el => el.classList.add('hl'));
  const keep = namesUnder(name);
  for (const [boxName, { mesh, baseOpacity }] of Object.entries(boxMeshes))
    setMeshOpacity(mesh, keep.has(boxName) ? baseOpacity : HOVER_DIM_OPACITY);
  for (const [key, mesh] of Object.entries(tipMeshes))
    setMeshOpacity(mesh, keep.has(key.split(':')[0]) ? 1 : HOVER_DIM_OPACITY);
}

function highlightOff(name) {
  document.querySelectorAll(`[data-name="${name}"]`).forEach(el => el.classList.remove('hl'));
  for (const { mesh, baseOpacity } of Object.values(boxMeshes)) setMeshOpacity(mesh, baseOpacity);
  for (const mesh of Object.values(tipMeshes)) setMeshOpacity(mesh, 1);
}

function renderLegend(frame) {
  const names = new Set();
  for (const plate of Object.keys(frame.snapshot).filter(k => k !== 'tip_racks'))
    for (const entry of Object.values(frame.snapshot[plate]))
      if (entry.compounds) for (const c of entry.compounds) names.add(c.name);

  const chips = [];
  for (const name of names) {
    const hue = hueForName(name);
    chips.push(`<span><span class="swatch" style="background:hsl(${hue},72%,55%)"></span>${name}</span>`);
  }
  chips.push(`<span><span class="swatch" style="background:#3a6a9a"></span>cells</span>`);
  chips.push(`<span><span class="swatch" style="background:#444"></span>buffer</span>`);
  chips.push(`<span><span class="swatch" style="background:#1abc9c;border-radius:50%"></span>tip</span>`);
  document.getElementById('legend').innerHTML = chips.join('');
}

function updateUI(frame) {
  if (!frame) return;
  // FRAMES holds only real steps (no leading/trailing bookend frames in this
  // app's live/streamed data model), so the highest valid step index is
  // simply the last position in the array.
  const total = FRAMES.length - 1;
  document.getElementById('step-label').innerHTML =
    frame.step >= 0
      ? `Step ${frame.step} / ${total}: <b>${frame.tool}</b>` +
      (frame.error ? ` <span style="color:#c00">ERR: ${frame.error}</span>` : '')
      : '';
  document.getElementById('args').textContent = frame.preview || '';
  const msgEl = document.getElementById('message');
  if (frame.message) {
    msgEl.textContent = frame.message;
    msgEl.style.display = '';
  } else {
    msgEl.style.display = 'none';
  }
  renderTree(frame);
  renderLegend(frame);
}

// ── Render loop ────────────────────────────────────────────────────────
function renderLoop(now) {
  requestAnimationFrame(renderLoop);
  tickAnim(now);
  controls.update();
  renderer.render(scene, camera);
}

// ── Init ───────────────────────────────────────────────────────────────
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

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const sun = new THREE.DirectionalLight(0xffffff, 0.9);
  sun.position.set(cx + 300, 700, -cy + 500);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xddeeff, 0.3);
  fill.position.set(cx - 400, 200, -cy - 300);
  scene.add(fill);

  buildDeck();
  buildRails();
  buildWells();
  buildTipSpots();
  buildPipette();

  if (FRAMES.length > 0) {
    goToFrame(START_STEP, false);
  } else {
    const frame = { snapshot: initialSnapshot() };
    updateWells(frame.snapshot);
    updateTipSpots(frame.snapshot);
    renderTree(frame);
    renderLegend(frame);
  }
  renderLoop(0);

  _hoverTargets = [
    ...Object.entries(wellMeshes).map(([key, mesh]) => ({ mesh, kind: 'well', key })),
    ...Object.entries(tipMeshes).map(([key, mesh]) => ({ mesh, kind: 'tip', key })),
    ...Object.entries(boxMeshes).map(([name, { mesh }]) => ({ mesh, kind: 'box', key: name })),
  ];
  renderer.domElement.addEventListener('mousemove', onCanvasMouseMove);
  renderer.domElement.addEventListener('mouseleave', hideTooltip);

  document.getElementById('btn-play').addEventListener('click', togglePlay);
  document.getElementById('btn-back').addEventListener('click', stepBack);
  document.getElementById('btn-step-anim').addEventListener('click', playOneStep);
  document.getElementById('btn-fwd').addEventListener('click', stepForward);
  document.getElementById('btn-live').addEventListener('click', goLive);
  document.getElementById('slider').addEventListener('input', e => {
    isPlaying = false; followLive = false; animState = null;
    if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }
    goToFrame(parseInt(e.target.value, 10), false);
    updatePlayBtn(); updateLiveBtn();
  });
  document.getElementById('btn-run').addEventListener('click', submitPrompt);
  document.getElementById('btn-quit').addEventListener('click', async () => {
    // The server takes a few seconds to actually exit (draining open SSE
    // connections) — flag that now instead of leaving the dot green until
    // the connection really drops.
    setConnBadge('connecting');
    document.getElementById('conn-badge').title = 'Shutting down…';
    await fetch('/quit', { method: 'POST' });
  });
  document.getElementById('prompt-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitPrompt(); }
  });
  document.getElementById('prompt-input').addEventListener('input', () => {
    if (phase === 'idle-done') render();
  });

  applyModeBadge(INITIAL_USE_HARDWARE);
  document.getElementById('mode-badge').addEventListener('click', toggleHardware);

  document.getElementById('btn-save-run').addEventListener('click', saveRun);
  document.getElementById('btn-load-run').addEventListener('click', () => {
    document.getElementById('load-run-input').click();
  });
  document.getElementById('load-run-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) loadRunFile(file);
    e.target.value = '';
  });

  if (new URLSearchParams(location.search).has('mock')) {
    document.addEventListener('keydown', e => {
      if (e.key === 'r' && document.activeElement !== document.getElementById('prompt-input')) {
        replayMock();
      }
    });
    replayMock();
  }

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  connectSSE();
}

// ── SSE ────────────────────────────────────────────────────────────────
let _activeProposalEl = null;

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function appendMsg(type, text) {
  document.getElementById('empty-state')?.remove();
  const conv = document.getElementById('conversation');
  const div = document.createElement('div');
  div.className = `msg msg-${type}`;
  if (type === 'user') {
    div.innerHTML = `<div class="msg-label">You</div>${escHtml(text)}`;
  } else if (type === 'agent') {
    div.innerHTML = `<div class="msg-label">Agent</div>${escHtml(text)}`;
  } else if (type === 'error') {
    div.innerHTML = `<div class="msg-label error">Error</div>${escHtml(text)}`;
  } else if (type === 'notice') {
    div.innerHTML = `<div class="msg-label">System</div>${escHtml(text)}`;
  }
  conv.appendChild(div);
  conv.scrollTop = conv.scrollHeight;
}

function appendProposals(tools) {
  document.getElementById('empty-state')?.remove();
  if (_activeProposalEl) _activeProposalEl.classList.add('executed');

  const conv = document.getElementById('conversation');
  const div = document.createElement('div');
  div.className = 'msg msg-proposals';
  const items = tools.map(t =>
    `<div class="proposal-item"><div class="tool-name">${escHtml(t.name)}</div><div class="proposal-preview">${escHtml(t.preview)}</div></div>`
  ).join('');
  div.innerHTML =
    `<div class="msg-label">Proposed actions</div>` +
    `<div class="proposal-list">${items}</div>` +
    `<div class="proposal-actions">` +
    `<button class="btn-confirm" onclick="submitConfirm()">✓ Run these</button>` +
    `<button class="btn-stop-inline" onclick="submitStop()">✗ Stop</button>` +
    `</div>`;
  conv.appendChild(div);
  conv.scrollTop = conv.scrollHeight;
  _activeProposalEl = div;
}

function appendToolResult(name, ok, error) {
  const conv = document.getElementById('conversation');
  const div = document.createElement('div');
  div.className = 'msg msg-tool';
  div.innerHTML =
    `<div class="msg-label">Tool result</div>` +
    `<span class="tool-name">${escHtml(name)}</span> ` +
    `<span class="tool-result ${ok ? 'ok' : 'error'}">${ok ? '✓ ok' : `✗ ${escHtml(error || '')}`}</span>`;
  conv.appendChild(div);
  conv.scrollTop = conv.scrollHeight;
}

function disableActiveProposals() {
  if (!_activeProposalEl) return;
  _activeProposalEl.querySelectorAll('button').forEach(b => { b.disabled = true; });
  _activeProposalEl = null;
}

function setStatus(text) {
  let el = document.getElementById('status-line');
  if (!el) {
    el = document.createElement('div');
    el.id = 'status-line';
    document.getElementById('conversation').appendChild(el);
  }
  if (text) {
    el.innerHTML = `<span class="status-dot"></span>${escHtml(text)}`;
    el.style.display = 'block';
    const conv = document.getElementById('conversation');
    conv.appendChild(el); // keep it pinned as the last row, below the latest message
    conv.scrollTop = conv.scrollHeight;
  } else {
    el.style.display = 'none';
  }
}

function handleSSE(evt) {
  switch (evt.type) {
    case 'thinking':
      setStatus('Agent is thinking…');
      break;
    case 'agent_text':
      setStatus(null);
      appendMsg('agent', evt.text);
      break;
    case 'proposals':
      setStatus(null);
      appendProposals(evt.tools);
      phase = 'awaiting';
      render();
      break;
    case 'acting':
      disableActiveProposals();
      setStatus('Executing…');
      phase = 'busy';
      render();
      break;
    case 'tool_result':
      appendToolResult(evt.name, evt.ok, evt.error);
      break;
    case 'frame':
      FRAMES.push(evt.data);
      document.getElementById('slider').max = FRAMES.length - 1;
      if (followLive) advanceToLatest();
      break;
    case 'done':
      setStatus(null);
      appendMsg('agent', 'Experiment complete.');
      // The server deliberately keeps the session alive here (see _think())
      // so a follow-up can redirect it — distinct from plain 'idle', which
      // covers 'stopped'/'error' where the session really is gone.
      phase = 'idle-done';
      render();
      break;
    case 'stopped':
      setStatus(null);
      disableActiveProposals();
      appendMsg('notice', 'Session stopped — the agent has been disconnected.');
      setRunButton(false);
      break;
    case 'error':
      setStatus(null);
      appendMsg('error', evt.message);
      setRunButton(false);
      break;
  }
}

const _CONN_TITLES = {
  connected: 'Connected to server',
  connecting: 'Pending connection',
  disconnected: 'Disconnected from server',
};

function setConnBadge(state) {
  const el = document.getElementById('conn-badge');
  el.classList.toggle('connected', state === 'connected');
  el.classList.toggle('connecting', state === 'connecting');
  el.classList.toggle('disconnected', state === 'disconnected');
  el.title = _CONN_TITLES[state];
}

function connectSSE() {
  setConnBadge('connecting');
  const es = new EventSource('/events');
  es.onopen = () => setConnBadge('connected');
  es.onmessage = e => handleSSE(JSON.parse(e.data));
  es.onerror = () => {
    setConnBadge('disconnected');
    es.close();
    setTimeout(connectSSE, 2000);
  };
}

// ── Mock conversation replay (dev only) ──────────────────────────────────
// Drives the exact same handleSSE() path the real SSE stream uses, but from
// a hardcoded script — no server round-trip, no LLM call, same result every
// time. Use this to iterate on conversation-panel CSS/JS: load once with
// ?mock=1 (or run replayMock() in the console), then just press "r" to
// replay after every edit instead of rebooting the app / re-prompting.
const MOCK_EVENTS = [
  { type: 'thinking' },
  { type: 'agent_text', text: "I'll test Drug A and Drug B individually, then all pairwise combinations, with 4 replicates each. Starting by priming the deck." },
  {
    type: 'proposals', tools: [
      {
        name: 'propose_prime',
        preview: 'source_plate col 1: Drug A @ 10 µM  (200 µL/well)\nsource_plate col 2: Drug B @ 10 µM  (200 µL/well)\nsource_plate col 3: media/buffer  (200 µL/well)',
      },
      {
        name: 'column_transfer',
        preview: 'source_plate col 1 → assay_plate col 1  (50 µL)  [transfer_cells]',
      },
    ],
  },
  { type: 'acting' },
  { type: 'tool_result', name: 'propose_prime', ok: true },
  { type: 'tool_result', name: 'column_transfer', ok: false, error: 'tip rack A1 empty' },
  { type: 'thinking' },
  { type: 'agent_text', text: 'Tip rack A1 was empty — reloading tips before continuing.' },
  {
    type: 'proposals', tools: [
      { name: 'pick_up_tips', preview: 'tip_rack_2 col 1' },
    ],
  },
  { type: 'acting' },
  { type: 'tool_result', name: 'pick_up_tips', ok: true },
  { type: 'done' },
];

function replayMock(events = MOCK_EVENTS, delayMs = 2000) {
  document.getElementById('empty-state')?.remove();
  document.getElementById('conversation').innerHTML = '';
  setStatus(null);
  setRunButton(false);
  events.forEach((evt, i) => setTimeout(() => handleSSE(evt), 1000 + i * delayMs));
}

// Once a run has produced any state worth clearing, the primary button
// offers a reset instead of silently letting a new prompt run on top of a
// dirty deck. Cleared back to false by resetSimulation().
let sessionDirty = false;

// 'idle'      — nothing running (or the session really is gone, e.g. after
//               stopped/error); button is Run or Reset depending on sessionDirty.
// 'busy'      — agent is thinking or executing tools; no user input accepted.
// 'awaiting'  — tools were proposed but not yet confirmed; the prompt box is
//               open again so the user can redirect instead of just confirming.
// 'idle-done' — the experiment finished, but the server deliberately kept the
//               session alive (see the 'done' handler) so it can still be
//               redirected instead of only reset.
let phase = 'idle';

function render() {
  const btn = document.getElementById('btn-run');
  const input = document.getElementById('prompt-input');
  btn.classList.toggle('running', phase === 'busy');
  if (phase === 'busy') {
    btn.textContent = '■ Stop';
  } else if (phase === 'awaiting') {
    btn.textContent = '↻ Redirect';
  } else if (phase === 'idle-done') {
    // Relabels live as the user types (see the 'input' listener in init()):
    // typing something means "continue this experiment", leaving it blank
    // and clicking means "discard it and start clean".
    btn.textContent = input.value.trim() ? '↻ Continue' : '↺ Reset simulation';
  } else if (sessionDirty) {
    btn.textContent = '↺ Reset simulation';
  } else {
    btn.textContent = '▶ Run simulation';
  }
  input.disabled = phase === 'busy';
  if (phase === 'awaiting') {
    input.placeholder = "Don't like the proposed actions? Describe what to do instead, or confirm below.";
  } else if (phase === 'idle-done') {
    input.placeholder = 'Forgot a step? Describe what to do next, or leave blank and click Reset.';
  } else {
    input.placeholder = 'Describe the experiment in plain language, e.g. Test Drug A and Drug B individually, then all pairwise combinations, with 4 replicates for each condition, including controls.';
  }
  document.getElementById('mode-badge').disabled = phase === 'busy';
  document.getElementById('model-picker').disabled = phase === 'busy';
}

function setRunButton(running) {
  phase = running ? 'busy' : 'idle';
  render();
}

async function resetSimulation() {
  await fetch('/reset', { method: 'POST' });

  isPlaying = false;
  followLive = true;
  animState = null;
  if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }

  FRAMES = [];
  currentFrameIdx = -1;
  document.getElementById('slider').max = 0;
  document.getElementById('slider').value = 0;
  updatePlayBtn();
  updateLiveBtn();

  snapPipette(GEOMETRY.home_pos);
  const frame = { snapshot: initialSnapshot() };
  updateWells(frame.snapshot);
  updateTipSpots(frame.snapshot);
  tipOnPipette.forEach(m => { m.visible = false; });
  renderTree(frame);
  renderLegend(frame);
  _compoundHues.clear();

  document.getElementById('step-label').innerHTML = '';
  document.getElementById('args').textContent = '';
  document.getElementById('message').style.display = 'none';

  document.getElementById('conversation').innerHTML =
    '<div id="empty-state">Enter an experiment prompt below.<br><br>' +
    'The agent will explain each step here<br>as it designs and runs the experiment.</div>';
  setStatus(null);

  sessionDirty = false;
  setRunButton(false);
}

// ── Simulation / hardware mode ───────────────────────────────────────────
function applyModeBadge(isHardware) {
  const badge = document.getElementById('mode-badge');
  badge.classList.remove('mode-pending');
  badge.classList.toggle('mode-hardware', isHardware);
  badge.textContent = isHardware ? 'Hardware' : 'Simulation';
  badge.title = `Click to switch to ${isHardware ? 'Simulation' : 'Hardware'}`;
}

async function toggleHardware() {
  const badge = document.getElementById('mode-badge');
  const goingToHardware = badge.textContent !== 'Hardware';

  // Immediate feedback: don't leave the badge looking unchanged while we
  // wait on a real USB connection attempt, which can take a moment.
  badge.disabled = true;
  badge.classList.add('mode-pending');
  badge.textContent = goingToHardware ? 'Connecting…' : 'Switching…';

  let res, data;
  try {
    res = await fetch('/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ use_hardware: goingToHardware }),
    });
    data = await res.json();
  } catch (err) {
    appendMsg('error', `Failed to reach server: ${err}`);
    badge.disabled = false;
    applyModeBadge(!goingToHardware);
    return;
  }

  badge.disabled = false;
  if (!res.ok) {
    appendMsg('error', data.error || 'Failed to switch mode.');
    applyModeBadge(!goingToHardware);
    return;
  }

  applyModeBadge(data.use_hardware);
}

// ── Save / load a run ─────────────────────────────────────────────────────
function saveRun() {
  if (FRAMES.length === 0) return;
  const payload = { title: document.title, savedAt: new Date().toISOString(), frames: FRAMES };
  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `run-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function loadRunFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    let data;
    try {
      data = JSON.parse(reader.result);
    } catch {
      appendMsg('error', 'Could not parse run file — not valid JSON.');
      return;
    }
    if (!Array.isArray(data.frames) || data.frames.length === 0) {
      appendMsg('error', 'Run file has no frames to play back.');
      return;
    }

    isPlaying = false;
    followLive = false;
    animState = null;
    if (playTimeout) { clearTimeout(playTimeout); playTimeout = null; }

    FRAMES = data.frames;
    document.getElementById('slider').max = FRAMES.length - 1;
    goToFrame(0, false);
    updatePlayBtn();
    updateLiveBtn();
    sessionDirty = true;
    setRunButton(false);
  };
  reader.readAsText(file);
}

async function submitPrompt() {
  const input = document.getElementById('prompt-input');

  if (phase === 'busy') { await submitStop(); return; }
  if (phase === 'awaiting') { await submitRedirect(); return; }
  if (phase === 'idle-done') {
    if (input.value.trim()) await submitRedirect();
    else await resetSimulation();
    return;
  }
  if (sessionDirty) { await resetSimulation(); return; }

  const goal = input.value.trim();
  if (!goal) return;

  appendMsg('user', goal);
  input.value = '';
  setRunButton(true);
  sessionDirty = true;

  FRAMES = [];
  currentFrameIdx = -1;
  document.getElementById('slider').max = 0;
  document.getElementById('slider').value = 0;
  followLive = true;
  updateLiveBtn();
  _compoundHues.clear();

  const provider = document.getElementById('model-picker').value;
  const res = await fetch('/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ goal, provider }),
  });
  if (!res.ok) {
    const data = await res.json();
    appendMsg('error', data.error || 'Failed to start run.');
    sessionDirty = false; // nothing actually started — plain "Run" is still correct
    setRunButton(false);
  }
}

async function submitConfirm() {
  disableActiveProposals();
  phase = 'busy';
  render();
  // Scrubbing back through history (stepBack/stepForward/slider) drops out of
  // live mode so it isn't yanked out from under the user mid-scrub — but
  // running the next action means there's new live progress to show, so jump
  // back in: catch up (animated) through anything already loaded, then keep
  // following as new frames arrive.
  followLive = true;
  updateLiveBtn();
  advanceToLatest();
  const res = await fetch('/confirm', { method: 'POST' });
  if (!res.ok) {
    const data = await res.json();
    appendMsg('error', data.error || 'Failed to confirm.');
  }
}

async function submitRedirect() {
  const input = document.getElementById('prompt-input');
  const text = input.value.trim();
  if (!text) return;

  appendMsg('user', text);
  input.value = '';
  disableActiveProposals();
  phase = 'busy';
  render();

  const res = await fetch('/redirect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text }),
  });
  if (!res.ok) {
    const data = await res.json();
    appendMsg('error', data.error || 'Failed to redirect.');
  }
}

async function submitStop() {
  try {
    await fetch('/stop', { method: 'POST' });
  } catch (err) {
    appendMsg('error', `Failed to reach server: ${err}`);
    return;
  }
  // The 'stopped' SSE broadcast normally drives the UI update (see
  // handleSSE) — this is a safety net in case the SSE connection happens to
  // be down right when /stop completes, which would otherwise leave the
  // button stuck on "Stop"/"Redirect" even though the server-side session
  // really did end.
  setTimeout(() => {
    if (phase !== 'idle') handleSSE({ type: 'stopped' });
  }, 1500);
}

init();
