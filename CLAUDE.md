# star-sim — Hamilton STAR digital twin

## Goal

Let biology researchers describe an experiment in natural language and get a
liquid-handling protocol that is validated in simulation before it ever touches
hardware. PyLabRobot is the platform-agnostic layer, so the same protocol runs
against the simulator or a physical Hamilton STAR.

The eventual deliverable for the wetlab collaborator is a file their Hamilton PC
can run without installing anything new and without writing Venus script. See
[Hamilton Venus file formats](#hamilton-venus-file-formats) for why that file is
a worklist rather than a generated method.

## What exists today

A FastAPI app with an SSE event stream, a three.js deck visualizer, and an LLM
tool-calling agent driving a simulated STARlet. Run it with `python app.py`
(opens a browser on port 8000).

```
app.py            FastAPI + SSE. Routes: / /run /reset /mode /confirm
                  /redirect /stop /quit /events. Human-in-the-loop: the agent
                  proposes tool calls, the user confirms or redirects, then it
                  acts and thinks again.
agent.py          Tool schemas, system prompt, dispatch, and two client
                  wrappers (Anthropic, and OpenAI-compatible for Groq / Ollama /
                  Moonshot / OpenRouter). The UI's model picker defaults to Groq.
                  Also `run_scripted()` — replays a fixed tool-call list with no
                  LLM, which is how the regression tests drive the sim.
render.py         Walks the PyLabRobot deck into `geometry()`, records a frame
                  per step via `LiveCapture`, and inlines everything into
                  static/app.html.
star_sim/
  deck.py         build_starlet_deck() -> deck + named resource handles
  lab.py          make_liquid_handler(use_hardware) -> ChatterBox sim or STAR
  env.py          RobotEnv: observe(), reset(), movement-listener hooks
  channels.py     well selectors + schedule() — the head-geometry rules
  primitives.py   transfer() and mix()
  tip_manager.py  per-tip allocation from the rack
  plate_map.py    compound/cell identity and concentration per well
tests/            pytest; `pytest` at the repo root runs them
_samples/         real Hamilton method exports (see below) — not used by any code
```

### The operation model

Everything is addressed **per well**. A selector is a single well (`B3`), a range
(`A1:H1`), or a list (`["A1","C4"]`); ranges walk the block **column-major**,
down each column then across.

Two constraints come from the physical head and are enforced in
`channels.schedule()`:

- Channels sit on a fixed 9 mm pitch, so **channel *i* only reaches row *i***.
  A transfer stays within its row — `A1 -> A5` is fine, `A1 -> B5` is rejected.
- **One movement addresses one column.** A transfer spanning several columns is
  carried out as one movement per column, in selector order.

`schedule()` takes any well list and emits the sequence of movements the machine
would actually make, so callers never think about channels. Its invariant: within
one `Move`, all sources share a column, all destinations share a column, and no
channel repeats.

Three operations that used to be separate tools now fall out of `transfer`:

| Intent | Call |
| --- | --- |
| column to column | `src="A1:H1"`, `dst="A3:H3"` |
| one reagent into many columns | `src="A1:H1"`, `dst="A2:H6"`, `tips="reuse"` |
| serial dilution chain | `src="A1:H5"`, `dst="A2:H6"`, `tips="reuse"` |

Tip reuse is what makes the last two cheap. It is safe in a dilution chain
because each step aspirates from the well the previous step just filled; when
reuse draws from a well the held tips did not fill, `transfer` returns a
carryover warning rather than failing silently.

**Note on selector semantics:** this deliberately differs from PyLabRobot's own
`plate["A1:D2"]`, which walks the same block *row-major*. Column-major matches
`ItemizedResource._ordering`, matches how an 8-channel head fills a plate, and is
what keeps a multi-column transfer from interleaving columns. PyLabRobot's
string-range form is marked for deprecation in its own source, so there is
nothing to stay compatible with.

## Hamilton Venus file formats

A Venus method is not one file. It is a family sharing a base name. `_samples/`
holds a real export (`Drug plating.hsl` / `.med`, plus its worklist
`Drugs for ben 1.xlsx`).

- **`.hsl`** — plain ASCII, control flow only. `main()` opens a file, reads
  records via `file1.AddField(...)`, and loops `EasyAspirate` / `EasyDispense`
  over the rows. No protocol-specific logic lives here.
- **`.stp`** — binary, undocumented. A property store keyed by the step GUIDs the
  `.hsl` references. The real per-step parameterization lives here: tip type
  (e.g. `MlStar50ulTipWithFilter`), channel pattern, liquid class, error
  recovery. Not exposed to the worklist at all.
- **`.med`** — binary, ~450 KB, a serialized workflow document. Best guess:
  needed for editing in the Venus GUI, not for headless execution —
  **unverified**.
- **`.sub`** — plain text boilerplate, an `OnAbort` stub. Auto-included by the
  `.hsl`. Ignorable.

**The `.hsl` is a generic template, not a protocol.** Diffing `Drug plating.hsl`
against a second method (`CherryPick_96 tube rack to 384_DMSO`, which lives
outside this repo) gives 32 differing lines: auto-generated step GUIDs, a
timestamp, and exactly one meaningful line — `arrayindex = arrayindex + 8` vs
`+ 1`, i.e. quadrant stepping for 96->384 versus 1:1 plating. Everything else
that distinguishes the two protocols lives in the external worklist.

**So the app should never generate `.hsl`/`.stp`/`.med`.** Synthesising `.stp`
entries would mean reverse-engineering an undocumented binary that can change
between Hamilton versions — high effort, fragile, low payoff. The worklist-only
path the collaborator asked for is not a downgrade; it matches Hamilton's own
architecture.

### Confirmed worklist schema

Columns in order, as read by `file1.AddField()` at `Drug plating.hsl:63-67`:

| Column | Type | Notes |
| --- | --- | --- |
| Source Labware | string | e.g. `SourcePlate1` |
| Source Position | string | well/tube coordinate, e.g. `A1` |
| Transfer Volumes | float (`hslFloat`) | µL in the sample data |
| Target Labware | string | e.g. `TargetPlate1` |
| Target Position | string | well coordinate |

Opened as `strFileName + " " + "Sheet1$"` — Excel-ODBC table addressing, not a
raw `.csv`. `Drugs for ben 1.xlsx` has 276 rows including the header, so 275 data
rows.

That file carries two extra columns beyond the schema: `F = Drug Name` and
`G = Target Position` — a second column with the same header as `E`. Columns A-E
are exactly the schema in order, and `G` is constant `A1` while `E` varies per
row, so both look like human annotations. The Excel ODBC driver de-duplicates
repeated column names (`Target Position`, `Target Position1`), so `AddField`
almost certainly binds to `E`. Worth confirming, but not the landmine it first
appeared to be.

`Drug Plate 1 (1).xlsx` (not in `_samples/`) is a *design* artifact, not an
instruction file — a lookup of drug-combination codes to wells (`1. ABCD -> A1`).
It gets compiled into the worklist; Venus never reads it.

## Next milestone: worklist export

Emit a run as a worklist matching the schema above. The per-well revamp was the
prerequisite: a scheduled `Move` already *is* a worklist row group (source well,
target well, volume), so the projection is 1:1 rather than a lossy ×8 fan-out.

`LiveCapture.record_step()` already persists the raw tool `args` on each frame,
so a saved run is self-describing and can be exported without replaying the
simulation.

Two things will not be expressible and should surface as warnings, not be
dropped: `propose_prime` (a declaration of starting contents, not a robot action
— emit an operator checklist) and tip strategy (fixed per GUID in `.stp`, not
driven by the worklist).

**Open question, unresolved:** whether the template will read a `.csv` at all.
The ODBC Text driver treats the filename as the table, so the `" Sheet1$"` suffix
would not resolve. Keep the row-producing function pure so swapping in an
`.xlsx` writer is a serializer change plus `openpyxl`.

## Benchmark direction

The agent's real output is a **tool-call sequence**, so that is the unit to
score — not `(template_id, worklist rows)`. A worklist is a downstream
serialization of the tool calls, and a lossy one; scoring it would conflate
protocol errors with serialization gaps.

Offline and fully checkable: prompt -> expected tool calls, no robot needed.
`run_scripted()` already replays a fixed tool-call list with no LLM, which is the
harness to build on. Score per argument (plate, wells, volume) rather than
pass/fail, and include adversarial cases: ambiguous prompts, volumes near labware
limits, insufficient tips, missing source wells, and cross-row transfers that
should be rejected.

## Open questions for the wetlab collaborator

- Is `.med` required at execution time, or only for editing in the Venus GUI?
  (Test: run a method with only `.hsl` + `.stp` + `.sub` present.)
- What does the file reader do with duplicate column headers?
- Exact format expected — `.xlsx` via ODBC `Sheet1$`, or will a raw `.csv` work?
- Do other generic templates already exist (different liquid classes, mixing,
  multi-channel patterns)?
