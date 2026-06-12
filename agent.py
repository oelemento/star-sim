"""AI agent loop for the Hamilton STAR digital twin.

The agent receives a natural-language goal, then uses Claude's tool-use API to
issue liquid-handling commands, observe deck state, and update the plate map
until the experiment is complete.

Also provides functions for running hardcoded scripts and agent loops with
local models through Ollama.
"""

from __future__ import annotations

import json
from dotenv import load_dotenv
load_dotenv()

import anthropic
from anthropic.types import Message

from star_sim.env import RobotEnv
from star_sim.plate_map import Compound, WellContents, mix_contents
from protocols.primitives import column_transfer, mix_column, multi_dispense, serial_transfer

ANTHROPIC_MODEL = "claude-opus-4-8"

# ---------------------------------------------------------------------------
# Tool schemas (passed to the Claude API on every request)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "observe",
        "description": (
            "Return the current deck state: volume (µL) in every well of every plate, "
            "tip availability, remaining full tip columns, and the plate map recording "
            "compound/cell contents noted so far."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "column_transfer",
        "description": (
            "Aspirate `volume` µL from all 8 wells of src_col on src_plate and dispense "
            "into dst_col on dst_plate. Consumes one fresh tip column."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "src_plate": {"type": "string", "enum": ["source_plate", "dest_plate"]},
                "src_col": {"type": "integer", "minimum": 1, "maximum": 12, "description": "1-indexed column"},
                "dst_plate": {"type": "string", "enum": ["source_plate", "dest_plate"]},
                "dst_col": {"type": "integer", "minimum": 1, "maximum": 12},
                "volume": {"type": "number", "minimum": 1, "maximum": 1000, "description": "µL per channel"},
                "transfer_cells": {
                    "type": "boolean",
                    "description": "Set true only after mix_column has resuspended cells in the source wells.",
                },
            },
            "required": ["src_plate", "src_col", "dst_plate", "dst_col", "volume"],
        },
    },
    {
        "name": "multi_dispense",
        "description": (
            "Aspirate `volume` µL from src_col on src_plate and dispense into each column "
            "in dst_cols on dst_plate, re-aspirating before each destination. "
            "Reuses one tip set throughout (safe: tips only ever touch one source). "
            "Prefer this over multiple column_transfer calls when distributing one source "
            "to several destinations — it uses one tip column instead of N."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "src_plate": {"type": "string", "enum": ["source_plate", "dest_plate"]},
                "src_col": {"type": "integer", "minimum": 1, "maximum": 12},
                "dst_plate": {"type": "string", "enum": ["source_plate", "dest_plate"]},
                "dst_cols": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 12},
                    "description": "Columns to fill, e.g. [2,3,4,5,6]",
                },
                "volume": {"type": "number", "minimum": 1, "maximum": 1000, "description": "µL per channel per destination"},
                "transfer_cells": {"type": "boolean"},
            },
            "required": ["src_plate", "src_col", "dst_plate", "dst_cols", "volume"],
        },
    },
    {
        "name": "mix_column",
        "description": "Mix a column in place by repeatedly aspirating and re-dispensing into the same wells.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plate": {"type": "string", "enum": ["source_plate", "dest_plate"]},
                "col": {"type": "integer", "minimum": 1, "maximum": 12},
                "volume": {"type": "number", "minimum": 1, "maximum": 1000},
                "repetitions": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
            "required": ["plate", "col", "volume"],
        },
    },
    {
        "name": "serial_transfer",
        "description": (
            "Transfer `volume` µL from column k to column k+1 for k in [start_col, end_col). "
            "end_col receives the final dispense. One tip set is reused for the whole sequence "
            "(safe: tips always move in the direction of increasing dilution)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plate": {"type": "string", "enum": ["source_plate", "dest_plate"]},
                "start_col": {"type": "integer", "minimum": 1, "maximum": 12},
                "end_col": {"type": "integer", "minimum": 2, "maximum": 12},
                "volume": {"type": "number", "minimum": 1, "maximum": 1000},
                "transfer_cells": {"type": "boolean"},
            },
            "required": ["plate", "start_col", "end_col", "volume"],
        },
    },
    {
        "name": "propose_prime",
        "description": (
            "Declare all initial plate contents before the experiment starts. "
            "Call this as your very first action. "
            "Use it for source_plate reagents AND dest_plate cell seeding. "
            "On simulation it seeds the volume tracker; on hardware it prints a "
            "checklist and waits for the operator to prepare the plates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reagents": {
                    "type": "array",
                    "description": "One entry per column to initialise, on any plate.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "plate": {
                                "type": "string",
                                "enum": ["source_plate", "dest_plate"],
                                "description": "Which plate to load (default: source_plate)",
                            },
                            "col": {"type": "integer", "minimum": 1, "maximum": 12},
                            "compound": {"type": "string", "description": "e.g. 'dye', 'buffer', 'DMSO'"},
                            "concentration_um": {"type": "number", "description": "µM; omit for buffer/media"},
                            "cells": {"type": "string", "description": "Cell line name, e.g. 'HeLa'"},
                            "cell_density_per_ml": {"type": "number", "description": "cells/mL"},
                            "volume_ul": {"type": "number", "description": "µL per well"},
                            "notes": {"type": "string"},
                        },
                        "required": ["col", "volume_ul"],
                    },
                },
            },
            "required": ["reagents"],
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM = """\
Always respond in English.

You are controlling a Hamilton STARlet liquid-handling robot through a digital twin.

DECK LAYOUT
  source_plate  96-well deep-well plate, 12 cols x 8 rows (A-H). 2000 µL max per well.
                Holds reagent stocks (compounds, buffer, media).
  dest_plate    96-well deep-well plate, 12 cols x 8 rows (A-H). 2000 µL max per well.
                Receives the experiment.
  tip_rack      96 Hamilton 1000 µL filtered tips in 12 columns (A-H x 1-12).
                The 8-channel head picks up one full column (8 tips) per operation.
                Each column is single-use — once dropped, it cannot be reused.

OPERATION MODEL
  Every liquid-handling tool acts on all 8 wells of a column simultaneously.
  Volumes are in microlitres (µL). Concentrations are in micromolar (µM).

RULES
  1. Call propose_prime first to declare all initial plate contents — source_plate reagents
     and dest_plate cell seeding. On hardware this pauses for operator plate preparation.
  2. Compound concentrations are tracked automatically via mass balance through every
     transfer and presented by observe(). Never compute or annotate them manually.
  3. Cells stay where they are seeded. To move cells, first call mix_column to resuspend
     them, then pass transfer_cells=true to the transfer call.
  4. Never aspirate more than a well contains; never exceed the well max volume.
     If a tool returns {"error": ...}, read the message and adjust before retrying or quitting.
  5. Plan tip consumption up front. The rack has 12 columns (96 tips total).
     Use multi_dispense (not repeated column_transfer) when one source feeds many
     destinations. Use serial_transfer (not repeated column_transfer) for dilution chains.
  6. When done, summarise the completed experiment and the resulting plate layout.
"""

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ToolCall = tuple[str, dict]  # (tool_name, args_dict)

# ---------------------------------------------------------------------------
# Scripted mode: replay a fixed sequence of tool calls, no LLM required
# ---------------------------------------------------------------------------

async def run_scripted(
    env: RobotEnv,
    script: list[ToolCall],
    step_delay: float = 0.0,
    confirm: bool = False,
) -> None:
    """Execute a pre-written sequence of tool calls without any LLM.

    Useful for verifying that dispatch, robot primitives, error handling, and
    plate-map updates all work correctly before spending API credits.

    Set confirm=True to pause before each tool call (step-through mode).
    Raises RuntimeError immediately on any tool error so the problem is visible.
    """
    for name, args in script:
        if confirm:
            decision = _confirm_scripted(name, args)
            if decision is None:
                print(f"  [skipped] {name}")
                continue
            name, args = decision

        args_str = json.dumps(args)
        if len(args_str) > 120:
            args_str = args_str[:117] + "..."
        print(f"\n[scripted] {name}({args_str})")
        result = await _dispatch(name, args, env, step_delay, silent=confirm)
        summary = json.dumps(result, indent=2)
        if len(summary) > 300:
            summary = summary[:297] + "..."
        print(f"  → {summary}")
        if "error" in result:
            raise RuntimeError(f"Tool '{name}' failed: {result['error']}")

# ---------------------------------------------------------------------------
# Canned priming configs  (passed as args to propose_prime)
# ---------------------------------------------------------------------------

SERIAL_DILUTION_PRIME: dict = {
    "reagents": [
        {"col": 1, "compound": "dye", "concentration_um": 200, "volume_ul": 300},
        {"col": 2, "volume_ul": 600},  # buffer/diluent: volume only, no compound identity
    ]
}

# Canned script that replicates the 6-column 2-fold serial dilution.
# Used as the default for `python run.py --scripted`.
#
# Expected final state (verify with env.observe() after the run):
#   dest_plate col 1: 100 µL, 200 µM dye   (gave 100 µL away, kept concentration)
#   dest_plate col 2: 100 µL, 100 µM dye
#   dest_plate col 3: 100 µL,  50 µM dye
#   dest_plate col 4: 100 µL,  25 µM dye
#   dest_plate col 5: 100 µL, 12.5 µM dye
#   dest_plate col 6: 200 µL, 6.25 µM dye  (only received, never gave)
SERIAL_DILUTION_SCRIPT: list[ToolCall] = [
    ("propose_prime", SERIAL_DILUTION_PRIME),
    # 1. Pre-fill dest cols 2-6 with 100 µL buffer (one tip column reused)
    ("multi_dispense", {
        "src_plate": "source_plate", "src_col": 2,
        "dst_plate": "dest_plate", "dst_cols": [2, 3, 4, 5, 6],
        "volume": 100,
    }),
    # 2. Seed dest col 1 with 200 µL dye at 200 µM
    ("column_transfer", {
        "src_plate": "source_plate", "src_col": 1,
        "dst_plate": "dest_plate", "dst_col": 1,
        "volume": 200,
    }),
    # 3. Serial 1→2→…→6: concentrations computed automatically
    ("serial_transfer", {
        "plate": "dest_plate", "start_col": 1, "end_col": 6, "volume": 100,
    }),
]


# ---------------------------------------------------------------------------
# Anthropic agent (production)
# ---------------------------------------------------------------------------

async def run_agent(
    env: RobotEnv,
    goal: str,
    step_delay: float = 0.0,
    confirm: bool = False,
) -> tuple[str, list[ToolCall]]:
    """Drive Claude to execute `goal` on the robot.

    Updates env.plate_map throughout. Returns (final_text, record) where record
    is a list of (name, args) tool calls that were actually executed.
    Requires ANTHROPIC_API_KEY in the environment.
    """
    client = anthropic.AsyncAnthropic()
    messages: list[dict] = [{"role": "user", "content": goal}]
    record: list[ToolCall] = []

    print("\n[user]  " + goal)

    while True:
        response: Message = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Display any agent messages
        for block in response.content:
            if hasattr(block, "text") and block.text.strip():
                print(f"\n[agent]  {block.text.strip()}")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final = next((b.text for b in response.content if hasattr(b, "text")), "NO FINAL TEXT GIVEN")
            return final, record

        # Execute tool calls, with potential user interaction
        tool_results = []
        injection: UserInjection | None = None
        for block in response.content:
            if block.type != "tool_use":
                continue
            name, args = block.name, dict(block.input)

            if confirm and injection is None:
                try:
                    decision = _confirm_agent(name, args)
                except UserInjection as exc:
                    injection = exc
                    decision = None
                if decision is None:
                    print(f"  [skipped] {name}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"skipped": True}),
                    })
                    continue
                name, args = decision
            result = await _dispatch(name, args, env, step_delay, silent=confirm)
            record.append((name, args))
            summary = json.dumps(result)
            if len(summary) > 300:
                summary = summary[:297] + "..."
            print(f"  → {summary}\n")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

        if injection is not None:
            messages.append({"role": "user", "content": injection.text})
        

# ---------------------------------------------------------------------------
# Interactive confirmation
# ---------------------------------------------------------------------------

class UserInjection(Exception):
    """Raised during agent confirm when the user wants to redirect the LLM."""
    def __init__(self, text: str) -> None:
        self.text = text


def _preview(name: str, args: dict) -> str:
    """Return a human-readable one-or-few-line description of a tool call."""
    match name:
        case "propose_prime":
            lines = ["propose_prime:"]
            for r in args.get("reagents", []):
                plate = r.get("plate", "source_plate")
                col = r.get("col", "?")
                vol = r.get("volume_ul", "?")
                parts: list[str] = []
                if "compound" in r:
                    conc = f" @ {r['concentration_um']} µM" if "concentration_um" in r else ""
                    parts.append(f"{r['compound']}{conc}")
                if "cells" in r:
                    density = f" ({r['cell_density_per_ml']:.2g} cells/mL)" if "cell_density_per_ml" in r else ""
                    parts.append(f"{r['cells']}{density}")
                if not parts:
                    parts.append("media/buffer")
                lines.append(f"  {plate} col {col}: {', '.join(parts)}  ({vol} µL/well)")
            return "\n".join(lines)

        case "column_transfer":
            tc = "  [transfer_cells]" if args.get("transfer_cells") else ""
            return (
                f"column_transfer: {args['src_plate']} col {args['src_col']} → "
                f"{args['dst_plate']} col {args['dst_col']}  ({args['volume']} µL){tc}"
            )

        case "multi_dispense":
            cols = args.get("dst_cols", [])
            tc = "  [transfer_cells]" if args.get("transfer_cells") else ""
            return (
                f"multi_dispense: {args['src_plate']} col {args['src_col']} → "
                f"{args['dst_plate']} cols {cols}  ({args['volume']} µL each){tc}"
            )

        case "serial_transfer":
            cols = " → ".join(str(c) for c in range(args["start_col"], args["end_col"] + 1))
            tc = "  [transfer_cells]" if args.get("transfer_cells") else ""
            return (
                f"serial_transfer: {args['plate']} cols {cols}  ({args['volume']} µL steps){tc}"
            )

        case "mix_column":
            reps = args.get("repetitions", 3)
            return f"mix_column: {args['plate']} col {args['col']}  ({args['volume']} µL × {reps} reps)"

        case "observe":
            return "observe: read current deck state"

        case _:
            raw = json.dumps(args)
            if len(raw) > 200:
                raw = raw[:197] + "..."
            return f"{name}({raw})"


def _confirm_scripted(name: str, args: dict) -> tuple[str, dict] | None:
    """Step-through prompt for scripted replay: run, skip, replace, or quit.

    No message injection — there is no LLM to redirect in scripted mode.
    Returns (name, args) to execute (possibly replaced), None to skip.
    Raises KeyboardInterrupt on quit.
    """
    print(f"\n[next]  {_preview(name, args)}")
    try:
        ans = input("  Enter=run  s=skip  r=replace  q=quit: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt
    if ans == "q":
        raise KeyboardInterrupt("User aborted")
    if ans == "s":
        return None
    if ans == "r":
        new_name = input("  Tool name: ").strip()
        raw = input("  Args (JSON, or empty for {}): ").strip()
        new_args = json.loads(raw) if raw else {}
        return new_name, new_args
    return name, args


def _confirm_agent(name: str, args: dict) -> tuple[str, dict] | None:
    """Full interactive prompt for agent runs: run, skip, replace, message, or quit.

    Returns (name, args) to execute (possibly replaced), None to skip.
    Raises UserInjection when the user wants to redirect the LLM.
    Raises KeyboardInterrupt on quit.
    """
    print(f"\n[next]  {_preview(name, args)}")
    try:
        ans = input("  Enter=run  s=skip  r=replace  m=message  q=quit: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt
    if ans == "q":
        raise KeyboardInterrupt("User aborted")
    if ans == "s":
        return None
    if ans == "r":
        new_name = input("  Tool name: ").strip()
        raw = input("  Args (JSON, or empty for {}): ").strip()
        new_args = json.loads(raw) if raw else {}
        return new_name, new_args
    if ans == "m":
        raise UserInjection(input("  Message to agent: ").strip())
    return name, args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col_vols(env: RobotEnv, plate_name: str, col: int) -> dict[str, float]:
    """Return {row: volume_ul} for all 8 rows of a column, reading the PLR tracker."""
    plate = getattr(env.layout, plate_name)
    return {row: plate.get_item(f"{row}{col}").tracker.get_used_volume() for row in "ABCDEFGH"}


def _clear_emptied_col(env: RobotEnv, plate_name: str, col: int) -> None:
    """Wipe plate_map entries for wells in a column that PLR reports as empty after aspiration."""
    plate = getattr(env.layout, plate_name)
    for row in "ABCDEFGH":
        well_id = f"{row}{col}"
        if plate.get_item(well_id).tracker.get_used_volume() <= 0:
            env.plate_map.set_well(plate_name, well_id, WellContents())


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

async def _dispatch(name: str, args: dict, env: RobotEnv, step_delay: float, silent: bool = False) -> dict:
    try:
        match name:
            case "propose_prime":
                reagents = args["reagents"]
                by_plate: dict[str, list] = {}
                for r in reagents:
                    by_plate.setdefault(r.get("plate", "source_plate"), []).append(r)
                if env._use_hardware:
                    for plate_name, plate_reagents in by_plate.items():
                        print(f"\n[prime]  {plate_name}:")
                        for r in plate_reagents:
                            parts = []
                            if "compound" in r:
                                conc = f" @ {r['concentration_um']} µM" if "concentration_um" in r else ""
                                parts.append(f"{r['compound']}{conc}")
                            if "cells" in r:
                                density = f" @ {r['cell_density_per_ml']:.0f}/mL" if "cell_density_per_ml" in r else ""
                                parts.append(f"{r['cells']}{density}")
                            print(f"  col {r['col']:2d}: {', '.join(parts) or 'media'}  ({r['volume_ul']} µL/well)")
                    input("\n  Prepare the above plates, then press Enter to continue...")
                elif not silent:
                    print("\n  Simulation continuing (priming set by default)")
                # Seed the tracker unconditionally — declared volumes are the planning ground
                # truth on both sim and hardware. On hardware the operator just confirmed they
                # loaded exactly these amounts, so observe() will reflect that immediately.
                for plate_name, plate_reagents in by_plate.items():
                    plate_obj = getattr(env.layout, plate_name)
                    for r in plate_reagents:
                        # set volumes
                        for row in "ABCDEFGH":
                            plate_obj.get_item(f"{row}{r['col']}").tracker.set_volume(r["volume_ul"])
                        # set compounds
                        conc = r.get("concentration_um", 0.0)
                        compounds = (
                            [Compound(name=r["compound"], concentration_um=conc)]
                            if "compound" in r and conc > 0 else []
                        )
                        # set cells
                        env.plate_map.set_column(
                            plate_name, r["col"],
                            WellContents(
                                compounds=compounds,
                                cells=r.get("cells"),
                                cell_density_per_ml=r.get("cell_density_per_ml"),
                                notes=r.get("notes", ""),
                            ),
                        )
                return {"ok": True, "loaded": len(reagents), "hardware": env._use_hardware}

            case "observe":
                return env.observe()

            case "column_transfer":
                src_plate, src_col = args["src_plate"], args["src_col"]
                dst_plate, dst_col = args["dst_plate"], args["dst_col"]
                volume = args["volume"]
                tc = args.get("transfer_cells", False)
                dst_pre = _col_vols(env, dst_plate, dst_col)
                await column_transfer(env, src_plate, src_col, dst_plate, dst_col, volume, step_delay=step_delay)
                for row in "ABCDEFGH":
                    env.plate_map.set_well(dst_plate, f"{row}{dst_col}", mix_contents(
                        env.plate_map.get_well(src_plate, f"{row}{src_col}"), volume,
                        env.plate_map.get_well(dst_plate, f"{row}{dst_col}"), dst_pre[row],
                        transfer_cells=tc,
                    ))
                _clear_emptied_col(env, src_plate, src_col)
                return {"ok": True}

            case "multi_dispense":
                src_plate, src_col = args["src_plate"], args["src_col"]
                dst_plate, dst_cols = args["dst_plate"], args["dst_cols"]
                volume = args["volume"]
                tc = args.get("transfer_cells", False)
                dst_pre = {col: _col_vols(env, dst_plate, col) for col in dst_cols}
                await multi_dispense(env, src_plate, src_col, dst_plate, dst_cols, volume, step_delay=step_delay)
                for col in dst_cols:
                    for row in "ABCDEFGH":
                        env.plate_map.set_well(dst_plate, f"{row}{col}", mix_contents(
                            env.plate_map.get_well(src_plate, f"{row}{src_col}"), volume,
                            env.plate_map.get_well(dst_plate, f"{row}{col}"), dst_pre[col][row],
                            transfer_cells=tc,
                        ))
                _clear_emptied_col(env, src_plate, src_col)
                return {"ok": True}

            case "mix_column":
                await mix_column(
                    env, args["plate"], args["col"], args["volume"],
                    repetitions=args.get("repetitions", 3), step_delay=step_delay,
                )
                return {"ok": True}  # in-place: concentrations and cell density unchanged

            case "serial_transfer":
                plate_name = args["plate"]
                start_col, end_col = args["start_col"], args["end_col"]
                volume = args["volume"]
                tc = args.get("transfer_cells", False)
                pre_vols = {
                    (row, col): _col_vols(env, plate_name, col)[row]
                    for col in range(start_col, end_col + 1)
                    for row in "ABCDEFGH"
                }
                await serial_transfer(env, plate_name, start_col, end_col, volume, step_delay=step_delay)
                local: dict[tuple, WellContents | None] = {
                    k: env.plate_map.get_well(plate_name, f"{k[0]}{k[1]}") for k in pre_vols
                }
                local_vols = dict(pre_vols)
                for col in range(start_col, end_col):
                    for row in "ABCDEFGH":
                        src_k, dst_k = (row, col), (row, col + 1)
                        local[dst_k] = mix_contents(
                            local[src_k], volume, local[dst_k], local_vols[dst_k],
                            transfer_cells=tc,
                        )
                        local_vols[src_k] -= volume
                        local_vols[dst_k] += volume
                for (row, col), contents in local.items():
                    if contents is not None:
                        well_id = f"{row}{col}"
                        if local_vols[(row, col)] <= 0:
                            env.plate_map.set_well(plate_name, well_id, WellContents())
                        else:
                            env.plate_map.set_well(plate_name, well_id, contents)
                return {"ok": True}

            case _:
                return {"error": f"unknown tool '{name}'"}

    except Exception as exc:
        return {"error": str(exc)}

