"""AI agent loop for the Hamilton STAR digital twin.

The agent receives a natural-language goal, then uses Claude's tool-use API to
issue liquid-handling commands, observe deck state, and update the plate map
until the experiment is complete.

Also provides functions for running hardcoded scripts and agent loops with
local models through Ollama.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

import anthropic
from anthropic.types import Message

from star_sim.env import RobotEnv
from star_sim.plate_map import Compound, WellContents, mix_contents
from protocols.primitives import column_transfer, mix_column, multi_dispense, serial_transfer

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
  1. When starting, give a succinct yet comprehensive overview of everything you intend to accomplish. 
  2. Before each tool call, write a short description of what you are about to do
     and why.
  3. Call propose_prime first to declare all initial source plate contents. Use volumes that
     are highly tractable to a human operator. In hardware deployment, propose_prime will
     wait for operator confirmation.
  4. Compound concentrations are tracked automatically via mass balance through every
     transfer and presented by observe().
  5. Cells stay where they are seeded. To move cells, first call mix_column to resuspend
     them, then pass transfer_cells=true to the transfer call.
  6. Never aspirate more than a well contains; never exceed the well max volume.
     If a tool returns {"error": ...}, read the message and adjust before retrying or quitting.
  7. Plan tip consumption up front. The rack has 12 columns (96 tips total).
     Use multi_dispense (not repeated column_transfer) when one source feeds many
     destinations. Use serial_transfer (not repeated column_transfer) for dilution chains.
  8. When done, summarise the completed experiment and the resulting plate layout.
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
# Agent client abstraction
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    text: str | None
    # Each entry: (tool_use_id, name, args) — id only needed for result routing
    tool_uses: list[tuple[str, str, dict]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return not self.tool_uses


class AnthropicClient:
    """Agent client backed by the Anthropic API."""

    def __init__(self, goal: str, model: str = "claude-opus-4-8") -> None:
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._messages: list[dict] = [{"role": "user", "content": goal}]

    async def complete(self) -> AgentResponse:
        response: Message = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=self._messages,
        )
        self._messages.append({"role": "assistant", "content": response.content})
        text = next((b.text for b in response.content if hasattr(b, "text") and b.text.strip()), None)
        tool_uses = [
            (b.id, b.name, dict(b.input))
            for b in response.content if b.type == "tool_use"
        ]
        return AgentResponse(text=text, tool_uses=tool_uses)

    def submit_tool_results(self, results: list[tuple[str, dict]]) -> None:
        self._messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": json.dumps(r)}
            for tid, r in results
        ]})

    def inject_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        }}
        for t in tools
    ]


class OpenAICompatClient:
    """Agent client for any OpenAI-compatible endpoint (Groq, Ollama, etc.)."""

    def __init__(self, goal: str, model: str, base_url: str, api_key: str) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._openai_tools = _to_openai_tools(TOOLS)
        self._messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": goal},
        ]

    async def complete(self) -> AgentResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            tools=self._openai_tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        msg_dict = msg.model_dump(exclude_none=True)
        msg_dict.pop("annotations", None)  # newer openai SDK adds this; not accepted by all providers
        self._messages.append(msg_dict)
        text = msg.content.strip() if msg.content and msg.content.strip() else None
        tool_uses = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_uses.append((tc.id, tc.function.name, args))
        return AgentResponse(text=text, tool_uses=tool_uses)

    def submit_tool_results(self, results: list[tuple[str, dict]]) -> None:
        for tid, r in results:
            self._messages.append({
                "role": "tool",
                "tool_call_id": tid,
                "content": json.dumps(r),
            })

    def inject_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})


# ---------------------------------------------------------------------------
# Unified agent loop
# ---------------------------------------------------------------------------

async def run_agent(
    client: AnthropicClient | OpenAICompatClient,
    env: RobotEnv,
    step_delay: float = 0.0,
    confirm: bool = False,
) -> tuple[str, list[ToolCall], list[str | None]]:
    """Drive an agent client to execute a goal on the robot.

    Works with AnthropicClient or any OpenAICompatClient (Groq, Ollama, etc.).
    Returns (final_text, record, messages) where record is the list of executed
    tool calls and messages[i] is the agent's explanatory text (if any) from the
    turn that produced record[i] — the same text repeats across every tool call
    issued in one turn, since one explanation can cover a batch of actions.
    """
    record: list[ToolCall] = []
    messages: list[str | None] = []

    while True:
        print(f"\n[user]  Prompting model ({client._model})...")
        response = await client.complete()

        if response.text:
            print(f"\n[agent]  {response.text}")
        if not confirm:
            for _, name, args in response.tool_uses:
                args_str = json.dumps(args)
                if len(args_str) > 120:
                    args_str = args_str[:117] + "..."
                print(f"\n[tool]  {name}({args_str})")

        if response.done:
            return response.text or "", record, messages

        tool_results: list[tuple[str, dict]] = []
        injection: UserInjection | None = None

        for tid, name, args in response.tool_uses:
            if confirm and injection is None:
                try:
                    decision = _confirm_agent(name, args)
                except UserInjection as exc:
                    injection = exc
                    decision = None
                if decision is None:
                    print(f"  [skipped] {name}")
                    tool_results.append((tid, {"skipped": True}))
                    continue
                name, args = decision

            result = await _dispatch(name, args, env, step_delay, silent=confirm)
            record.append((name, args))
            messages.append(response.text)
            summary = json.dumps(result)
            if len(summary) > 300:
                summary = summary[:297] + "..."
            print(f"  → {summary}\n")
            tool_results.append((tid, result))

        client.submit_tool_results(tool_results)

        if injection is not None:
            client.inject_user_message(injection.text)
        

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

                def on_movement(kind: str, info: dict) -> None:
                    if kind != "dispense" or info.get("plate") != dst_plate or info.get("col") != dst_col:
                        return
                    for row in "ABCDEFGH":
                        env.plate_map.set_well(dst_plate, f"{row}{dst_col}", mix_contents(
                            env.plate_map.get_well(src_plate, f"{row}{src_col}"), volume,
                            env.plate_map.get_well(dst_plate, f"{row}{dst_col}"), dst_pre[row],
                            transfer_cells=tc,
                        ))

                env.add_movement_listener(on_movement)
                try:
                    await column_transfer(env, src_plate, src_col, dst_plate, dst_col, volume, step_delay=step_delay)
                finally:
                    env.remove_movement_listener(on_movement)
                _clear_emptied_col(env, src_plate, src_col)
                return {"ok": True}

            case "multi_dispense":
                src_plate, src_col = args["src_plate"], args["src_col"]
                dst_plate, dst_cols = args["dst_plate"], args["dst_cols"]
                volume = args["volume"]
                tc = args.get("transfer_cells", False)
                dst_pre = {col: _col_vols(env, dst_plate, col) for col in dst_cols}

                def on_movement(kind: str, info: dict) -> None:
                    col = info.get("col")
                    if kind != "dispense" or info.get("plate") != dst_plate or col not in dst_pre:
                        return
                    for row in "ABCDEFGH":
                        env.plate_map.set_well(dst_plate, f"{row}{col}", mix_contents(
                            env.plate_map.get_well(src_plate, f"{row}{src_col}"), volume,
                            env.plate_map.get_well(dst_plate, f"{row}{col}"), dst_pre[col][row],
                            transfer_cells=tc,
                        ))

                env.add_movement_listener(on_movement)
                try:
                    await multi_dispense(env, src_plate, src_col, dst_plate, dst_cols, volume, step_delay=step_delay)
                finally:
                    env.remove_movement_listener(on_movement)
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

                local_vols = {
                    (row, col): _col_vols(env, plate_name, col)[row]
                    for col in range(start_col, end_col + 1)
                    for row in "ABCDEFGH"
                }
                local: dict[tuple, WellContents | None] = {
                    k: env.plate_map.get_well(plate_name, f"{k[0]}{k[1]}") for k in local_vols
                }

                def on_movement(kind: str, info: dict) -> None:
                    if kind != "dispense" or info.get("plate") != plate_name:
                        return
                    dst_col = info.get("col")
                    src_col = dst_col - 1
                    if not (start_col <= src_col < end_col):
                        return
                    for row in "ABCDEFGH":
                        src_k, dst_k = (row, src_col), (row, dst_col)
                        local[dst_k] = mix_contents(
                            local[src_k], volume, local[dst_k], local_vols[dst_k],
                            transfer_cells=tc,
                        )
                        local_vols[src_k] -= volume
                        local_vols[dst_k] += volume
                        well_id = f"{row}{dst_col}"
                        if local_vols[dst_k] <= 0:
                            env.plate_map.set_well(plate_name, well_id, WellContents())
                        else:
                            env.plate_map.set_well(plate_name, well_id, local[dst_k])

                env.add_movement_listener(on_movement)
                try:
                    await serial_transfer(env, plate_name, start_col, end_col, volume, step_delay=step_delay)
                finally:
                    env.remove_movement_listener(on_movement)
                return {"ok": True}

            case _:
                return {"error": f"unknown tool '{name}'"}

    except Exception as exc:
        return {"error": str(exc)}

