"""AI agent loop for the Hamilton STAR digital twin.

The agent receives a natural-language goal, then uses Claude's tool-use API to
issue liquid-handling commands, observe deck state, and update the plate map
until the experiment is complete.

Also provides functions for running hardcoded scripts and agent loops with other models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import os
from typing import Awaitable, Callable
from dotenv import load_dotenv
load_dotenv()

import anthropic
from anthropic.types import Message

from star_sim.env import RobotEnv
from star_sim.plate_map import Compound, WellContents, mix_contents
from star_sim.channels import ROWS, describe, parse_selector
from star_sim.primitives import mix, transfer

# ---------------------------------------------------------------------------
# Tool schemas (passed to the Claude API on every request)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "observe",
        "description": (
            "Return the current deck state: volume (µL) in every non-empty well of "
            "every plate, which tip-rack rows still hold a tip in each column, the "
            "total tips remaining, and the plate map recording compound/cell "
            "contents noted so far."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "transfer",
        "description": (
            "Move liquid from source wells to destination wells.\n"
            "src/dst are well selectors: a single well ('B3'), a range ('A1:H1'), or "
            "an explicit list (['A1','C4']). A range covers the block between its "
            "corners, walked down each column then across — 'A1:H1' is all 8 rows of "
            "column 1; 'A1:H6' is columns 1-6.\n"
            "Equal-length selections pair up one-to-one. A smaller source repeats "
            "across the destinations, so one source column can fill several "
            "destination columns, and one source well can fill several wells in its "
            "own row.\n"
            "A transfer must stay within a row (A1->A5 is fine, A1->B5 is not), "
            "because each of the 8 channels is fixed to its own row."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "src_plate": {"type": "string", "enum": ["source_plate", "dest_plate"]},
                "src": {"type": "string", "description": "Source well selector, e.g. 'A1:H1' or 'B3'."},
                "dst_plate": {"type": "string", "enum": ["source_plate", "dest_plate"]},
                "dst": {"type": "string", "description": "Destination well selector."},
                "volume": {
                    "type": "number",
                    "description": "µL per destination well.",
                },
                "tips": {
                    "type": "string",
                    "enum": ["new", "reuse"],
                    "description": (
                        "'new' (default) takes fresh tips for every column of "
                        "destinations — always safe. 'reuse' keeps one set for the "
                        "whole call: use it for a dilution chain, or when "
                        "distributing one clean reagent into many wells. Reusing "
                        "tips across unrelated wells risks carryover and will "
                        "return a warning."
                    ),
                },
                "transfer_cells": {
                    "type": "boolean",
                    "description": "Set true only after mix has resuspended cells in the source wells.",
                },
            },
            "required": ["src_plate", "src", "dst_plate", "dst", "volume"],
        },
    },
    {
        "name": "mix",
        "description": (
            "Aspirate and re-dispense in place to resuspend settled cells or "
            "homogenise a well after a transfer. Moves no liquid between wells and "
            "leaves concentrations unchanged. Takes the same well selectors as transfer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plate": {"type": "string", "enum": ["source_plate", "dest_plate"]},
                "wells": {"type": "string", "description": "Well selector, e.g. 'A1:H1'."},
                "volume": {"type": "number", "description": "µL drawn up per cycle."},
                "repetitions": {"type": "integer", "minimum": 1, "description": "Default 3."},
                "tips": {"type": "string", "enum": ["new", "reuse"]},
            },
            "required": ["plate", "wells", "volume"],
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
                    "description": "One entry per group of wells to initialise, on any plate.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "plate": {
                                "type": "string",
                                "enum": ["source_plate", "dest_plate"],
                                "description": "Which plate to load (default: source_plate)",
                            },
                            "col": {
                                "type": "integer", "minimum": 1, "maximum": 12,
                                "description": "Fill all 8 wells of this column. Use `wells` instead for anything narrower.",
                            },
                            "wells": {
                                "type": "string",
                                "description": (
                                    "Well selector ('A1', 'A1:H1', 'A1:D3') naming exactly "
                                    "which wells to load. Give either this or `col`."
                                ),
                            },
                            "compound": {"type": "string", "description": "e.g. 'dye', 'buffer', 'DMSO'"},
                            "concentration_um": {"type": "number", "description": "µM; omit for buffer/media"},
                            "cells": {"type": "string", "description": "Cell line name, e.g. 'HeLa'"},
                            "cell_density_per_ml": {"type": "number", "description": "cells/mL"},
                            "volume_ul": {"type": "number", "description": "µL per well"},
                            "notes": {"type": "string"},
                        },
                        "required": ["volume_ul"],
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
  tip_rack      96 Hamilton 1000 µL filtered tips, 8 rows (A-H) x 12 columns.
                Tips are consumed individually: a one-well transfer costs one tip,
                a full-column transfer costs eight. Used tips are discarded and
                cannot be picked up again.

OPERATION MODEL
  You address wells directly, using selectors: a single well ('B3'), a range
  ('A1:H1' = all 8 rows of column 1; 'A1:H6' = columns 1-6), or a list
  (['A1','C4']). Ranges walk down each column, then across to the next.

  The head has 8 channels on a fixed pitch, which constrains what one movement
  can do:
    - Channel N is fixed to row N, so a transfer stays within its row.
      A1 -> A5 is fine; A1 -> B5 is not. Route between rows via an intermediate well.
    - One movement addresses one column. A transfer spanning several columns is
      carried out as one movement per column, in the order the selector lists them.

  Volumes are in microlitres (µL). Concentrations are in micromolar (µM).

RULES
  1. When starting, give a succinct yet comprehensive overview of everything you intend to accomplish. 
  2. Before each tool call, write a short description of what you are about to do
     and why.
  3. Call propose_prime first to declare all initial source plate contents. Use volumes that
     are highly tractable to a human operator.
  4. Compound concentrations are tracked automatically via mass balance through every
     transfer and presented by observe().
  5. Cells stay where they are seeded. To move cells, first call mix to resuspend
     them, then pass transfer_cells=true to the transfer call.
  6. Never aspirate more than a well contains; never exceed the well max volume.
     If a tool returns {"error": ...}, read the message and adjust before retrying or quitting.
  7. Plan tip consumption up front — 96 tips total. Prefer one wide transfer over
     many narrow ones: a single call with a multi-column destination reuses the
     channels efficiently, and tips='reuse' spends one set of tips for the whole
     call instead of one set per column. Two idioms worth knowing:
       - distribute one reagent:  src 'A1:H1', dst 'A2:H6', tips 'reuse'
       - serial dilution chain:   src 'A1:H5', dst 'A2:H6', tips 'reuse'
         (each step aspirates from the well it just filled, so there is no carryover)
     Use tips='new' whenever unrelated wells must not cross-contaminate.
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
    on_step: Callable[[str, dict, str | None, dict], Awaitable[None]] | None = None,
) -> None:
    """Execute a pre-written sequence of tool calls without any LLM.

    Useful for verifying that dispatch, robot primitives, error handling, and
    plate-map updates all work correctly before spending API credits.

    Set confirm=True to pause before each tool call (step-through mode).
    Raises RuntimeError immediately on any tool error so the problem is visible.

    on_step, if given, is awaited after every executed tool call with
    (name, args, message, result) — same contract as run_agent's on_step — so a
    live visualizer can update after each dispatch, not just once at the end.
    Scripted runs have no agent text, so message is always None.
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

        if on_step:
            await on_step(name, args, None, result)

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
    # 1. Pre-fill dest cols 2-6 with 100 µL buffer. The source column repeats
    #    across the five destination columns; one set of tips serves all of them
    #    because the diluent is clean.
    ("transfer", {
        "src_plate": "source_plate", "src": "A2:H2",
        "dst_plate": "dest_plate", "dst": "A2:H6",
        "volume": 100, "tips": "reuse",
    }),
    # 2. Seed dest col 1 with 200 µL dye at 200 µM
    ("transfer", {
        "src_plate": "source_plate", "src": "A1:H1",
        "dst_plate": "dest_plate", "dst": "A1:H1",
        "volume": 200,
    }),
    # 3. Serial 1→2→…→6: five column-to-column steps in one call, concentrations
    #    computed automatically. Safe to reuse tips — each step aspirates from the
    #    well the previous step just filled.
    ("transfer", {
        "src_plate": "dest_plate", "src": "A1:H5",
        "dst_plate": "dest_plate", "dst": "A2:H6",
        "volume": 100, "tips": "reuse",
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

    async def complete(self, on_delta: Callable[[str, str], None] | None = None) -> AgentResponse:
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=self._messages,
        ) as stream:
            async for text in stream.text_stream:
                if on_delta:
                    on_delta("text", text)
            response: Message = await stream.get_final_message()
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


def _model_may_segment(model: str) -> bool:
    """Whether this model is known to sometimes only emit text OR a tool call
    in a single completion, never both together — confirmed so far only for
    gpt-oss. Unverified for other OpenAI-compatible models (e.g. Kimi K2) —
    extend this only once actually observed to segment.
    """
    return "gpt-oss" in model


def _thinking_extra_body(model: str, fast_mode: bool) -> dict | None:
    """Best-effort request to skip/reduce reasoning on Kimi models that
    support it, for when the caller wants a faster (non-thinking) turn.
    See https://platform.kimi.ai/docs/guide/use-thinking-models — the knob
    differs per model generation, and kimi-k2.7-code has no such knob at
    all (thinking is mandatory there), so this intentionally returns None
    for anything it doesn't recognize.
    """
    if not fast_mode:
        return None
    if model == "kimi-k2.6":
        return {"thinking": {"type": "disabled", "keep": None}}
    if model == "kimi-k3":
        return {"reasoning_effort": "low"}
    return None


# Belt-and-suspenders alongside _thinking_extra_body: some models still
# narrate step-by-step reasoning as plain visible text even with their
# thinking/reasoning-content channel disabled via the API, especially on
# follow-up turns — that prose lands as ordinary content, which no
# API-level flag can suppress after the fact.
_FAST_MODE_SYSTEM_SUFFIX = (
    "\n\nFast mode is enabled: respond directly and concisely. Do not narrate "
    "your reasoning process in your reply — decide, then act."
)


class OpenAICompatClient:
    """Agent client for any OpenAI-compatible endpoint (Groq, Ollama, etc.)."""

    def __init__(
        self, goal: str, model: str, base_url: str, api_key: str, fast_mode: bool = False,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._openai_tools = _to_openai_tools(TOOLS)
        self._may_segment = _model_may_segment(model)
        self._extra_body = _thinking_extra_body(model, fast_mode)
        system_content = SYSTEM + (_FAST_MODE_SYSTEM_SUFFIX if fast_mode else "")
        self._messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": goal},
        ]

    async def _complete_once(self, on_delta: Callable[[str, str], None] | None = None) -> AgentResponse:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            tools=self._openai_tools,
            tool_choice="auto",
            stream=True,
            extra_body=self._extra_body,
        )
        content_parts: list[str] = []
        # index -> partial tool call, merged across chunks (id/name/arguments
        # each typically arrive split across multiple deltas)
        tool_calls: dict[int, dict] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                if on_delta:
                    on_delta("text", delta.content)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning and on_delta:
                on_delta("reasoning", reasoning)
            for tc in (delta.tool_calls or []):
                slot = tool_calls.setdefault(
                    tc.index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["function"]["arguments"] += tc.function.arguments

        text = "".join(content_parts).strip() or None
        ordered_tool_calls = [tool_calls[i] for i in sorted(tool_calls)]

        msg_dict: dict = {"role": "assistant"}
        if text is not None:
            msg_dict["content"] = text
        if ordered_tool_calls:
            msg_dict["tool_calls"] = ordered_tool_calls
        self._messages.append(msg_dict)

        tool_uses = []
        for tc in ordered_tool_calls:
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_uses.append((tc["id"], tc["function"]["name"], args))
        return AgentResponse(text=text, tool_uses=tool_uses)

    async def complete(self, on_delta: Callable[[str, str], None] | None = None) -> AgentResponse:
        response = await self._complete_once(on_delta)
        if self._may_segment and response.text and not response.tool_uses:
            # Cheap, rare-path fix for a real correctness bug: a text-only
            # turn would otherwise make AgentResponse.done misreport
            # "finished the whole experiment" when the model just hasn't
            # gotten to the tool call yet. One bounded nudge, only when this
            # actually happens.
            self._messages.append({
                "role": "user",
                "content": "Continue: call the tool for the step you just described.",
            })
            follow_up = await self._complete_once(on_delta)
            return AgentResponse(text=response.text, tool_uses=follow_up.tool_uses)
        return response

    def submit_tool_results(self, results: list[tuple[str, dict]]) -> None:
        for tid, r in results:
            self._messages.append({
                "role": "tool",
                "tool_call_id": tid,
                "content": json.dumps(r),
            })

    def inject_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        
        
_PROVIDER_DEFAULTS: dict[str, dict] = {
    "anthropic":  {"model": "claude-opus-4-8",     "base_url": None},
    "groq":       {"model": "openai/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1"},
    "ollama":     {"model": "llama3.1:8b",          "base_url": "http://localhost:11434/v1"},
    "moonshot":   {"model": "kimi-k2.6", "base_url": "https://api.moonshot.ai/v1"},
    "openrouter": {"model": "moonshotai/kimi-k2",   "base_url": "https://openrouter.ai/api/v1"},
}

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic":     "ANTHROPIC_API_KEY",
    "groq":          "GROQ_API_KEY",
    "ollama":        "",           # Ollama doesn't need a key
    "moonshot":      "MOONSHOT_API_KEY",
    "openrouter":    "OPENROUTER_API_KEY",
    "openai-compat": "OPENAI_API_KEY",
}


def build_client(
    provider: str,
    goal: str,
    model: str | None = None,
    base_url: str | None = None,
    fast_mode: bool = False,
) -> AnthropicClient | OpenAICompatClient:
    defaults = _PROVIDER_DEFAULTS.get(provider, {"model": None, "base_url": None})
    resolved_model = model or defaults["model"]
    resolved_url = base_url or defaults["base_url"]

    if provider == "anthropic":
        if not resolved_model:
            resolved_model = "claude-opus-4-8"
        return AnthropicClient(goal=goal, model=resolved_model)

    # All other providers use the OpenAI-compatible client
    env_key = _PROVIDER_ENV_KEYS.get(provider, "OPENAI_API_KEY")
    api_key = os.environ.get(env_key, "none") if env_key else "ollama"
    if not resolved_url:
        raise SystemExit(f"--base-url is required for provider '{provider}'")
    if not resolved_model:
        raise SystemExit(f"--model is required for provider '{provider}'")
    return OpenAICompatClient(
        goal=goal,
        model=resolved_model,
        base_url=resolved_url,
        api_key=api_key,
        fast_mode=fast_mode,
    )


def describe_llm_error(exc: Exception) -> str:
    """One clean line for a provider API error, instead of the raw dump both
    the openai and anthropic SDKs produce by default.

    Both SDKs build their exception's message/str() as literally
    `f"Error code: {status_code} - {body}"` — a Python dict repr glued onto a
    prefix, unpleasant to read and useless for a non-engineer. The actual
    human-readable text lives one level down at body["error"]["message"];
    both SDKs shape `.body` identically (duck-typed here so this works for
    either without importing both exception hierarchies).
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        err = body["error"]
        message = err.get("message")
        if message:
            status = getattr(exc, "status_code", None)
            prefix = f"HTTP {status}" if status else "API error"
            code = f" [{err['code']}]" if err.get("code") else ""
            return f"{prefix}{code}: {message}"
    return str(exc)


# ---------------------------------------------------------------------------
# Step-by-step session (used by the web server)
# ---------------------------------------------------------------------------

class AgentSession:
    """One experiment session driven one turn at a time.

    think() asks the LLM what to do next and stores the proposed tool uses.
    act()   executes those tool uses against the robot and submits results back.

    The caller controls when to advance — typically after showing the user
    the proposed actions and receiving confirmation:

        session = AgentSession(goal, env)
        response = await session.think()
        while not response.done:
            # show response.text and response.tool_uses to user, wait for ok
            await session.act(on_step=on_step)
            response = await session.think()
        # response.text is the final summary
    """

    def __init__(self, env: RobotEnv, client: AnthropicClient | OpenAICompatClient) -> None:
        self._env = env
        self._client = client
        self._pending_tool_uses: list[tuple[str, str, dict]] = []  # (tool_use_id, name, args)
        # self._last_text: str | None = None

    async def think(self, on_delta: Callable[[str, str], None] | None = None) -> AgentResponse:
        """Run one LLM completion. Returns the agent's text and proposed tool uses.

        on_delta, if given, is called synchronously as ("text" | "reasoning", chunk)
        pairs arrive from the provider's streamed response — lets a live UI show
        progress instead of a blank wait for the full completion.
        """
        print(f"\n[user]  Prompting model ({self._client._model})...")
        response = await self._client.complete(on_delta=on_delta)
        if response.text:
            print(f"\n[agent]  {response.text}")
        for _, name, args in response.tool_uses:
            args_str = json.dumps(args)
            if len(args_str) > 120:
                args_str = args_str[:117] + "..."
            print(f"\n[tool]  {name}({args_str})")
        # self._last_text = response.text
        self._pending_tool_uses = response.tool_uses
        return response

    async def act(
        self,
        step_delay: float = 0.0,
        on_step: Callable[[str, dict, str | None, dict], Awaitable[None]] | None = None,
    ) -> None:
        """Execute the tool uses proposed by the last think() and submit results.

        on_step, if given, is awaited after every executed tool call with
        (name, args, message, result) — same contract documented on the class.
        """
        results: list[tuple[str, dict]] = []
        for tid, name, args in self._pending_tool_uses:
            result = await _dispatch(name, args, self._env, step_delay)
            summary = json.dumps(result)
            if len(summary) > 300:
                summary = summary[:297] + "..."
            print(f"  → {summary}\n")
            results.append((tid, result))
            if on_step:
                await on_step(name, args, None, result)
        self._client.submit_tool_results(results)
        self._pending_tool_uses = []

    def redirect(self, text: str) -> None:
        """Decline the tool uses proposed by the last think() and steer the agent
        elsewhere instead of running them.

        Both provider APIs require every outstanding tool_use to get a matching
        tool_result before a new plain user turn is valid, so each pending call
        is marked skipped (never dispatched) rather than actually executed.
        """
        if self._pending_tool_uses:
            skipped = [
                (tid, {"skipped": True, "reason": "redirected by user"})
                for tid, _, _ in self._pending_tool_uses
            ]
            self._client.submit_tool_results(skipped)
            self._pending_tool_uses = []
        self._client.inject_user_message(text)
        

# ---------------------------------------------------------------------------
# Interactive confirmation
# ---------------------------------------------------------------------------

class UserInjection(Exception):
    """Raised during agent confirm when the user wants to redirect the LLM."""
    def __init__(self, text: str) -> None:
        self.text = text


def preview_tool(name: str, args: dict) -> str:
    """Return a human-readable one-or-few-line description of a tool call."""
    match name:
        case "propose_prime":
            lines = []
            for r in args.get("reagents", []):
                plate = r.get("plate", "source_plate")
                try:
                    where = describe(_prime_wells(r))
                except Exception:
                    where = "?"
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
                lines.append(f"{plate} {where}: {', '.join(parts)}  ({vol} µL/well)")
            return "\n".join(lines)

        case "transfer":
            tc = "  [transfer_cells]" if args.get("transfer_cells") else ""
            tips = args.get("tips", "new")
            reuse = "  [reusing tips]" if tips == "reuse" else ""
            try:
                srcs, dsts = parse_selector(args["src"]), parse_selector(args["dst"])
                where = f"{args['src_plate']} {describe(srcs)} \u2192 {args['dst_plate']} {describe(dsts)}"
            except Exception:
                where = f"{args['src_plate']} {args['src']} \u2192 {args['dst_plate']} {args['dst']}"
            return f"{where}  ({args['volume']} \u00b5L){reuse}{tc}"

        case "mix":
            reps = args.get("repetitions", 3)
            try:
                where = describe(parse_selector(args["wells"]))
            except Exception:
                where = str(args.get("wells"))
            return f"{args['plate']} {where}  ({args['volume']} \u00b5L \u00d7 {reps} reps)"

        case "observe":
            return "read current deck state"

        case _:
            raw = json.dumps(args)
            if len(raw) > 200:
                raw = raw[:197] + "..."
            return raw


def _confirm_scripted(name: str, args: dict) -> tuple[str, dict] | None:
    """Step-through prompt for scripted replay: run, skip, replace, or quit.

    No message injection — there is no LLM to redirect in scripted mode.
    Returns (name, args) to execute (possibly replaced), None to skip.
    Raises KeyboardInterrupt on quit.
    """
    print(f"\n[next]  {name}: {preview_tool(name, args)}")
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
    print(f"\n[next]  {name}: {preview_tool(name, args)}")
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

def _well_vol(env: RobotEnv, plate_name: str, well: str) -> float:
    """Current volume in one well, per PyLabRobot's tracker."""
    return getattr(env.layout, plate_name).get_item(well).tracker.get_used_volume()


def _clear_emptied(env: RobotEnv, plate_name: str, wells: list[str]) -> None:
    """Wipe plate_map entries for wells PyLabRobot now reports as empty."""
    for well in wells:
        if _well_vol(env, plate_name, well) <= 0:
            env.plate_map.set_well(plate_name, well, WellContents())


def _track_contents(env: RobotEnv, transfer_cells: bool) -> Callable[[str, dict], None]:
    """Build a movement listener that keeps the plate map in step with the liquid.

    Movements now carry their own wells and volumes, so one listener covers every
    transfer shape — including a dilution chain, where each step aspirates from a
    well an earlier step filled. The aspirate handler snapshots what the tips took
    up, keyed by channel; the dispense handler mixes that into the destination.
    Reading the source at aspirate time is what makes chains correct: no shadow
    copy of the plate is needed, because the plate map is already up to date by
    the time the next step draws from it.

    Volumes come from the tracker *after* the movement, so the pre-dispense volume
    is recovered by subtracting what was just added.
    """
    held: dict[int, tuple[WellContents | None, float]] = {}

    def on_movement(kind: str, info: dict) -> None:
        if info.get("mixing"):
            return  # in-place: same liquid back in the same well, nothing changes
        wells = info.get("wells") or []
        volumes = info.get("volumes") or []
        channels = info.get("channels") or []
        plate = info.get("plate")
        if kind == "aspirate":
            for well, vol, chan in zip(wells, volumes, channels):
                held[chan] = (env.plate_map.get_well(plate, well), vol)
            _clear_emptied(env, plate, list(wells))
        elif kind == "dispense":
            for well, vol, chan in zip(wells, volumes, channels):
                src_contents, _ = held.get(chan, (None, 0.0))
                dst_pre = max(_well_vol(env, plate, well) - vol, 0.0)
                env.plate_map.set_well(plate, well, mix_contents(
                    src_contents, vol,
                    env.plate_map.get_well(plate, well), dst_pre,
                    transfer_cells=transfer_cells,
                ))

    return on_movement


def _prime_wells(entry: dict) -> list[str]:
    """Which wells one propose_prime entry refers to — `wells` selector, or all of `col`."""
    if entry.get("wells"):
        return parse_selector(entry["wells"])
    if entry.get("col"):
        return [f"{row}{entry['col']}" for row in ROWS]
    raise ValueError(
        "Each reagent entry needs either 'wells' (e.g. 'A1:H1') or 'col'."
    )


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
                            where = describe(_prime_wells(r))
                            print(f"  {where:>20}: {', '.join(parts) or 'media'}  ({r['volume_ul']} µL/well)")
                    input("\n  Prepare the above plates, then press Enter to continue...")
                elif not silent:
                    print("\n  Simulation continuing (priming set by default)")
                # Seed the tracker unconditionally — declared volumes are the planning ground
                # truth on both sim and hardware. On hardware the operator just confirmed they
                # loaded exactly these amounts, so observe() will reflect that immediately.
                for plate_name, plate_reagents in by_plate.items():
                    plate_obj = getattr(env.layout, plate_name)
                    for r in plate_reagents:
                        wells = _prime_wells(r)
                        # set volumes
                        for well in wells:
                            plate_obj.get_item(well).tracker.set_volume(r["volume_ul"])
                        # set compounds — identity is recorded whenever a name was given,
                        # independent of concentration: DMSO/media are routinely primed with
                        # no concentration_um at all (the tool schema says as much), and that
                        # used to get silently treated as "no compound" and discarded, making
                        # vehicle-control and media-only wells indistinguishable from a truly
                        # undeclared buffer.
                        compounds = (
                            [Compound(name=r["compound"], concentration_um=r.get("concentration_um", 0.0))]
                            if "compound" in r else []
                        )
                        # set cells
                        for well in wells:
                            env.plate_map.set_well(
                                plate_name, well,
                                WellContents(
                                    compounds=list(compounds),
                                    cells=r.get("cells"),
                                    cell_density_per_ml=r.get("cell_density_per_ml"),
                                    notes=r.get("notes", ""),
                                ),
                            )
                return {"ok": True, "loaded": len(reagents), "hardware": env._use_hardware}

            case "observe":
                return env.observe()

            case "transfer":
                src_plate, dst_plate = args["src_plate"], args["dst_plate"]
                tc = args.get("transfer_cells", False)
                listener = _track_contents(env, tc)
                env.add_movement_listener(listener)
                try:
                    return await transfer(
                        env, src_plate, args["src"], dst_plate, args["dst"],
                        args["volume"],
                        tips=args.get("tips", "new"),
                        transfer_cells=tc,
                        step_delay=step_delay,
                    )
                finally:
                    env.remove_movement_listener(listener)

            case "mix":
                # In-place, so concentrations and cell density are unchanged and
                # no content listener is needed.
                return await mix(
                    env, args["plate"], args["wells"], args["volume"],
                    repetitions=args.get("repetitions", 3),
                    tips=args.get("tips", "new"),
                    step_delay=step_delay,
                )

            case _:
                return {"error": f"unknown tool '{name}'"}

    except Exception as exc:
        return {"error": str(exc)}

