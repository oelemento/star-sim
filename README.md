# star-sim — Hamilton STAR digital twin

A simulated Hamilton STARlet built on [PyLabRobot](https://docs.pylabrobot.org).
The same protocol can run against a **simulated** robot (ChatterBox backend +
browser visualizer) or the **physical** Hamilton STAR.
Built as a safe testbed for AI-directed liquid-handling experiments.

## Why

- Develop and validate protocols with **no robot and no risk** — overfills, missing tips, and empty-well aspirations surface in software, not on hardware.
  - For example, final volumes and concentrations are automatically calculated (not guess-timated by an LLM), enabling the validation of procedural results.
- Natural language interface for developing a sequence of experimental operations.
- Deploys directly to simulation backend or physical Hamilton at the press of a button.

## Install

```bash
pip install --user -r requirements.txt
```

## Run

```bash
# Launches the app 
python app.py
```

## Test

```bash
pip install --user pytest pytest-asyncio
pytest
```

## Layout

```
star_sim/
  deck.py         build_starlet_deck() -> deck + named resource handles
  lab.py          make_liquid_handler(use_hardware) -> sim or STAR backend
  env.py          observe() -> current machine state, useful for agent entry points
  channels.py     well selectors + schedule() -> the 8-channel head's geometry rules
  primitives.py   transfer() and mix(), the two operations the agent composes
  plate_map.py    PlateMap -> tracks compounds and concentrations
  tip_manager.py  TipManager -> per-tip allocation from the rack
agent.py      Tool schemas, system prompt, dispatch, and wrappers for the
              Anthropic and OpenAI-compatible APIs
app.py        Uvicorn app, provides browser-based interface
render.py     Deck geometry, per-step frame capture, and HTML rendering
tests/        pytest suite
_samples/     real Hamilton Venus method exports, for reference only
```

Wells are addressed directly: `"B3"`, a range `"A1:H1"`, or a list. Because the
8 channels sit on a fixed pitch, a transfer stays within its row and one movement
covers one column — `star_sim/channels.py` turns any well list into the sequence
of movements the machine would really make. See `CLAUDE.md` for the details.

## TODO
- Branching experimental designs
- Dynamic and easily configurable deck layouts (potential interop with existing Hamilton software tools?)
- Full documentation
- Advanced LLM model picker and configuration
- Long-term planning
- Multi-agent experimental validation
- Facile deployment onto in-unit Hamilton PC
- Worklist export (Venus-readable CSV/xlsx) — see `CLAUDE.md`
- HARD BENCHMARK FOR GOOD HAMILTON EXPERIMENTS

### Issues
- propose_prime can sometimes suggest "0 ul/well" volumes in anticipation of future liquid movement
- deck mouse hover info doesn't indicate cell density
- combining compounds results in a very white coloring scheme due to the math
- putting something into media causes the media tag to disappear..? including even media+media
- agent.py contains terminal-based deadcode
- weaker models such as gpt-oss:120b may produce tool calls with zero explanation