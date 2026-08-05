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

## Layout

**OUTDATED**

```
star_sim/
  deck.py         build_starlet_deck() -> deck + named resource handles
  lab.py          make_liquid_handler(use_hardware) -> sim or STAR backend
  env.py          observe() -> current machine state, useful for agent entry points
  plate_map.py    PlateMap -> tracks compounds and concentrations
  tip_manager.py  TipManager -> cursor over tip rack
protocols/
  serial_dilution.py   demo: 8-channel 2-fold serial dilution (unused)
  primitives.py        protocol primitives used by the agent such as single-use column transfer or serial transfer
agent.py      Wrapper for Anthropic and OpenAI-like APIs
app.py        Uvicorn app, provides browser-based interface
```

## TODO
- Branching experimental designs
- Dynamic and easily configurable deck layouts (potential interop with existing Hamilton software tools?)
- Single-channel pipetting
- Full documentation
- Advanced LLM model picker and configuration
- Long-term planning
- Multi-agent experimental validation
- Facile deployment onto in-unit Hamilton PC

### Issues
- propose_prime can sometimes suggest "0 ul/well" volumes in anticipation of future liquid movement
- deck mouse hover info doesn't indicate cell density
- combining compounds results in a very white coloring scheme due to the math
- putting something into media causes the media tag to disappear..? including even media+media
- agent.py contains terminal-based deadcode
- weaker models such as gpt-oss:120b may produce tool calls with zero explanation