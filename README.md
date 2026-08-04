# star-sim — Hamilton STAR digital twin

A simulated Hamilton STARlet built on [PyLabRobot](https://docs.pylabrobot.org).
The same protocol code runs against a **simulated** robot (ChatterBox backend +
browser visualizer) or the **physical** Hamilton STAR, selected by a single flag.
Built as a safe testbed for AI-directed liquid-handling experiments.

## Why

- Develop and validate protocols with **no robot and no risk** — overfills,
  missing tips, and empty-well aspirations surface in software, not on hardware.
- Volume and tip state are tracked, so the twin reflects real deck state.
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
agent.py      Claude-based, local Ollama-based, and scripted runs
app.py        Uvicorn app, exposes browser-based interface
```

## TODO
- Branching experimental designs
- Dynamic and easily configurable deck layouts
- Single-channel pipetting
- Full documentation