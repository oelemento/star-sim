# star-sim — Hamilton STAR digital twin

A simulated Hamilton STARlet built on [PyLabRobot](https://docs.pylabrobot.org).
The same protocol code runs against a **simulated** robot (ChatterBox backend +
browser visualizer) or the **physical** Hamilton STAR, selected by a single flag.
Built as a safe testbed for AI-directed liquid-handling experiments.

## Why

- Develop and validate protocols with **no robot and no risk** — overfills,
  missing tips, and empty-well aspirations surface in software, not on hardware.
- Volume and tip state are tracked, so the twin reflects real deck state.
- One flag (`--hardware`) flips the exact same code onto the real STAR later.

## Install

```bash
pip3.11 install --user -r requirements.txt
```

## Run

```bash
# Headless simulation (prints every tool call and result):
python run.py

# Simulation with the live 3D browser visualizer:
python run.py --visualize

# Custom experiment goal:
python run.py --goal "dispense 50 uL from source column 1 into dest columns 1 through 4"

# Drive the physical Hamilton STAR over USB (no visualizer):
python run.py --hardware

# Save the agent's steps as a JSON replay file:
python run.py --record

# Replay a previously recorded run as a scripted run:
python run.py --replay runs/2026-06-11T12-00-00.json

# Step through each tool call for review:
python run.py --confirm
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
run.py        CLI entry point
```
