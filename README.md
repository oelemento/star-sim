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
# Simulation, headless — prints every robot action and a volume report
python3.11 run.py

# Simulation with pre-set serial dilution script
python3.11 run.py --scripted

# Simulation with the live 3D browser visualizer
python3.11 run.py --visualize

# Drive the physical Hamilton STAR over USB
python3.11 run.py --hardware
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
