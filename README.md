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

# Simulation with the live 3D browser visualizer
python3.11 run.py --visualize

# Drive the physical Hamilton STAR over USB
python3.11 run.py --hardware
```

## Layout

```
star_sim/
  deck.py     build_starlet_deck() -> deck + named resource handles
  lab.py      make_liquid_handler(use_hardware) -> sim or STAR backend
protocols/
  serial_dilution.py   demo: 8-channel 2-fold serial dilution
run.py        CLI entry point
```

## What the demo does

An 8-channel 2-fold serial dilution across N columns of a 96-well plate:
pre-fill buffer, seed dye, then serially transfer down the row with fresh tips
per step. Deterministic and parameterized (`--columns`) so it can later be
driven by an AI agent.

## Next: AI-directed experiments

The deck exposes named resources and the protocol is a parameterized function —
the two pieces an agent layer needs. A planned next step wraps deck state as a
structured observation and the liquid-handling verbs as a tool schema, letting a
Claude agent run closed-loop design-measure-learn cycles against the twin (add a
simulated plate reader for the "measure" half).
```
