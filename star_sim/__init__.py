"""star_sim: a Hamilton STAR digital twin built on PyLabRobot.

Same protocol code runs against a simulated robot (ChatterBox backend + browser
visualizer) or the physical Hamilton STAR, selected by a single flag. Intended as
a safe testbed for AI-directed liquid-handling experiments.
"""

from .deck import build_starlet_deck
from .env import RobotEnv
from .lab import make_liquid_handler
from .plate_map import Compound, PlateMap, WellContents
from .tip_manager import OutOfTipsError, TipManager

__all__ = [
    "build_starlet_deck",
    "make_liquid_handler",
    "RobotEnv",
    "TipManager",
    "OutOfTipsError",
    "PlateMap",
    "WellContents",
    "Compound",
]
