"""Deck layout for the Hamilton STARlet digital twin.

Defines a small, reproducible STARlet deck: one 1000 uL filtered tip rack on a
tip carrier, plus a source and destination 96-well plate on a plate carrier.

The layout is intentionally backend-agnostic: the same `build_starlet_deck()`
result is used for both the simulated and the physical robot. Resource names are
stable so protocols (and an AI agent) can refer to them by name.
"""

from dataclasses import dataclass

import pylabrobot.resources as plr
from pylabrobot.resources import STARLetDeck


@dataclass
class DeckLayout:
    """Handles to the named resources placed on the deck.

    Returned alongside the deck so callers don't have to re-fetch resources by
    string lookup. Every field is a live PyLabRobot resource.
    """

    deck: STARLetDeck
    tip_rack: plr.TipRack
    source_plate: plr.Plate
    dest_plate: plr.Plate


# Rails are the numbered slots along the STARlet deck (1 = far left).
TIP_CARRIER_RAILS = 1
PLATE_CARRIER_RAILS = 10


def build_starlet_deck() -> DeckLayout:
    """Build a STARlet deck with a tip rack and two 96-well plates.

    Returns a `DeckLayout` carrying the deck plus direct handles to the tip rack,
    source plate, and destination plate. Does not assign any liquids or set tip
    state — that is the protocol's job, so the same deck can be reused across runs.
    """
    deck = STARLetDeck()

    # Tip carrier (5 sites) with one Hamilton 1000 uL filtered rack in site 0.
    tip_carrier = plr.TIP_CAR_480_A00(name="tip_carrier")
    tip_rack = plr.hamilton_96_tiprack_1000uL_filter(name="tip_rack")
    tip_carrier[0] = tip_rack

    # Plate carrier (5 sites) with source plate in site 0, destination in site 1.
    plate_carrier = plr.PLT_CAR_L5AC_A00(name="plate_carrier")
    source_plate = plr.Cor_96_wellplate_2mL_Vb(name="source_plate")
    dest_plate = plr.Cor_96_wellplate_2mL_Vb(name="dest_plate")
    plate_carrier[0] = source_plate
    plate_carrier[1] = dest_plate

    deck.assign_child_resource(tip_carrier, rails=TIP_CARRIER_RAILS)
    deck.assign_child_resource(plate_carrier, rails=PLATE_CARRIER_RAILS)

    return DeckLayout(
        deck=deck,
        tip_rack=tip_rack,
        source_plate=source_plate,
        dest_plate=dest_plate,
    )
