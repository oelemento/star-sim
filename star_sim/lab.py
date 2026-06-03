"""Backend factory: one switch between the simulated and physical robot.

`make_liquid_handler()` returns a configured `LiquidHandler`. Pass
`use_hardware=False` (the default) for the ChatterBox simulation backend, or
`use_hardware=True` to drive the physical Hamilton STAR over USB. The protocol
code on top is identical either way.
"""

from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.resources import set_tip_tracking, set_volume_tracking

from .deck import DeckLayout, build_starlet_deck


def make_liquid_handler(
    use_hardware: bool = False,
    num_channels: int = 8,
) -> tuple[LiquidHandler, DeckLayout]:
    """Create a LiquidHandler + deck layout for sim or hardware.

    Tip and volume tracking are enabled globally so the simulator validates
    state (no aspirating empty wells, no overfilling) and the visualizer can
    render tips and liquid levels.

    Args:
        use_hardware: False -> ChatterBox simulation backend (no robot needed);
            True -> physical Hamilton STAR over USB.
        num_channels: pipetting channels to model in simulation. Ignored by the
            STAR backend, which reports its own channel count from firmware.

    Returns:
        (liquid_handler, layout). The caller must `await liquid_handler.setup()`.
    """
    # Must be enabled before resources are created/used for tracking to apply.
    set_tip_tracking(True)
    set_volume_tracking(True)

    layout = build_starlet_deck()

    if use_hardware:
        from pylabrobot.liquid_handling.backends import STAR

        backend = STAR()
    else:
        from pylabrobot.liquid_handling.backends import LiquidHandlerChatterboxBackend

        backend = LiquidHandlerChatterboxBackend(num_channels=num_channels)

    lh = LiquidHandler(backend=backend, deck=layout.deck)
    return lh, layout
