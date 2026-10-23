"""Integration tests: primitives + tip manager + plate map against a live sim.

The headline test is `test_serial_dilution_matches_known_final_state`, which pins
the rewritten per-well `transfer` to the exact final state the previous
column-based implementation produced (and which the canned script's own comment
documents). It is the regression net for the per-well revamp: it exercises
pairing, move ordering, mass balance, and tip accounting in one go.
"""

import pytest

from agent import SERIAL_DILUTION_SCRIPT, _dispatch, run_scripted
from star_sim import RobotEnv
from star_sim.primitives import mix, transfer
from star_sim.tip_manager import OutOfTipsError

ROWS = "ABCDEFGH"


@pytest.fixture
async def env():
    async with RobotEnv() as e:
        yield e


async def prime(env, plate, wells, volume, compound=None, conc=None):
    entry = {"plate": plate, "wells": wells, "volume_ul": volume}
    if compound:
        entry["compound"] = compound
    if conc is not None:
        entry["concentration_um"] = conc
    result = await _dispatch("propose_prime", {"reagents": [entry]}, env, 0.0, silent=True)
    assert "error" not in result, result
    return result


def vol(env, plate, well):
    return getattr(env.layout, plate).get_item(well).tracker.get_used_volume()


def conc(env, plate, well, name="dye"):
    contents = env.plate_map.get_well(plate, well)
    if not contents:
        return None
    return next((c.concentration_um for c in contents.compounds if c.name == name), None)


# --- the regression net ------------------------------------------------------

@pytest.mark.parametrize("row", list(ROWS))
async def test_serial_dilution_matches_known_final_state(env, row):
    """Two-fold dilution across 6 columns, verified per row.

    Expected state is the pre-revamp behaviour, unchanged: each step moves 100 µL
    into 100 µL of diluent, halving the concentration, and column 6 only ever
    receives so it ends at double the volume.
    """
    await run_scripted(env, SERIAL_DILUTION_SCRIPT)

    for col, expected_conc in enumerate([200, 100, 50, 25, 12.5, 6.25], start=1):
        well = f"{row}{col}"
        assert conc(env, "dest_plate", well) == pytest.approx(expected_conc), well
        expected_vol = 200.0 if col == 6 else 100.0
        assert vol(env, "dest_plate", well) == pytest.approx(expected_vol), well


async def test_serial_dilution_tip_accounting(env):
    """Three tip columns, as before: one reused set per transfer call."""
    await run_scripted(env, SERIAL_DILUTION_SCRIPT)
    assert env.tips.remaining == 96 - 3 * 8
    assert set(env.observe()["tip_rack_free_rows"]) == set(range(4, 13))


# --- single-channel behaviour ------------------------------------------------

async def test_single_well_transfer_consumes_one_tip(env):
    await prime(env, "source_plate", "C1", 500, compound="dye", conc=100)

    before = env.tips.remaining
    result = await transfer(env, "source_plate", "C1", "dest_plate", "C7", 50)

    assert env.tips.remaining == before - 1, "a one-well transfer should cost one tip"
    assert result["moves"] == 1
    assert result["wells_filled"] == 1
    assert vol(env, "dest_plate", "C7") == pytest.approx(50)
    assert vol(env, "source_plate", "C1") == pytest.approx(450)
    # Nothing else moved.
    assert [w for w in env.observe()["dest_plate"]] == ["C7"]


async def test_single_channel_concentration_is_tracked(env):
    await prime(env, "source_plate", "A1", 500, compound="dye", conc=100)
    await prime(env, "dest_plate", "A2", 50)  # diluent, no compound

    listener_result = await _dispatch(
        "transfer",
        {"src_plate": "source_plate", "src": "A1",
         "dst_plate": "dest_plate", "dst": "A2", "volume": 50},
        env, 0.0,
    )
    assert "error" not in listener_result, listener_result
    # 50 µL of 100 µM into 50 µL of blank → 50 µM.
    assert conc(env, "dest_plate", "A2") == pytest.approx(50)


async def test_partial_column_leaves_other_rows_alone(env):
    await prime(env, "source_plate", ["A1", "C1", "E1"], 300, compound="dye", conc=10)

    before = env.tips.remaining
    await transfer(env, "source_plate", ["A1", "C1", "E1"],
                   "dest_plate", ["A2", "C2", "E2"], 100)

    assert env.tips.remaining == before - 3
    assert sorted(env.observe()["dest_plate"]) == ["A2", "C2", "E2"]


# --- tip management ----------------------------------------------------------

async def test_row_a_yields_one_tip_per_column(env):
    # Consume one tip at a time to show pickups are no longer column-granular.
    used = 0
    while True:
        try:
            spots = env.tips.acquire(["A"])
        except OutOfTipsError:
            break
        await env.lh.pick_up_tips(spots)
        await env.lh.discard_tips()
        used += 1
    assert used == 12, "row A has one tip per column"
    assert env.tips.remaining == 96 - 12


async def test_partially_used_column_is_filled_before_a_fresh_one(env):
    # Take row A of column 1, then ask for rows B-H: they should still come from
    # column 1, not open column 2.
    await env.lh.pick_up_tips(env.tips.acquire(["A"]))
    await env.lh.discard_tips()
    spots = env.tips.acquire(list("BCDEFGH"))
    assert all(s.name.endswith("1") for s in spots), [s.name for s in spots]


async def test_out_of_tips_names_what_is_left(env):
    # Drain row A entirely, then a request including row A must fail informatively.
    for _ in range(12):
        await env.lh.pick_up_tips(env.tips.acquire(["A"]))
        await env.lh.discard_tips()
    with pytest.raises(OutOfTipsError, match="Free rows per column"):
        env.tips.acquire(["A", "B"])


# --- mixing ------------------------------------------------------------------

async def test_mix_leaves_volume_and_concentration_unchanged(env):
    await prime(env, "dest_plate", "A1:H1", 200, compound="dye", conc=75)

    result = await mix(env, "dest_plate", "A1:H1", 100, repetitions=2)

    assert result["wells_mixed"] == 8
    for row in ROWS:
        assert vol(env, "dest_plate", f"{row}1") == pytest.approx(200)
        assert conc(env, "dest_plate", f"{row}1") == pytest.approx(75)


async def test_mix_single_well(env):
    await prime(env, "dest_plate", "D4", 150)
    before = env.tips.remaining
    await mix(env, "dest_plate", "D4", 50)
    assert env.tips.remaining == before - 1
    assert vol(env, "dest_plate", "D4") == pytest.approx(150)


# --- tip reuse warnings ------------------------------------------------------

async def test_reuse_across_unrelated_wells_warns(env):
    await prime(env, "source_plate", "A1:H2", 500, compound="dye", conc=10)

    result = await transfer(
        env, "source_plate", "A1:H2", "dest_plate", "A5:H6", 100, tips="reuse"
    )
    # The second movement draws from column 2, which these tips never filled.
    assert "warnings" in result
    assert "carryover" in result["warnings"][0]


async def test_dilution_chain_does_not_warn(env):
    await prime(env, "dest_plate", "A1:H1", 300, compound="dye", conc=100)
    await prime(env, "dest_plate", "A2:H4", 100)

    result = await transfer(
        env, "dest_plate", "A1:H3", "dest_plate", "A2:H4", 100, tips="reuse"
    )
    # Each step aspirates from the well the previous step filled — that is the
    # pattern tip reuse is safe for, so it must stay quiet.
    assert "warnings" not in result, result.get("warnings")


# --- error surfacing ---------------------------------------------------------

async def test_cross_row_transfer_returns_error_not_crash(env):
    await prime(env, "source_plate", "A1", 500)
    result = await _dispatch(
        "transfer",
        {"src_plate": "source_plate", "src": "A1",
         "dst_plate": "dest_plate", "dst": "B5", "volume": 10},
        env, 0.0,
    )
    assert "error" in result
    assert "crosses rows" in result["error"]


async def test_overaspiration_returns_error(env):
    await prime(env, "source_plate", "A1", 20)
    result = await _dispatch(
        "transfer",
        {"src_plate": "source_plate", "src": "A1",
         "dst_plate": "dest_plate", "dst": "A5", "volume": 500},
        env, 0.0,
    )
    assert "error" in result
