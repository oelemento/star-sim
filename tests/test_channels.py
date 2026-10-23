"""Unit tests for the selector/scheduling layer.

These are pure — no PyLabRobot, no event loop — so they pin down the head-geometry
rules cheaply and precisely.
"""

import pytest

from star_sim.channels import (
    SelectionError,
    channel_index,
    describe,
    pair,
    parse_selector,
    schedule,
)


# --- selectors ---------------------------------------------------------------

def test_single_well():
    assert parse_selector("B3") == ["B3"]


def test_column_range_is_all_eight_rows():
    assert parse_selector("A1:H1") == [f"{r}1" for r in "ABCDEFGH"]


def test_range_walks_column_major():
    # Down column 1 first, then column 2 — not row-major. This is what keeps a
    # multi-column transfer from interleaving columns.
    assert parse_selector("A1:D2") == ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]


def test_range_corners_may_be_given_in_any_order():
    assert parse_selector("H1:A1") == parse_selector("A1:H1")


def test_explicit_list_is_normalized_and_order_preserved():
    assert parse_selector(["c4", "a1"]) == ["C4", "A1"]


def test_two_digit_columns():
    assert parse_selector("A12") == ["A12"]


@pytest.mark.parametrize("bad", ["I1", "A0", "", "1A", "AA1", ":", "A1:"])
def test_malformed_selectors_rejected(bad):
    with pytest.raises(SelectionError):
        parse_selector(bad)


def test_duplicate_wells_rejected():
    # Silently doubling a transfer volume is never what the caller meant.
    with pytest.raises(SelectionError, match="more than once"):
        parse_selector(["A1", "A1"])


def test_channel_index_follows_row():
    assert [channel_index(f"{r}5") for r in "ABCDEFGH"] == list(range(8))


# --- pairing -----------------------------------------------------------------

def test_equal_lengths_pair_elementwise():
    assert pair("A1:B1", "A5:B5", 10) == [("A1", "A5", 10.0), ("B1", "B5", 10.0)]


def test_source_column_tiles_across_destination_columns():
    triples = pair("A1:H1", "A2:H3", 50)
    assert len(triples) == 16
    # Every pair stays in its own row, which is what makes it schedulable.
    assert all(s[0] == d[0] for s, d, _ in triples)


def test_single_source_repeats_across_a_row():
    assert [d for _, d, _ in pair("A1", ["A2", "A3"], 5)] == ["A2", "A3"]


def test_per_destination_volumes():
    assert [v for _, _, v in pair("A1:B1", "A5:B5", [10, 20])] == [10.0, 20.0]


def test_indivisible_counts_rejected():
    with pytest.raises(SelectionError, match="Cannot pair"):
        pair("A1:H1", "A1:C1", 10)


def test_wrong_number_of_volumes_rejected():
    with pytest.raises(SelectionError, match="volumes"):
        pair("A1:B1", "A5:B5", [10, 20, 30])


@pytest.mark.parametrize("vol", [0, -5])
def test_nonpositive_volume_rejected(vol):
    with pytest.raises(SelectionError, match="positive"):
        pair("A1", "A2", vol)


# --- scheduling: the physical invariants -------------------------------------

def assert_valid(moves):
    """Every move must be something the head could actually do in one go."""
    for m in moves:
        assert len({w[1:] for w in m.src_wells}) == 1, "sources span columns"
        assert len({w[1:] for w in m.dst_wells}) == 1, "destinations span columns"
        assert len(set(m.channels)) == len(m.channels), "a channel is used twice"
        assert len(m.src_wells) == len(m.dst_wells) == len(m.volumes) == len(m.channels)
        for s, d, c in zip(m.src_wells, m.dst_wells, m.channels):
            assert channel_index(s) == channel_index(d) == c


def test_full_column_transfer_is_one_move():
    moves = schedule(pair("A1:H1", "A3:H3", 100))
    assert len(moves) == 1
    assert len(moves[0]) == 8
    assert_valid(moves)


def test_single_channel_transfer_is_one_move_one_channel():
    moves = schedule(pair("C1", "C7", 25))
    assert len(moves) == 1
    assert moves[0].channels == (2,)
    assert_valid(moves)


def test_distribution_becomes_one_move_per_destination_column():
    moves = schedule(pair("A2:H2", "A2:H6", 100))
    assert [(m.src_col, m.dst_col) for m in moves] == [(2, c) for c in range(2, 7)]
    assert_valid(moves)


def test_serial_dilution_chain_is_ordered():
    # The whole point: col1->col2 must complete before col2->col3 begins.
    moves = schedule(pair("A1:H5", "A2:H6", 100))
    assert [(m.src_col, m.dst_col) for m in moves] == [(c, c + 1) for c in range(1, 6)]
    assert_valid(moves)


def test_partial_column_keeps_its_own_channels():
    moves = schedule(pair(["A1", "C1", "E1"], ["A4", "C4", "E4"], 30))
    assert len(moves) == 1
    assert moves[0].channels == (0, 2, 4)
    assert_valid(moves)


def test_channels_are_reported_in_row_order():
    moves = schedule(pair(["E1", "A1"], ["E2", "A2"], 10))
    assert moves[0].channels == (0, 4)
    assert moves[0].src_wells == ("A1", "E1")


def test_reusing_a_channel_in_one_column_splits_the_move():
    # Both pairs need channel 0, so they cannot happen in the same movement even
    # though they share a source and destination column.
    moves = schedule([("A1", "A5", 10.0), ("A1", "A6", 10.0)])
    assert len(moves) == 2
    assert_valid(moves)


def test_cross_row_transfer_is_rejected():
    with pytest.raises(SelectionError, match="crosses rows"):
        schedule(pair("A1", "B5", 10))


def test_describe_is_compact():
    assert describe(["A1"]) == "A1"
    assert "8 wells" in describe([f"{r}1" for r in "ABCDEFGH"])
