"""Tests for the data preparation steps, on small hand-made tables."""

import pandas as pd

from ml_powerlifting_potential.data import (
    apply_filters,
    build_filters,
    encode_features,
    keep_best_per_lifter,
)


def make_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a raw-like table: a valid lifter by default, each row overrides some columns."""
    valid = {
        "Name": "Valid Lifter",
        "Sex": "M",
        "Event": "SBD",
        "Equipment": "Raw",
        "Age": 25.0,
        "BodyweightKg": 80.0,
        "Best3SquatKg": 180.0,
        "Best3BenchKg": 120.0,
        "Best3DeadliftKg": 220.0,
        "TotalKg": 520.0,
        "Place": "1",
        "Tested": "Yes",
        "Date": "2020-05-01",
        "Federation": "FFForce",
    }
    df = pd.DataFrame([{**valid, **row} for row in rows])
    df["Tested"] = df["Tested"].astype("string")  # same dtype as load_raw
    return df


def test_filters_remove_each_invalid_row():
    df = make_rows(
        [
            {},  # valid
            {"Event": "B"},  # bench only meet
            {"Equipment": "Single-ply"},  # equipped
            {"Place": "DQ"},  # disqualified
            {"Place": "DD"},  # doping disqualification
            {"Best3SquatKg": -150.0},  # failed squat
            {"Best3BenchKg": 4.0},  # lighter than an empty bar
            {"Sex": "Mx"},
            {"BodyweightKg": 30.0},
            {"Age": None},  # unknown age
            {"Age": 12.0},  # youth meet
            {"Date": "1995-06-01"},  # before 2000
        ]
    )
    filtered, funnel_df = apply_filters(df, build_filters(df))

    assert len(filtered) == 1
    assert filtered.index[0] == 0
    assert funnel_df["rows_left"].iloc[0] == len(
        df
    )  # first funnel step is the raw size
    assert funnel_df["rows_left"].iloc[-1] == 1


def test_keep_best_per_lifter_keeps_highest_total():
    df = make_rows(
        [
            {"Name": "A", "TotalKg": 500.0},
            {"Name": "A", "TotalKg": 550.0},
            {"Name": "B", "TotalKg": 400.0},
        ]
    )
    best = keep_best_per_lifter(df)

    assert len(best) == 2
    assert best["Name"].is_unique
    assert best.loc[best["Name"] == "A", "TotalKg"].item() == 550.0


def test_encode_features_turns_text_into_numbers():
    df = make_rows(
        [
            {"Sex": "M", "Tested": "Yes"},
            {"Sex": "F", "Tested": None},  # empty cell: not a tested category
        ]
    )
    dataset = encode_features(df)

    assert dataset["Sex"].tolist() == [1, 0]
    assert dataset["Tested"].tolist() == [
        1,
        0,
    ]  # regression test: an empty cell used to crash
    assert dataset.isna().sum().sum() == 0
