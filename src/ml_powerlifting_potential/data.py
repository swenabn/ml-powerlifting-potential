"""Loading, filtering and preparing the OpenPowerlifting data."""

import pandas as pd

from ml_powerlifting_potential import config


def load_raw() -> pd.DataFrame:
    """Read only the useful columns of the raw CSV."""
    return pd.read_csv(
        config.RAW_CSV, usecols=config.USECOLS, dtype={"Tested": "string"}
    )


def build_filters(df: pd.DataFrame) -> dict[str, pd.Series]:
    """One boolean mask per row-level filter (each one judges a single row)."""
    return {
        "SBD only": df["Event"] == "SBD",
        "Raw only": df["Equipment"] == "Raw",
        "Not disqualified (DQ)": df["Place"] != "DQ",
        "No doping disqualification (DD)": df["Place"] != "DD",
        f"Squat >= {config.MIN_BAR_KG} kg": df["Best3SquatKg"] >= config.MIN_BAR_KG,
        f"Bench >= {config.MIN_BAR_KG} kg": df["Best3BenchKg"] >= config.MIN_BAR_KG,
        f"Deadlift >= {config.MIN_BAR_KG} kg": df["Best3DeadliftKg"]
        >= config.MIN_BAR_KG,
        "Sex M or F": df["Sex"].isin(["M", "F"]),
        f"Bodyweight >= {config.MIN_BODYWEIGHT_KG} kg": df["BodyweightKg"]
        >= config.MIN_BODYWEIGHT_KG,
        f"Age known and >= {config.MIN_AGE}": df["Age"] >= config.MIN_AGE,
        f"Meet date since {config.FIRST_MEET_DATE[:4]}": df["Date"]
        >= config.FIRST_MEET_DATE,
    }


def apply_filters(
    df: pd.DataFrame, filters: dict[str, pd.Series]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the masks one by one; return the filtered table and the funnel table."""
    keep = pd.Series(True, index=df.index)
    funnel = [("Raw dataset", len(df))]
    for name, mask in filters.items():
        keep = keep & mask
        funnel.append((name, int(keep.sum())))
    return df[keep], pd.DataFrame(funnel, columns=["step", "rows_left"])


def prepare_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: raw CSV -> numeric modeling table (+ funnel table)."""
    raw = load_raw()
    filtered, funnel_df = apply_filters(raw, build_filters(raw))

    best_idx = filtered.groupby("Name")[
        "TotalKg"
    ].idxmax()  # best total of each athlete
    best = filtered.loc[best_idx]
    funnel_df.loc[len(funnel_df)] = ["One row per athlete", len(best)]
    funnel_df["rows_removed"] = -funnel_df["rows_left"].diff()

    dataset = best[config.FEATURES + config.TARGETS].copy()
    dataset["Sex"] = (dataset["Sex"] == "M").astype(int)  # M -> 1, F -> 0
    dataset["Tested"] = (dataset["Tested"] == "Yes").astype(int)  # Yes -> 1, empty -> 0
    return dataset, funnel_df


def load_dataset() -> pd.DataFrame:
    """Read the dataset prepared by prepare_dataset (fast, for modeling notebooks)."""
    return pd.read_csv(config.PROCESSED_CSV)


if __name__ == "__main__":
    dataset, funnel_df = prepare_dataset()
    config.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.PROCESSED_CSV, index=False)
    print(funnel_df.to_string(index=False))
    print(dataset.shape)
