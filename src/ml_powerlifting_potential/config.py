"""Project-wide constants: paths, thresholds, column names."""

from pathlib import Path

# config.py lives in <root>/src/powerlifting_potential/, so the project root is 2 levels up
ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = (
    ROOT / "data" / "raw" / "openpowerlifting.csv"
)  # TODO: replace with the exact file name from step 1
PROCESSED_CSV = ROOT / "data" / "processed" / "dataset.csv"

SEED = 42

# Only these columns are read from the 950 MB file (about a third of the memory)
USECOLS = [
    "Name",
    "Sex",
    "Event",
    "Equipment",
    "Age",
    "BodyweightKg",
    "Best3SquatKg",
    "Best3BenchKg",
    "Best3DeadliftKg",
    "TotalKg",
    "Place",
    "Tested",
    "Date",
    "Federation",
]

FEATURES = ["Best3BenchKg", "BodyweightKg", "Age", "Sex", "Tested"]
TARGETS = ["Best3SquatKg", "Best3DeadliftKg"]

# Plausibility thresholds
MIN_AGE = 16
MIN_BODYWEIGHT_KG = 40
MIN_BAR_KG = 20  # an empty bar weighs 20 kg
FIRST_MEET_DATE = "2000-01-01"
