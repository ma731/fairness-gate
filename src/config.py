"""Central config. Everything that defines the experiment lives here, not in scripts."""

from pathlib import Path

# Raw ACS CSVs are a few GB, so they live outside OneDrive and outside the repo.
CACHE_DIR = Path.home() / ".cache" / "folktables"

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Five large, demographically varied states for the in-distribution study.
TRAIN_STATES = ["CA", "TX", "NY", "FL", "IL"]

# Held out entirely. Used once, at the end, to measure geographic shift.
SHIFT_STATES = ["NV", "MS", "WV", "ME"]

# Fit on the earliest year, tune on the next, test on a later one. Having several survey
# years is why this uses ACS rather than a static benchmark.
TRAIN_YEAR = "2015"
VAL_YEAR = "2016"
TEST_YEAR = "2018"

ALL_YEARS = [TRAIN_YEAR, VAL_YEAR, TEST_YEAR]

# ACS race recode (RAC1P). Kept verbatim from the Census codebook.
RACE_LABELS = {
    1: "White alone",
    2: "Black or African American alone",
    3: "American Indian alone",
    4: "Alaska Native alone",
    5: "American Indian and Alaska Native tribes specified",
    6: "Asian alone",
    7: "Native Hawaiian and Other Pacific Islander alone",
    8: "Some other race alone",
    9: "Two or more races",
}

SEX_LABELS = {1: "Male", 2: "Female"}

# The crossed attribute. Code is race * 10 + sex, so 12 is "White alone, Female".
# Auditing race and sex separately can show two acceptable marginals while a cell at
# their intersection is far worse, which is the documented pattern in the literature and
# the reason this exists.
INTERSECTION = "RACExSEX"


def intersection_label(code: int) -> str:
    race = RACE_LABELS.get(code // 10, f"race {code // 10}")
    sex = SEX_LABELS.get(code % 10, f"sex {code % 10}")
    return f"{race}, {sex}"

RANDOM_SEED = 20260922
