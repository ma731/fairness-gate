"""Fetch the raw ACS person files we need, once, into the local cache.

Run this before anything else. It is idempotent: folktables skips files already on disk.
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from folktables import ACSDataSource

from src.config import ALL_YEARS, CACHE_DIR, SHIFT_STATES, TRAIN_STATES


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    states = TRAIN_STATES + SHIFT_STATES

    for year in ALL_YEARS:
        source = ACSDataSource(
            survey_year=year, horizon="1-Year", survey="person", root_dir=str(CACHE_DIR)
        )
        for state in states:
            t0 = time.time()
            df = source.get_data(states=[state], download=True)
            print(
                f"{year} {state}: {len(df):>7,} rows  ({time.time() - t0:5.1f}s)",
                flush=True,
            )

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
