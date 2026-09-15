"""The DataFrames the requests are asked about.

Copied rather than imported from `examples/`: an example is free to change, and
a frame that changes silently invalidates every gold contract written against
it. Each frame's fingerprint is pinned in the dataset manifest, and loading the
dataset refuses to go on if the two disagree.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
import pandas as pd

REGIONS = ["North", "South", "Central", "East"]
CHANNELS = ["Store", "Online"]


def sales(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-06-30", freq="D")
    df = pd.DataFrame({
        "date": rng.choice(dates, n),
        "region": rng.choice(REGIONS, n, p=[0.35, 0.25, 0.2, 0.2]),
        "channel": rng.choice(CHANNELS, n, p=[0.6, 0.4]),
        "units": rng.integers(1, 40, n),
        "price": np.round(rng.gamma(4, 12, n), 2),
        "satisfaction": np.clip(rng.normal(4.1, 0.7, n), 1, 5).round(1),
    })
    df["revenue"] = (df["units"] * df["price"]).round(2)
    return df.sort_values("date").reset_index(drop=True)


FRAMES: dict[str, Callable[[], pd.DataFrame]] = {
    "sales": sales,
}


def load(name: str) -> pd.DataFrame:
    try:
        return FRAMES[name]()
    except KeyError:
        raise KeyError(f"unknown frame {name!r}; available: {sorted(FRAMES)}") from None


def fingerprint(df: pd.DataFrame) -> str:
    """A short hash of the frame's content, dtypes included."""
    digest = hashlib.sha256()
    digest.update(repr(df.dtypes.to_dict()).encode())
    digest.update(df.to_csv(index=False).encode())
    return digest.hexdigest()[:16]
