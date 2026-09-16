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


def weather(n: int = 1400, seed: int = 11) -> pd.DataFrame:
    """Column names a model has to copy rather than guess: spaces, capitals,
    a unit in parentheses. Rainfall has real gaps, so `isnull` means something.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n // 2, freq="D")
    stations = ["Cerro Azul", "Playa Norte", "Valle Seco"]
    day = np.tile(np.arange(n // 2), 2)[:n]
    season = 8 * np.sin(2 * np.pi * day / 365.25)
    df = pd.DataFrame({
        "Date": np.tile(dates, 2)[:n],
        "Station Name": rng.choice(stations, n, p=[0.4, 0.35, 0.25]),
        "Max Temp": np.round(22 + season + rng.normal(0, 3, n), 1),
        "Min Temp": np.round(11 + season + rng.normal(0, 2.5, n), 1),
        "Rainfall (mm)": np.round(rng.gamma(1.2, 4.0, n), 1),
        "Humidity": rng.integers(30, 96, n),
    })
    df.loc[rng.choice(n, n // 8, replace=False), "Rainfall (mm)"] = np.nan
    return df.sort_values(["Date", "Station Name"]).reset_index(drop=True)


def survey(n: int = 1200, seed: int = 23) -> pd.DataFrame:
    """One row per respondent: an id too unique to draw, a free-text column,
    and enough countries that plotting them all is a cardinality error.
    """
    rng = np.random.default_rng(seed)
    plans = ["Free", "Pro", "Enterprise"]
    products = ["Editor", "Sync", "Insights", "Mobile", "API", "Billing"]
    countries = [f"Country {i:02d}" for i in range(28)]
    fragments = ["works well", "too slow", "confusing setup", "great support",
                 "missing features", "worth the price"]
    df = pd.DataFrame({
        "respondent_id": [f"r{i:05d}" for i in rng.permutation(n)],
        "submitted": rng.choice(pd.date_range("2025-01-01", "2025-12-31", freq="D"), n),
        "plan": rng.choice(plans, n, p=[0.55, 0.32, 0.13]),
        "product": rng.choice(products, n),
        "country": rng.choice(countries, n),
        "score": np.clip(rng.normal(7.2, 2.0, n), 0, 10).round().astype(int),
        "seats": rng.integers(1, 250, n),
        "comment": rng.choice(fragments, n),
    })
    return df.sort_values("submitted").reset_index(drop=True)


FRAMES: dict[str, Callable[[], pd.DataFrame]] = {
    "sales": sales,
    "weather": weather,
    "survey": survey,
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
