from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.synthetic_data import generate_synthetic_equity_panel


REQUIRED_REAL_COLUMNS = {
    "date",
    "asset",
    "sector",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_real_equity_panel(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a real multi-asset OHLCV CSV.

    No synthetic fallback is performed.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Real data file not found: {path}. "
            "No synthetic fallback was used."
        )

    frame = pd.read_csv(path)
    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    missing = sorted(REQUIRED_REAL_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Real data is missing required columns: {missing}"
        )

    if frame.empty:
        raise ValueError("Real data file is empty")

    parsed_dates = pd.to_datetime(
        frame["date"],
        errors="coerce",
        utc=True,
    )

    if parsed_dates.isna().any():
        invalid_count = int(parsed_dates.isna().sum())
        raise ValueError(
            f"Real data contains {invalid_count} invalid dates"
        )

    frame["date"] = (
        parsed_dates.dt.tz_convert(None).dt.normalize()
    )

    for column in ["asset", "sector"]:
        if frame[column].isna().any():
            raise ValueError(
                f"Real data contains missing {column} values"
            )

        frame[column] = (
            frame[column].astype(str).str.strip()
        )

        if frame[column].eq("").any():
            raise ValueError(
                f"Real data contains empty {column} values"
            )

    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    numeric_values = frame[PRICE_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        invalid_count = int(
            (~np.isfinite(numeric_values)).sum()
        )
        raise ValueError(
            f"Real data contains {invalid_count} "
            "non-finite OHLCV values"
        )

    if (
        frame[["open", "high", "low", "close"]] <= 0
    ).any().any():
        raise ValueError(
            "Real OHLC prices must be strictly positive"
        )

    if (frame["volume"] < 0).any():
        raise ValueError(
            "Real-data volume cannot be negative"
        )

    if (frame["high"] < frame["low"]).any():
        raise ValueError(
            "Real data contains rows where high is below low"
        )

    duplicates = frame.duplicated(
        subset=["date", "asset"],
        keep=False,
    )

    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(
            f"Real data contains {duplicate_count} "
            "duplicate date/asset rows"
        )

    if frame["asset"].nunique() < 3:
        raise ValueError(
            "Real data must contain at least three assets"
        )

    if frame["sector"].nunique() < 2:
        raise ValueError(
            "Real data must contain at least two sectors"
        )

    frame = frame.sort_values(
        ["asset", "date"]
    ).reset_index(drop=True)

    frame["return_1d"] = (
        frame.groupby("asset", sort=False)["close"]
        .pct_change(fill_method=None)
    )

    frame["dollar_volume"] = (
        frame["close"] * frame["volume"]
    )

    frame = frame.dropna(
        subset=["return_1d"]
    ).copy()

    output_columns = [
        "date",
        "asset",
        "sector",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "return_1d",
    ]

    result = (
        frame[output_columns]
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )

    numeric_output = result[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dollar_volume",
            "return_1d",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_output).all():
        raise ValueError(
            "Derived real-data panel contains "
            "non-finite values"
        )

    return result


def load_equity_panel(
    data_mode: str,
    data_path: str | Path | None = None,
    *,
    n_assets: int = 120,
    n_days: int = 760,
    n_sectors: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Load a selected dataset without implicit fallback."""

    if data_mode == "real":
        if data_path is None:
            raise ValueError(
                "data_path is required when "
                "data_mode='real'. "
                "No synthetic fallback was used."
            )

        return load_real_equity_panel(data_path)

    if data_mode == "synthetic_smoke_test":
        return generate_synthetic_equity_panel(
            n_assets=n_assets,
            n_days=n_days,
            n_sectors=n_sectors,
            seed=seed,
        )

    raise ValueError(
        "data_mode must be either "
        "'real' or 'synthetic_smoke_test'"
    )
