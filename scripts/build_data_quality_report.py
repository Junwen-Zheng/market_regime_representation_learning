from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data-quality reports for a real OHLCV panel."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/prices.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/real"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = pd.read_csv(args.input)
    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    numeric = frame[numeric_columns].to_numpy(dtype=float)

    ordered = frame.sort_values(
        ["asset", "date"]
    ).copy()
    ordered["return_1d"] = (
        ordered.groupby("asset")["close"]
        .pct_change(fill_method=None)
    )

    asset_coverage = (
        frame.groupby(["asset", "sector"])
        .agg(
            rows=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            missing_open=("open", lambda values: int(values.isna().sum())),
            missing_high=("high", lambda values: int(values.isna().sum())),
            missing_low=("low", lambda values: int(values.isna().sum())),
            missing_close=("close", lambda values: int(values.isna().sum())),
            missing_volume=("volume", lambda values: int(values.isna().sum())),
            zero_volume_rows=("volume", lambda values: int((values == 0).sum())),
        )
        .reset_index()
        .sort_values("asset")
    )

    n_assets = int(frame["asset"].nunique())
    n_dates = int(frame["date"].nunique())
    expected_balanced_rows = n_assets * n_dates
    valid_returns = ordered["return_1d"].dropna()

    report = pd.DataFrame(
        [
            {"metric": "source_path", "value": str(args.input)},
            {"metric": "raw_rows", "value": len(frame)},
            {"metric": "assets", "value": n_assets},
            {"metric": "sectors", "value": int(frame["sector"].nunique())},
            {"metric": "trading_days", "value": n_dates},
            {"metric": "first_date", "value": frame["date"].min().date()},
            {"metric": "last_date", "value": frame["date"].max().date()},
            {
                "metric": "duplicate_date_asset_rows",
                "value": int(frame.duplicated(["date", "asset"]).sum()),
            },
            {
                "metric": "missing_ohlcv_values",
                "value": int(frame[numeric_columns].isna().sum().sum()),
            },
            {
                "metric": "nonfinite_ohlcv_values",
                "value": int((~np.isfinite(numeric)).sum()),
            },
            {
                "metric": "nonpositive_price_values",
                "value": int(
                    (frame[["open", "high", "low", "close"]] <= 0)
                    .sum()
                    .sum()
                ),
            },
            {
                "metric": "negative_volume_rows",
                "value": int((frame["volume"] < 0).sum()),
            },
            {
                "metric": "high_below_low_rows",
                "value": int((frame["high"] < frame["low"]).sum()),
            },
            {
                "metric": "expected_balanced_rows",
                "value": expected_balanced_rows,
            },
            {
                "metric": "missing_balanced_panel_rows",
                "value": expected_balanced_rows - len(frame),
            },
            {
                "metric": "minimum_asset_rows",
                "value": int(asset_coverage["rows"].min()),
            },
            {
                "metric": "maximum_asset_rows",
                "value": int(asset_coverage["rows"].max()),
            },
            {
                "metric": "maximum_absolute_daily_return",
                "value": float(valid_returns.abs().max()),
            },
            {
                "metric": "daily_returns_over_20_percent",
                "value": int((valid_returns.abs() > 0.20).sum()),
            },
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report.to_csv(
        args.output_dir / "data_quality_report.csv",
        index=False,
    )
    asset_coverage.to_csv(
        args.output_dir / "asset_coverage.csv",
        index=False,
    )

    print(report.to_string(index=False))
    print()
    print(asset_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
