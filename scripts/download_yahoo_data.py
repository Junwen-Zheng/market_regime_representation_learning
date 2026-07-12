from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download an adjusted US-equity OHLCV panel from Yahoo."
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "config" / "real_universe.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw" / "prices.csv",
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-07-02")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def extract_asset_frame(
    downloaded: pd.DataFrame,
    asset: str,
    sector: str,
) -> pd.DataFrame:
    if downloaded.empty:
        raise RuntimeError("Yahoo returned an empty download")

    if not isinstance(downloaded.columns, pd.MultiIndex):
        asset_frame = downloaded.copy()
    elif asset in downloaded.columns.get_level_values(0):
        asset_frame = downloaded[asset].copy()
    elif asset in downloaded.columns.get_level_values(1):
        asset_frame = downloaded.xs(
            asset,
            axis=1,
            level=1,
            drop_level=True,
        ).copy()
    else:
        raise RuntimeError(
            f"Yahoo returned no column group for {asset}"
        )

    asset_frame = asset_frame.reset_index()
    asset_frame.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in asset_frame.columns
    ]

    if "datetime" in asset_frame.columns:
        asset_frame = asset_frame.rename(
            columns={"datetime": "date"}
        )

    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = sorted(required.difference(asset_frame.columns))
    if missing:
        raise ValueError(
            f"{asset} is missing downloaded columns: {missing}"
        )

    result = asset_frame[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    result["asset"] = asset
    result["sector"] = sector

    result = result.dropna(
        subset=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    return result[
        [
            "date",
            "asset",
            "sector",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


def main() -> None:
    args = parse_args()

    universe = pd.read_csv(args.universe)
    universe.columns = [
        str(column).strip().lower()
        for column in universe.columns
    ]

    required = {"asset", "sector"}
    missing = sorted(required.difference(universe.columns))
    if missing:
        raise ValueError(
            f"Universe is missing required columns: {missing}"
        )

    if args.limit is not None:
        universe = universe.head(args.limit)

    assets = universe["asset"].astype(str).str.strip().tolist()
    sector_map = dict(
        zip(
            universe["asset"].astype(str).str.strip(),
            universe["sector"].astype(str).str.strip(),
        )
    )

    downloaded = yf.download(
        tickers=assets,
        start=args.start,
        end=args.end,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=False,
        timeout=60,
    )

    frames: list[pd.DataFrame] = []

    for asset in assets:
        frame = extract_asset_frame(
            downloaded=downloaded,
            asset=asset,
            sector=sector_map[asset],
        )

        if frame.empty:
            raise RuntimeError(
                f"No complete OHLCV rows remained for {asset}"
            )

        frames.append(frame)

        print(
            f"{asset}: rows={len(frame)} "
            f"start={frame['date'].min().date()} "
            f"end={frame['date'].max().date()}"
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(
        ["date", "asset"]
    ).reset_index(drop=True)

    duplicates = int(
        panel.duplicated(["date", "asset"]).sum()
    )
    if duplicates:
        raise RuntimeError(
            f"Downloaded panel contains {duplicates} duplicate rows"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    panel.to_csv(args.output, index=False)

    print(f"Wrote {len(panel):,} rows to {args.output}")
    print(f"Assets: {panel['asset'].nunique()}")
    print(f"Sectors: {panel['sector'].nunique()}")
    print(
        "Date range: "
        f"{panel['date'].min().date()} "
        f"to {panel['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
