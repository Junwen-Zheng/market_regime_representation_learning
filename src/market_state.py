from __future__ import annotations

import pandas as pd


def build_market_state_features(panel: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Construct daily market-state features from a panel of asset returns."""
    required = {"date", "asset", "sector", "return_1d", "dollar_volume"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    daily = panel.groupby("date").agg(
        market_return=("return_1d", "mean"),
        cross_sectional_dispersion=("return_1d", "std"),
        breadth=("return_1d", lambda x: (x > 0).mean()),
        median_dollar_volume=("dollar_volume", "median"),
    )
    daily["realized_volatility_20d"] = daily["market_return"].rolling(lookback).std()
    daily["trend_60d"] = daily["market_return"].rolling(60).sum()
    daily["liquidity_stress"] = -daily["median_dollar_volume"].pct_change(lookback)

    sector_returns = (
        panel.groupby(["date", "sector"])["return_1d"].mean().unstack("sector").sort_index()
    )
    daily["sector_dispersion"] = sector_returns.std(axis=1)
    daily["sector_return_max_minus_min"] = sector_returns.max(axis=1) - sector_returns.min(axis=1)

    # Simple proxy for correlation stress: rolling mean absolute market return
    # normalised by cross-sectional dispersion.
    daily["market_move_to_dispersion"] = (
        daily["market_return"].abs().rolling(lookback).mean()
        / daily["cross_sectional_dispersion"].replace(0, pd.NA)
    )

    feature_cols = [
        "market_return",
        "cross_sectional_dispersion",
        "breadth",
        "realized_volatility_20d",
        "trend_60d",
        "liquidity_stress",
        "sector_dispersion",
        "sector_return_max_minus_min",
        "market_move_to_dispersion",
    ]
    return daily[feature_cols].dropna().reset_index()
