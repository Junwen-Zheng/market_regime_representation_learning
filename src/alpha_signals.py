from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_zscore_by_date(df: pd.DataFrame, column: str) -> pd.Series:
    grouped = df.groupby("date")[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((df[column] - mean) / std).fillna(0.0)


def build_alpha_dataset(panel: pd.DataFrame, forward_horizon: int = 10) -> pd.DataFrame:
    """Build cross-sectional alpha signals and forward relative returns."""
    df = panel.sort_values(["asset", "date"]).copy()
    g = df.groupby("asset", group_keys=False)

    df["return_5d"] = g["close"].pct_change(5)
    df["return_20d"] = g["close"].pct_change(20)
    df["return_60d"] = g["close"].pct_change(60)
    df["realized_vol_20d"] = g["return_1d"].rolling(20).std().reset_index(level=0, drop=True)
    df["dollar_volume_20d"] = g["dollar_volume"].rolling(20).mean().reset_index(level=0, drop=True)

    future_return = g["close"].shift(-forward_horizon) / df["close"] - 1.0
    df["forward_return_10d"] = future_return
    df["forward_relative_return_10d"] = (
        df["forward_return_10d"]
        - df.groupby("date")["forward_return_10d"].transform("mean")
    )

    df["reversal_5d"] = -df["return_5d"]
    df["momentum_20d"] = df["return_20d"]
    df["momentum_60d"] = df["return_60d"]
    df["momentum_20d_vol_adj"] = df["return_20d"] / df["realized_vol_20d"].replace(0, np.nan)
    df["liquidity_adjusted_momentum"] = df["momentum_20d_vol_adj"] * np.log1p(
        df["dollar_volume_20d"].rank(pct=True)
    )

    signal_cols = [
        "reversal_5d",
        "momentum_20d",
        "momentum_60d",
        "momentum_20d_vol_adj",
        "liquidity_adjusted_momentum",
    ]
    for col in signal_cols:
        df[f"{col}_z"] = _safe_zscore_by_date(df, col)

    keep = [
        "date",
        "asset",
        "sector",
        "forward_relative_return_10d",
        *signal_cols,
        *[f"{col}_z" for col in signal_cols],
    ]
    return df[keep].replace([np.inf, -np.inf], np.nan).dropna()
