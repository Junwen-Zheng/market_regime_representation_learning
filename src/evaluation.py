from __future__ import annotations

import numpy as np
import pandas as pd


def spearman_by_date(df: pd.DataFrame, signal_col: str, target_col: str) -> pd.DataFrame:
    """Compute daily cross-sectional Spearman rank IC."""
    rows = []
    for date, group in df.groupby("date"):
        if group[signal_col].nunique() < 3 or group[target_col].nunique() < 3:
            ic = np.nan
        else:
            ic = group[signal_col].rank().corr(group[target_col].rank())
        rows.append({"date": date, "signal": signal_col, "rank_ic": ic})
    return pd.DataFrame(rows).dropna()


def aggregate_rank_ic(alpha_df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    """Aggregate rank-IC statistics for each signal."""
    rows = []
    for signal in signal_cols:
        daily = spearman_by_date(alpha_df, signal, "forward_relative_return_10d")
        mean_ic = daily["rank_ic"].mean()
        std_ic = daily["rank_ic"].std()
        rows.append(
            {
                "signal": signal,
                "mean_rank_ic": mean_ic,
                "std_rank_ic": std_ic,
                "ic_information_ratio": mean_ic / std_ic if std_ic and not np.isnan(std_ic) else np.nan,
                "n_days": len(daily),
            }
        )
    return pd.DataFrame(rows)


def conditional_rank_ic_by_regime(
    alpha_df: pd.DataFrame,
    regime_assignments: pd.DataFrame,
    signal_cols: list[str],
) -> pd.DataFrame:
    """Compute rank-IC statistics sliced by learned regime."""
    merged = alpha_df.merge(regime_assignments[["date", "regime"]], on="date", how="inner")
    rows = []
    for regime, regime_df in merged.groupby("regime"):
        for signal in signal_cols:
            daily = spearman_by_date(regime_df, signal, "forward_relative_return_10d")
            mean_ic = daily["rank_ic"].mean()
            std_ic = daily["rank_ic"].std()
            rows.append(
                {
                    "regime": int(regime),
                    "signal": signal,
                    "mean_rank_ic": mean_ic,
                    "std_rank_ic": std_ic,
                    "ic_information_ratio": mean_ic / std_ic if std_ic and not np.isnan(std_ic) else np.nan,
                    "n_days": len(daily),
                    "n_observations": len(regime_df),
                }
            )
    return pd.DataFrame(rows).sort_values(["signal", "regime"])


def summarise_regimes(market_state: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarise interpretable regime characteristics."""
    merged = market_state.merge(assignments[["date", "regime"]], on="date", how="inner")
    features = [c for c in merged.columns if c not in {"date", "regime"}]
    summary = merged.groupby("regime")[features].mean().reset_index()
    summary["n_days"] = merged.groupby("regime").size().values
    return summary
