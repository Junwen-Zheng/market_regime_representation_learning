from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import spearman_by_date


def newey_west_se_of_mean(
    values: pd.Series | np.ndarray,
    max_lag: int,
) -> float:
    """Estimate the Newey-West HAC standard error of a sample mean."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) < 2:
        return float("nan")

    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")

    effective_lag = min(max_lag, len(array) - 1)

    centered = array - array.mean()
    n_observations = len(centered)

    long_run_variance = float(
        np.dot(centered, centered) / n_observations
    )

    for lag in range(1, effective_lag + 1):
        weight = 1.0 - lag / (effective_lag + 1.0)

        autocovariance = float(
            np.dot(
                centered[lag:],
                centered[:-lag],
            )
            / n_observations
        )

        long_run_variance += (
            2.0 * weight * autocovariance
        )

    variance_of_mean = max(
        long_run_variance / n_observations,
        0.0,
    )

    return float(np.sqrt(variance_of_mean))


def _summarise_ic_values(
    values: pd.Series,
    max_lag: int,
) -> dict[str, float | int]:
    clean = pd.Series(values, dtype=float).dropna()

    mean_ic = float(clean.mean())
    std_ic = float(clean.std(ddof=1))
    hac_se = newey_west_se_of_mean(
        clean.to_numpy(),
        max_lag=max_lag,
    )

    if np.isfinite(hac_se) and hac_se > 0:
        hac_t_stat = mean_ic / hac_se
        ci95_low = mean_ic - 1.96 * hac_se
        ci95_high = mean_ic + 1.96 * hac_se
    else:
        hac_t_stat = float("nan")
        ci95_low = float("nan")
        ci95_high = float("nan")

    return {
        "mean_rank_ic": mean_ic,
        "std_rank_ic": std_ic,
        "hac_se_mean": hac_se,
        "hac_t_stat": hac_t_stat,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "n_days": len(clean),
    }


def aggregate_rank_ic_hac(
    alpha_df: pd.DataFrame,
    signal_cols: list[str],
    target_col: str = "forward_relative_return_10d",
    max_lag: int = 9,
) -> pd.DataFrame:
    """Summarise daily rank IC with HAC uncertainty estimates."""

    rows: list[dict[str, object]] = []

    for signal in signal_cols:
        daily = spearman_by_date(
            alpha_df,
            signal,
            target_col,
        )

        summary = _summarise_ic_values(
            daily["rank_ic"],
            max_lag=max_lag,
        )

        rows.append(
            {
                "signal": signal,
                "hac_max_lag": max_lag,
                **summary,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "mean_rank_ic",
        ascending=False,
    ).reset_index(drop=True)


def rank_ic_by_year(
    alpha_df: pd.DataFrame,
    signal_cols: list[str],
    target_col: str = "forward_relative_return_10d",
    max_lag: int = 9,
) -> pd.DataFrame:
    """Summarise signal rank IC separately for each calendar year."""

    rows: list[dict[str, object]] = []

    for signal in signal_cols:
        daily = spearman_by_date(
            alpha_df,
            signal,
            target_col,
        ).copy()

        daily["date"] = pd.to_datetime(daily["date"])
        daily["year"] = daily["date"].dt.year

        for year, year_frame in daily.groupby("year"):
            summary = _summarise_ic_values(
                year_frame["rank_ic"],
                max_lag=max_lag,
            )

            rows.append(
                {
                    "signal": signal,
                    "year": int(year),
                    "hac_max_lag": max_lag,
                    **summary,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["signal", "year"]
    ).reset_index(drop=True)


def non_overlapping_rank_ic_offsets(
    alpha_df: pd.DataFrame,
    signal_cols: list[str],
    target_col: str = "forward_relative_return_10d",
    horizon: int = 10,
) -> pd.DataFrame:
    """Evaluate each signal across non-overlapping date offsets."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    rows: list[dict[str, object]] = []

    for signal in signal_cols:
        daily = spearman_by_date(
            alpha_df,
            signal,
            target_col,
        ).sort_values("date").reset_index(drop=True)

        for offset in range(horizon):
            sampled = daily.iloc[offset::horizon]

            rows.append(
                {
                    "signal": signal,
                    "offset": offset,
                    "horizon": horizon,
                    "mean_rank_ic": float(
                        sampled["rank_ic"].mean()
                    ),
                    "std_rank_ic": float(
                        sampled["rank_ic"].std(ddof=1)
                    ),
                    "n_days": len(sampled),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["signal", "offset"]
    ).reset_index(drop=True)
