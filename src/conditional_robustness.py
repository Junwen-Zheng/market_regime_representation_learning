from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import spearman_by_date
from src.robustness import newey_west_se_of_mean


def newey_west_se_of_mean_by_position(
    values: pd.Series | np.ndarray,
    positions: pd.Series | np.ndarray,
    max_lag: int,
) -> float:
    """Estimate HAC uncertainty using actual market-day positions.

    Unlike applying Newey-West after filtering to one regime, this
    function only treats observations as lagged neighbours when their
    original market-day positions are within `max_lag`.
    """

    array = np.asarray(values, dtype=float)
    position_array = np.asarray(positions, dtype=float)

    if len(array) != len(position_array):
        raise ValueError(
            "values and positions must have equal length"
        )

    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")

    finite = (
        np.isfinite(array)
        & np.isfinite(position_array)
    )

    array = array[finite]
    position_array = position_array[finite]

    if len(array) < 2:
        return float("nan")

    order = np.argsort(position_array)

    array = array[order]
    position_array = position_array[order]

    if not np.all(
        position_array == np.floor(position_array)
    ):
        raise ValueError(
            "positions must contain integer-valued indices"
        )

    positions_int = position_array.astype(int)

    if len(np.unique(positions_int)) != len(
        positions_int
    ):
        raise ValueError("positions must be unique")

    centered = array - array.mean()
    n_observations = len(centered)

    long_run_variance = float(
        np.dot(centered, centered)
        / n_observations
    )

    value_by_position = dict(
        zip(
            positions_int.tolist(),
            centered.tolist(),
        )
    )

    effective_lag = min(
        max_lag,
        int(
            positions_int.max()
            - positions_int.min()
        ),
    )

    for lag in range(1, effective_lag + 1):
        products = [
            current_value
            * value_by_position[position - lag]
            for position, current_value
            in value_by_position.items()
            if position - lag in value_by_position
        ]

        if not products:
            continue

        weight = 1.0 - lag / (
            effective_lag + 1.0
        )

        autocovariance = (
            float(np.sum(products))
            / n_observations
        )

        long_run_variance += (
            2.0
            * weight
            * autocovariance
        )

    variance_of_mean = max(
        long_run_variance / n_observations,
        0.0,
    )

    return float(np.sqrt(variance_of_mean))


def conditional_rank_ic_hac(
    alpha_df: pd.DataFrame,
    regime_assignments: pd.DataFrame,
    signal_cols: list[str],
    target_col: str = "forward_relative_return_10d",
    regime_col: str = "regime",
    max_lag: int = 9,
    minimum_days: int = 60,
) -> pd.DataFrame:
    """Calculate regime-conditioned IC with guarded HAC inference."""

    required_assignment_columns = {
        "date",
        regime_col,
    }

    missing_assignment_columns = sorted(
        required_assignment_columns.difference(
            regime_assignments.columns
        )
    )

    if missing_assignment_columns:
        raise ValueError(
            "regime_assignments is missing columns: "
            f"{missing_assignment_columns}"
        )

    if minimum_days <= 1:
        raise ValueError(
            "minimum_days must be greater than one"
        )

    assignments = regime_assignments[
        ["date", regime_col]
    ].copy()

    assignments["date"] = pd.to_datetime(
        assignments["date"]
    )

    if assignments["date"].duplicated().any():
        raise ValueError(
            "regime_assignments must contain "
            "one row per date"
        )

    calendar = pd.DataFrame(
        {
            "date": sorted(
                pd.to_datetime(
                    alpha_df["date"]
                ).unique()
            )
        }
    )

    calendar["market_day_position"] = np.arange(
        len(calendar),
        dtype=int,
    )

    assigned_day_counts = (
        assignments.groupby(regime_col)
        .size()
        .to_dict()
    )

    regimes = sorted(
        assignments[regime_col]
        .dropna()
        .unique()
        .tolist()
    )

    rows: list[dict[str, object]] = []

    for signal in signal_cols:
        daily = spearman_by_date(
            alpha_df,
            signal,
            target_col,
        ).copy()

        daily["date"] = pd.to_datetime(
            daily["date"]
        )

        daily = daily.merge(
            calendar,
            on="date",
            how="left",
            validate="one_to_one",
        )

        conditional = daily.merge(
            assignments,
            on="date",
            how="inner",
            validate="one_to_one",
        )

        for regime in regimes:
            regime_frame = conditional.loc[
                conditional[regime_col] == regime
            ].copy()

            valid = regime_frame[
                "rank_ic"
            ].notna()

            values = regime_frame.loc[
                valid,
                "rank_ic",
            ].astype(float)

            positions = regime_frame.loc[
                valid,
                "market_day_position",
            ].astype(int)

            n_days = len(values)
            n_assigned_days = int(
                assigned_day_counts[regime]
            )

            mean_ic = (
                float(values.mean())
                if n_days > 0
                else float("nan")
            )

            std_ic = (
                float(values.std(ddof=1))
                if n_days > 1
                else float("nan")
            )

            inference_eligible = (
                n_days >= minimum_days
            )

            if inference_eligible:
                hac_se = (
                    newey_west_se_of_mean_by_position(
                        values=values,
                        positions=positions,
                        max_lag=max_lag,
                    )
                )
            else:
                hac_se = float("nan")

            if (
                inference_eligible
                and np.isfinite(hac_se)
                and hac_se > 0
            ):
                hac_t_stat = mean_ic / hac_se
                ci95_low = mean_ic - 1.96 * hac_se
                ci95_high = mean_ic + 1.96 * hac_se
            else:
                hac_t_stat = float("nan")
                ci95_low = float("nan")
                ci95_high = float("nan")

            rows.append(
                {
                    "signal": signal,
                    regime_col: regime,
                    "mean_rank_ic": mean_ic,
                    "std_rank_ic": std_ic,
                    "hac_max_lag": max_lag,
                    "hac_se_mean": hac_se,
                    "hac_t_stat": hac_t_stat,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "n_ic_days": n_days,
                    "n_assigned_days": n_assigned_days,
                    "ic_day_coverage": (
                        n_days / n_assigned_days
                    ),
                    "minimum_days": minimum_days,
                    "inference_eligible": (
                        inference_eligible
                    ),
                    "inference_status": (
                        "eligible"
                        if inference_eligible
                        else "insufficient_sample"
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["signal", regime_col]
    ).reset_index(drop=True)
