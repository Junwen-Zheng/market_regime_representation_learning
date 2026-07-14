from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.conditional_robustness import (
    conditional_rank_ic_hac,
    newey_west_se_of_mean_by_position,
)
from src.robustness import newey_west_se_of_mean


def _example_data():
    rng = np.random.default_rng(111)

    dates = pd.bdate_range(
        "2020-01-01",
        periods=120,
    )

    rows = []

    for date in dates:
        signals = rng.normal(size=12)
        targets = (
            0.25 * signals
            + rng.normal(size=12)
        )

        for asset_index in range(12):
            rows.append(
                {
                    "date": date,
                    "asset": f"A{asset_index:02d}",
                    "signal_a": signals[
                        asset_index
                    ],
                    "forward_relative_return_10d": (
                        targets[asset_index]
                    ),
                }
            )

    alpha_df = pd.DataFrame(rows)

    assignments = pd.DataFrame(
        {
            "date": dates,
            "regime": [
                0 if index < 90 else 3
                for index in range(len(dates))
            ],
        }
    )

    return alpha_df, assignments


def test_position_hac_matches_standard_hac_when_contiguous():
    rng = np.random.default_rng(112)
    values = rng.normal(size=300)

    expected = newey_west_se_of_mean(
        values,
        max_lag=9,
    )

    actual = newey_west_se_of_mean_by_position(
        values=values,
        positions=np.arange(len(values)),
        max_lag=9,
    )

    assert np.isclose(actual, expected)


def test_sparse_regime_is_visible_but_ineligible():
    alpha_df, assignments = _example_data()

    result = conditional_rank_ic_hac(
        alpha_df=alpha_df,
        regime_assignments=assignments,
        signal_cols=["signal_a"],
        max_lag=9,
        minimum_days=60,
    )

    eligible = result.loc[
        result["regime"] == 0
    ].iloc[0]

    sparse = result.loc[
        result["regime"] == 3
    ].iloc[0]

    assert eligible["n_ic_days"] == 90
    assert bool(
        eligible["inference_eligible"]
    )
    assert np.isfinite(
        eligible["hac_se_mean"]
    )

    assert sparse["n_ic_days"] == 30
    assert not bool(
        sparse["inference_eligible"]
    )
    assert (
        sparse["inference_status"]
        == "insufficient_sample"
    )
    assert np.isnan(sparse["hac_se_mean"])
    assert np.isnan(sparse["hac_t_stat"])
    assert np.isnan(sparse["ci95_low"])
    assert np.isnan(sparse["ci95_high"])


def test_duplicate_assignment_dates_are_rejected():
    alpha_df, assignments = _example_data()

    duplicated = pd.concat(
        [
            assignments,
            assignments.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="one row per date",
    ):
        conditional_rank_ic_hac(
            alpha_df=alpha_df,
            regime_assignments=duplicated,
            signal_cols=["signal_a"],
        )
