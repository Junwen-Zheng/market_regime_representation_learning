from __future__ import annotations

import numpy as np

from src.alpha_signals import build_alpha_dataset
from src.robustness import (
    aggregate_rank_ic_hac,
    newey_west_se_of_mean,
    non_overlapping_rank_ic_offsets,
    rank_ic_by_year,
)
from src.synthetic_data import generate_synthetic_equity_panel
from src.workflow import SIGNAL_COLS


def _alpha_dataset():
    panel = generate_synthetic_equity_panel(
        n_assets=30,
        n_days=260,
        n_sectors=5,
        seed=31,
    )
    return build_alpha_dataset(panel)


def test_newey_west_standard_error_is_positive_and_finite():
    rng = np.random.default_rng(32)
    innovations = rng.normal(size=500)

    values = np.zeros(500)

    for index in range(1, len(values)):
        values[index] = (
            0.7 * values[index - 1]
            + innovations[index]
        )

    standard_error = newey_west_se_of_mean(
        values,
        max_lag=9,
    )

    assert np.isfinite(standard_error)
    assert standard_error > 0


def test_aggregate_hac_summary_has_expected_fields():
    result = aggregate_rank_ic_hac(
        _alpha_dataset(),
        SIGNAL_COLS,
        max_lag=9,
    )

    expected = {
        "signal",
        "hac_max_lag",
        "mean_rank_ic",
        "std_rank_ic",
        "hac_se_mean",
        "hac_t_stat",
        "ci95_low",
        "ci95_high",
        "n_days",
    }

    assert expected.issubset(result.columns)
    assert len(result) == len(SIGNAL_COLS)
    assert result["n_days"].gt(0).all()
    assert np.isfinite(result["hac_se_mean"]).all()


def test_non_overlapping_offsets_cover_all_offsets():
    result = non_overlapping_rank_ic_offsets(
        _alpha_dataset(),
        SIGNAL_COLS,
        horizon=10,
    )

    assert len(result) == len(SIGNAL_COLS) * 10
    assert set(result["offset"]) == set(range(10))
    assert result["n_days"].gt(0).all()
    assert np.isfinite(result["mean_rank_ic"]).all()


def test_yearly_rank_ic_summary_is_nonempty():
    result = rank_ic_by_year(
        _alpha_dataset(),
        SIGNAL_COLS,
        max_lag=9,
    )

    assert not result.empty
    assert {
        "signal",
        "year",
        "mean_rank_ic",
        "hac_se_mean",
        "hac_t_stat",
        "n_days",
    }.issubset(result.columns)
    assert result["n_days"].gt(0).all()
