from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.walk_forward_regimes import build_walk_forward_regime_assignments


def _make_market_state(n_days: int, shock_start: int | None = None) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    t = np.arange(n_days, dtype=float)

    market_state = pd.DataFrame(
        {
            "date": dates,
            "market_return": 0.001 * np.sin(t / 7.0),
            "realised_volatility": 0.02 + 0.002 * np.cos(t / 11.0),
            "cross_sectional_dispersion": 0.01 + 0.001 * np.sin(t / 5.0),
            "breadth": 0.50 + 0.05 * np.cos(t / 9.0),
            "liquidity_stress": 0.10 + 0.01 * np.sin(t / 13.0),
            "sector_dispersion": 0.015 + 0.002 * np.cos(t / 6.0),
            "trend_strength": 0.02 * np.sin(t / 17.0),
        }
    )

    if shock_start is not None:
        future_mask = np.arange(n_days) >= shock_start

        market_state.loc[future_mask, "market_return"] += 0.08
        market_state.loc[future_mask, "realised_volatility"] += 0.20
        market_state.loc[future_mask, "cross_sectional_dispersion"] += 0.10
        market_state.loc[future_mask, "breadth"] -= 0.35
        market_state.loc[future_mask, "liquidity_stress"] += 0.80
        market_state.loc[future_mask, "sector_dispersion"] += 0.15
        market_state.loc[future_mask, "trend_strength"] -= 0.20

    return market_state


def test_walk_forward_assignments_use_only_prior_fit_window():
    market_state = _make_market_state(n_days=140)

    result = build_walk_forward_regime_assignments(
        market_state,
        n_regimes=3,
        pca_components=2,
        min_train_days=60,
        refit_frequency=20,
        random_state=21,
    )

    assignments = result.assignments
    fit_windows = result.fit_windows

    assert not assignments.empty
    assert not fit_windows.empty
    assert {"date", "pc1", "pc2", "regime"}.issubset(assignments.columns)
    assert {
        "model_fit_start_date",
        "model_fit_end_date",
        "regime_model_id",
        "assignment_window_start_date",
        "assignment_window_end_date",
    }.issubset(assignments.columns)

    assert (
        pd.to_datetime(assignments["date"])
        > pd.to_datetime(assignments["model_fit_end_date"])
    ).all()

    assert (
        pd.to_datetime(fit_windows["assignment_start_date"])
        > pd.to_datetime(fit_windows["fit_end_date"])
    ).all()


def test_walk_forward_refits_across_multiple_assignment_windows():
    market_state = _make_market_state(n_days=155)

    result = build_walk_forward_regime_assignments(
        market_state,
        n_regimes=3,
        pca_components=2,
        min_train_days=55,
        refit_frequency=25,
        random_state=22,
    )

    assignments = result.assignments
    fit_windows = result.fit_windows

    assert assignments["regime_model_id"].nunique() == len(fit_windows)
    assert assignments["regime_model_id"].nunique() > 1
    assert len(assignments) == len(market_state) - 55

    window_sizes = fit_windows["n_assignment_observations"].tolist()
    assert window_sizes[:-1] == [25, 25, 25, 25]
    assert window_sizes[-1] == 0 or window_sizes[-1] <= 25


def test_future_rows_do_not_change_past_walk_forward_assignments():
    base = _make_market_state(n_days=120)
    with_future_shock = _make_market_state(n_days=160, shock_start=120)

    base_result = build_walk_forward_regime_assignments(
        base,
        n_regimes=3,
        pca_components=2,
        min_train_days=50,
        refit_frequency=10,
        random_state=23,
    ).assignments

    shocked_result = build_walk_forward_regime_assignments(
        with_future_shock,
        n_regimes=3,
        pca_components=2,
        min_train_days=50,
        refit_frequency=10,
        random_state=23,
    ).assignments

    shocked_past = shocked_result.loc[
        shocked_result["date"].isin(base_result["date"]),
        [
            "date",
            "pc1",
            "pc2",
            "regime",
            "model_fit_start_date",
            "model_fit_end_date",
            "regime_model_id",
        ],
    ].reset_index(drop=True)

    base_past = base_result[
        [
            "date",
            "pc1",
            "pc2",
            "regime",
            "model_fit_start_date",
            "model_fit_end_date",
            "regime_model_id",
        ]
    ].reset_index(drop=True)

    assert_frame_equal(
        base_past,
        shocked_past,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_walk_forward_rejects_duplicate_dates():
    market_state = _make_market_state(n_days=80)
    duplicated = pd.concat([market_state, market_state.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="one row per date"):
        build_walk_forward_regime_assignments(
            duplicated,
            n_regimes=3,
            pca_components=2,
            min_train_days=40,
            refit_frequency=10,
            random_state=24,
        )


def test_rolling_window_requires_train_window_days():
    market_state = _make_market_state(n_days=100)

    with pytest.raises(ValueError, match="train_window_days is required"):
        build_walk_forward_regime_assignments(
            market_state,
            n_regimes=3,
            pca_components=2,
            min_train_days=50,
            refit_frequency=10,
            expanding_window=False,
            random_state=25,
        )


def test_rolling_window_uses_fixed_length_training_history():
    market_state = _make_market_state(n_days=130)

    result = build_walk_forward_regime_assignments(
        market_state,
        n_regimes=3,
        pca_components=2,
        min_train_days=50,
        refit_frequency=20,
        expanding_window=False,
        train_window_days=50,
        random_state=26,
    )

    assert result.fit_windows["n_train_observations"].eq(50).all()
    assert not result.fit_windows["expanding_window"].any()
