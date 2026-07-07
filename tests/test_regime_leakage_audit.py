from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.regime_representation import fit_regime_model


def _make_market_state(n_days: int, shock_start: int | None = None) -> pd.DataFrame:
    """Build deterministic market-state features for leakage testing."""
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


@pytest.mark.xfail(
    reason=(
        "Current regime model is full-sample/leaky. Future market-state rows can "
        "affect past PCA coordinates or regime labels. This should pass only after "
        "walk-forward regime assignment is implemented."
    ),
    strict=True,
)
def test_future_market_states_do_not_change_past_regime_assignments():
    """A time-safe regime pipeline should not let future rows change past labels."""
    past_only = _make_market_state(n_days=80)
    with_future = _make_market_state(n_days=120, shock_start=80)

    past_result = fit_regime_model(
        past_only,
        n_regimes=3,
        pca_components=2,
        random_state=7,
    ).assignments

    full_sample_result = fit_regime_model(
        with_future,
        n_regimes=3,
        pca_components=2,
        random_state=7,
    ).assignments

    full_sample_past_slice = full_sample_result.loc[
        full_sample_result["date"].isin(past_only["date"]),
        ["date", "pc1", "pc2", "regime"],
    ].reset_index(drop=True)

    past_result = past_result[["date", "pc1", "pc2", "regime"]].reset_index(drop=True)

    assert_frame_equal(
        past_result,
        full_sample_past_slice,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
