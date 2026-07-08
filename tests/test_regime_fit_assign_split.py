from __future__ import annotations

import pytest

from src.market_state import build_market_state_features
from src.regime_representation import (
    assign_regimes,
    fit_predict_full_sample_regimes,
    fit_regime_estimator,
    fit_regime_model,
    regime_model_explained_variance,
    regime_model_loadings,
)
from src.synthetic_data import generate_synthetic_equity_panel


def _market_state():
    panel = generate_synthetic_equity_panel(
        n_assets=30,
        n_days=180,
        n_sectors=5,
        seed=11,
    )
    return build_market_state_features(panel)


def test_regime_estimator_can_assign_later_rows_without_refitting():
    market_state = _market_state()

    train = market_state.iloc[:100].copy()
    later = market_state.iloc[100:140].copy()

    model = fit_regime_estimator(
        train,
        n_regimes=3,
        pca_components=2,
        random_state=11,
    )

    assignments = assign_regimes(model, later)

    assert len(assignments) == len(later)
    assert {"date", "pc1", "pc2", "regime"}.issubset(assignments.columns)
    assert {"model_fit_start_date", "model_fit_end_date"}.issubset(assignments.columns)
    assert assignments["model_fit_end_date"].nunique() == 1
    assert assignments["model_fit_end_date"].iloc[0] == train["date"].max()


def test_fit_regime_model_keeps_backward_compatible_result_shape():
    market_state = _market_state()

    result = fit_regime_model(
        market_state,
        n_regimes=3,
        pca_components=2,
        random_state=12,
    )

    assert {"date", "pc1", "pc2", "regime"}.issubset(result.assignments.columns)
    assert {"feature", "pc1", "pc2"}.issubset(result.loadings.columns)
    assert {"component", "explained_variance_ratio"}.issubset(
        result.explained_variance.columns
    )
    assert result.pipeline is result.model.pipeline


def test_explicit_full_sample_alias_matches_compatibility_function():
    market_state = _market_state()

    result = fit_regime_model(
        market_state,
        n_regimes=3,
        pca_components=2,
        random_state=13,
    )
    alias_result = fit_predict_full_sample_regimes(
        market_state,
        n_regimes=3,
        pca_components=2,
        random_state=13,
    )

    assert result.assignments[["date", "regime"]].equals(
        alias_result.assignments[["date", "regime"]]
    )


def test_regime_metadata_helpers_use_fitted_model():
    market_state = _market_state()
    model = fit_regime_estimator(
        market_state,
        n_regimes=3,
        pca_components=2,
        random_state=14,
    )

    loadings = regime_model_loadings(model)
    explained_variance = regime_model_explained_variance(model)

    assert len(loadings) == len(model.feature_columns)
    assert list(explained_variance["component"]) == ["pc1", "pc2"]


def test_assign_regimes_rejects_missing_model_features():
    market_state = _market_state()
    model = fit_regime_estimator(
        market_state,
        n_regimes=3,
        pca_components=2,
        random_state=15,
    )

    broken = market_state.drop(columns=[model.feature_columns[0]])

    with pytest.raises(ValueError, match="missing model feature columns"):
        assign_regimes(model, broken)
