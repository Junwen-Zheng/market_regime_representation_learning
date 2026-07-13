from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_state import build_market_state_features
from src.regime_alignment import (
    align_regime_centroids,
    canonical_regime_mapping,
    canonical_stress_mapping,
    minimum_cost_label_mapping,
    regime_centroids_in_feature_space,
)
from src.regime_representation import fit_regime_estimator
from src.synthetic_data import generate_synthetic_equity_panel


def _market_state() -> pd.DataFrame:
    panel = generate_synthetic_equity_panel(
        n_assets=35,
        n_days=240,
        n_sectors=5,
        seed=81,
    )
    return build_market_state_features(panel)


def test_canonical_stress_mapping_ignores_raw_label_order():
    centroids = pd.DataFrame(
        {
            "raw_regime": [0, 1, 2],
            "market_return": [0.8, 0.0, -0.8],
            "breadth": [1.0, 0.0, -1.0],
            "realized_volatility_20d": [
                -1.0,
                0.0,
                1.0,
            ],
            "liquidity_stress": [-1.0, 0.0, 1.0],
        }
    )

    permuted = centroids.copy()
    permuted["raw_regime"] = [2, 0, 1]

    first = canonical_stress_mapping(centroids)
    second = canonical_stress_mapping(permuted)

    first_profiles = centroids.merge(
        first,
        on="raw_regime",
    )[
        [
            "market_return",
            "breadth",
            "realized_volatility_20d",
            "liquidity_stress",
            "regime",
        ]
    ].sort_values("market_return")

    second_profiles = permuted.merge(
        second,
        on="raw_regime",
    )[
        [
            "market_return",
            "breadth",
            "realized_volatility_20d",
            "liquidity_stress",
            "regime",
        ]
    ].sort_values("market_return")

    assert first_profiles.reset_index(
        drop=True
    ).equals(
        second_profiles.reset_index(drop=True)
    )


def test_minimum_cost_mapping_recovers_permutation():
    previous = np.array(
        [
            [-2.0, -1.0],
            [0.0, 0.0],
            [2.0, 1.0],
        ]
    )

    current = previous[[2, 0, 1]] + 0.01

    mapping, distances = minimum_cost_label_mapping(
        previous_vectors=previous,
        current_vectors=current,
    )

    assert mapping.tolist() == [2, 0, 1]
    assert np.isfinite(distances).all()
    assert (distances < 0.02).all()


def test_centroid_extraction_is_finite():
    market_state = _market_state()

    model = fit_regime_estimator(
        market_state.iloc[:100],
        n_regimes=3,
        pca_components=2,
        random_state=82,
    )

    centroids = regime_centroids_in_feature_space(
        model
    )

    assert len(centroids) == 3
    assert {
        "raw_regime",
        "stress_score",
        *model.feature_columns,
    }.issubset(centroids.columns)

    numeric = centroids.select_dtypes(
        include="number"
    ).to_numpy(dtype=float)

    assert np.isfinite(numeric).all()


def test_sequential_alignment_is_one_to_one():
    market_state = _market_state()

    previous_model = fit_regime_estimator(
        market_state.iloc[:100],
        n_regimes=3,
        pca_components=2,
        random_state=83,
    )

    current_model = fit_regime_estimator(
        market_state.iloc[:130],
        n_regimes=3,
        pca_components=2,
        random_state=83,
    )

    initial_mapping = canonical_regime_mapping(
        previous_model
    )

    previous_centroids = (
        regime_centroids_in_feature_space(
            previous_model
        )
        .merge(
            initial_mapping[
                ["raw_regime", "regime"]
            ],
            on="raw_regime",
            how="inner",
        )
    )

    aligned = align_regime_centroids(
        previous_model=previous_model,
        previous_aligned_centroids=(
            previous_centroids
        ),
        current_model=current_model,
    )

    assert set(aligned["raw_regime"]) == {0, 1, 2}
    assert set(aligned["regime"]) == {0, 1, 2}
    assert aligned["regime"].is_unique
    assert np.isfinite(
        aligned["match_distance"]
    ).all()
    assert (
        aligned["mapping_method"]
        == "sequential_centroid_match"
    ).all()
