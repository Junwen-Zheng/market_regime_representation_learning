from __future__ import annotations

import pandas as pd

from src.market_state import build_market_state_features
from src.synthetic_data import generate_synthetic_equity_panel
from src.walk_forward_regimes import (
    build_walk_forward_regime_assignments,
)


def _market_state(
    n_days: int = 300,
    seed: int = 91,
) -> pd.DataFrame:
    panel = generate_synthetic_equity_panel(
        n_assets=35,
        n_days=n_days,
        n_sectors=5,
        seed=seed,
    )

    return build_market_state_features(panel)


def _run(
    market_state: pd.DataFrame,
):
    return build_walk_forward_regime_assignments(
        market_state=market_state,
        n_regimes=3,
        pca_components=2,
        min_train_days=100,
        refit_frequency=20,
        random_state=92,
    )


def test_assignments_preserve_raw_and_aligned_labels():
    result = _run(_market_state())

    assignments = result.assignments

    assert {
        "date",
        "raw_regime",
        "regime",
        "regime_model_id",
    }.issubset(assignments.columns)

    assert assignments["raw_regime"].notna().all()
    assert assignments["regime"].notna().all()

    assert assignments["raw_regime"].between(
        0,
        2,
    ).all()

    assert assignments["regime"].between(
        0,
        2,
    ).all()


def test_every_refit_has_one_to_one_label_mapping():
    result = _run(_market_state())

    mappings = result.regime_mappings

    for model_id, frame in mappings.groupby(
        "regime_model_id"
    ):
        assert set(frame["raw_regime"]) == {0, 1, 2}
        assert set(frame["regime"]) == {0, 1, 2}

        assert frame["raw_regime"].is_unique
        assert frame["regime"].is_unique

        if model_id == 0:
            assert (
                frame["mapping_method"]
                == "canonical_stress_order"
            ).all()

            assert frame["match_distance"].eq(
                0.0
            ).all()
        else:
            assert (
                frame["mapping_method"]
                == "sequential_centroid_match"
            ).all()

            assert frame["match_distance"].ge(
                0.0
            ).all()


def test_assignment_labels_agree_with_mapping_table():
    result = _run(_market_state())

    assignments = result.assignments[
        [
            "date",
            "regime_model_id",
            "raw_regime",
            "regime",
        ]
    ]

    mappings = result.regime_mappings[
        [
            "regime_model_id",
            "raw_regime",
            "regime",
        ]
    ]

    merged = assignments.merge(
        mappings,
        on=[
            "regime_model_id",
            "raw_regime",
        ],
        how="left",
        suffixes=(
            "_assignment",
            "_mapping",
        ),
        validate="many_to_one",
    )

    assert merged["regime_mapping"].notna().all()

    assert (
        merged["regime_assignment"]
        == merged["regime_mapping"]
    ).all()


def test_centroid_table_contains_each_aligned_regime():
    result = _run(_market_state())

    centroids = result.aligned_centroids

    for _, frame in centroids.groupby(
        "regime_model_id"
    ):
        assert len(frame) == 3
        assert set(frame["raw_regime"]) == {0, 1, 2}
        assert set(frame["regime"]) == {0, 1, 2}


def test_appending_future_data_does_not_change_history():
    full_market_state = _market_state(
        n_days=340,
        seed=93,
    )

    cutoff_index = 220

    historical_market_state = (
        full_market_state.iloc[:cutoff_index]
        .copy()
        .reset_index(drop=True)
    )

    historical_result = _run(
        historical_market_state
    )

    full_result = _run(full_market_state)

    historical_assignments = (
        historical_result.assignments[
            [
                "date",
                "raw_regime",
                "regime",
                "regime_model_id",
            ]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    cutoff_date = historical_market_state[
        "date"
    ].max()

    full_historical_assignments = (
        full_result.assignments.loc[
            full_result.assignments["date"]
            <= cutoff_date,
            [
                "date",
                "raw_regime",
                "regime",
                "regime_model_id",
            ],
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        historical_assignments,
        full_historical_assignments,
    )
