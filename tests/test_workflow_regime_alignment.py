from __future__ import annotations

import numpy as np

from src.workflow import run_research_pipeline


def test_workflow_exports_regime_alignment_outputs(
    tmp_path,
):
    outputs = run_research_pipeline(
        output_dir=tmp_path,
        data_mode="synthetic_smoke_test",
        seed=101,
        n_assets=35,
        n_days=280,
        n_sectors=5,
        n_regimes=3,
        pca_components=2,
        min_train_days=100,
        refit_frequency=20,
    )

    expected_outputs = {
        "regime_assignments",
        "regime_fit_windows",
        "regime_label_mappings",
        "regime_aligned_centroids",
    }

    assert expected_outputs.issubset(outputs)

    expected_files = {
        "regime_assignments.csv",
        "regime_fit_windows.csv",
        "regime_label_mappings.csv",
        "regime_aligned_centroids.csv",
    }

    for filename in expected_files:
        assert (tmp_path / filename).exists()

    assignments = outputs["regime_assignments"]
    mappings = outputs["regime_label_mappings"]
    centroids = outputs["regime_aligned_centroids"]

    assert {
        "raw_regime",
        "regime",
        "regime_model_id",
    }.issubset(assignments.columns)

    for _, frame in mappings.groupby(
        "regime_model_id"
    ):
        assert len(frame) == 3
        assert set(frame["raw_regime"]) == {0, 1, 2}
        assert set(frame["regime"]) == {0, 1, 2}
        assert frame["raw_regime"].is_unique
        assert frame["regime"].is_unique

    numeric_centroids = centroids.select_dtypes(
        include="number"
    ).to_numpy(dtype=float)

    assert np.isfinite(numeric_centroids).all()

    merged = assignments.merge(
        mappings[
            [
                "regime_model_id",
                "raw_regime",
                "regime",
            ]
        ],
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
