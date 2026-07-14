from __future__ import annotations

import numpy as np

from src.workflow import (
    SIGNAL_COLS,
    run_research_pipeline,
)


def test_workflow_exports_conditional_hac_output(
    tmp_path,
):
    outputs = run_research_pipeline(
        output_dir=tmp_path,
        data_mode="synthetic_smoke_test",
        seed=121,
        n_assets=35,
        n_days=320,
        n_sectors=5,
        n_regimes=3,
        pca_components=2,
        min_train_days=100,
        refit_frequency=20,
    )

    output_name = (
        "conditional_rank_ic_by_regime_hac"
    )

    assert output_name in outputs

    output_file = tmp_path / (
        "conditional_rank_ic_by_regime_hac.csv"
    )

    assert output_file.exists()

    result = outputs[output_name]

    expected_columns = {
        "signal",
        "regime",
        "mean_rank_ic",
        "std_rank_ic",
        "hac_max_lag",
        "hac_se_mean",
        "hac_t_stat",
        "ci95_low",
        "ci95_high",
        "n_ic_days",
        "n_assigned_days",
        "ic_day_coverage",
        "minimum_days",
        "inference_eligible",
        "inference_status",
    }

    assert expected_columns.issubset(result.columns)

    assert len(result) == len(SIGNAL_COLS) * 3

    assert not result.duplicated(
        ["signal", "regime"]
    ).any()

    assert result["hac_max_lag"].eq(9).all()
    assert result["minimum_days"].eq(60).all()

    assert result["n_ic_days"].gt(0).all()
    assert result["n_assigned_days"].gt(0).all()

    assert result["ic_day_coverage"].between(
        0.0,
        1.0,
    ).all()

    assert set(
        result["inference_status"]
    ).issubset(
        {
            "eligible",
            "insufficient_sample",
        }
    )

    eligible = result.loc[
        result["inference_eligible"]
    ]

    assert np.isfinite(
        eligible[
            [
                "hac_se_mean",
                "hac_t_stat",
                "ci95_low",
                "ci95_high",
            ]
        ].to_numpy(dtype=float)
    ).all()

    ineligible = result.loc[
        ~result["inference_eligible"]
    ]

    assert ineligible[
        [
            "hac_se_mean",
            "hac_t_stat",
            "ci95_low",
            "ci95_high",
        ]
    ].isna().all().all()
