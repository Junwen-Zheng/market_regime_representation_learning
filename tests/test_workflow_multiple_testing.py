from __future__ import annotations

import numpy as np

from src.workflow import (
    SIGNAL_COLS,
    run_research_pipeline,
)


def test_workflow_exports_multiple_testing_outputs(
    tmp_path,
):
    outputs = run_research_pipeline(
        output_dir=tmp_path,
        data_mode="synthetic_smoke_test",
        seed=131,
        n_assets=35,
        n_days=320,
        n_sectors=5,
        n_regimes=3,
        pca_components=2,
        min_train_days=100,
        refit_frequency=20,
    )

    aggregate_name = (
        "aggregate_signal_rank_ic_hac_"
        "multiple_testing"
    )

    conditional_name = (
        "conditional_rank_ic_by_regime_hac_"
        "multiple_testing"
    )

    assert aggregate_name in outputs
    assert conditional_name in outputs

    assert (
        tmp_path
        / "aggregate_signal_rank_ic_hac_"
        "multiple_testing.csv"
    ).exists()

    assert (
        tmp_path
        / "conditional_rank_ic_by_regime_hac_"
        "multiple_testing.csv"
    ).exists()

    aggregate = outputs[aggregate_name]
    conditional = outputs[conditional_name]

    correction_columns = {
        "raw_p_value",
        "bh_fdr_adjusted_p_value",
        "holm_adjusted_p_value",
        "bh_fdr_reject",
        "holm_reject",
        "multiple_testing_family",
        "hypothesis_family_size",
        "multiple_testing_alpha",
        "multiple_testing_status",
    }

    assert correction_columns.issubset(
        aggregate.columns
    )

    assert correction_columns.issubset(
        conditional.columns
    )

    assert len(aggregate) == len(SIGNAL_COLS)

    assert aggregate[
        "hypothesis_family_size"
    ].eq(len(SIGNAL_COLS)).all()

    assert aggregate[
        "multiple_testing_family"
    ].eq("aggregate_signals").all()

    assert aggregate[
        "multiple_testing_status"
    ].eq("tested").all()

    assert np.isfinite(
        aggregate[
            [
                "raw_p_value",
                "bh_fdr_adjusted_p_value",
                "holm_adjusted_p_value",
            ]
        ].to_numpy(dtype=float)
    ).all()

    eligible = conditional.loc[
        conditional["inference_eligible"]
    ]

    ineligible = conditional.loc[
        ~conditional["inference_eligible"]
    ]

    expected_family_size = len(eligible)

    assert conditional[
        "hypothesis_family_size"
    ].eq(expected_family_size).all()

    assert conditional[
        "multiple_testing_family"
    ].eq("conditional_signal_regime").all()

    assert eligible[
        "multiple_testing_status"
    ].eq("tested").all()

    assert np.isfinite(
        eligible[
            [
                "raw_p_value",
                "bh_fdr_adjusted_p_value",
                "holm_adjusted_p_value",
            ]
        ].to_numpy(dtype=float)
    ).all()

    assert ineligible[
        "multiple_testing_status"
    ].eq("not_tested_ineligible").all()

    assert ineligible[
        [
            "raw_p_value",
            "bh_fdr_adjusted_p_value",
            "holm_adjusted_p_value",
        ]
    ].isna().all().all()

    assert not ineligible[
        "bh_fdr_reject"
    ].any()

    assert not ineligible[
        "holm_reject"
    ].any()
