from __future__ import annotations

import numpy as np
import pandas as pd

from src.multiple_testing import (
    apply_multiple_testing_corrections,
    benjamini_hochberg_adjust,
    holm_adjust,
)


def test_adjustments_match_known_example():
    p_values = np.array(
        [0.01, 0.04, 0.03, 0.20]
    )

    bh = benjamini_hochberg_adjust(
        p_values
    )
    holm = holm_adjust(
        p_values
    )

    expected_bh = np.array(
        [
            0.04,
            0.05333333333333334,
            0.05333333333333334,
            0.20,
        ]
    )

    expected_holm = np.array(
        [
            0.04,
            0.09,
            0.09,
            0.20,
        ]
    )

    assert np.allclose(
        bh,
        expected_bh,
    )
    assert np.allclose(
        holm,
        expected_holm,
    )


def test_ineligible_rows_are_not_tested():
    results = pd.DataFrame(
        {
            "signal": [
                "signal_a",
                "signal_b",
                "signal_c",
            ],
            "hac_t_stat": [
                3.0,
                1.0,
                np.nan,
            ],
            "inference_eligible": [
                True,
                True,
                False,
            ],
        }
    )

    corrected = (
        apply_multiple_testing_corrections(
            results,
            eligibility_col=(
                "inference_eligible"
            ),
            family_name=(
                "conditional_signal_regime"
            ),
        )
    )

    assert corrected[
        "hypothesis_family_size"
    ].eq(2).all()

    tested = corrected.loc[
        corrected["inference_eligible"]
    ]

    assert tested[
        [
            "raw_p_value",
            "bh_fdr_adjusted_p_value",
            "holm_adjusted_p_value",
        ]
    ].notna().all().all()

    ineligible = corrected.loc[
        ~corrected["inference_eligible"]
    ].iloc[0]

    assert np.isnan(
        ineligible["raw_p_value"]
    )
    assert np.isnan(
        ineligible[
            "bh_fdr_adjusted_p_value"
        ]
    )
    assert np.isnan(
        ineligible[
            "holm_adjusted_p_value"
        ]
    )

    assert not bool(
        ineligible["bh_fdr_reject"]
    )
    assert not bool(
        ineligible["holm_reject"]
    )

    assert (
        ineligible[
            "multiple_testing_status"
        ]
        == "not_tested_ineligible"
    )


def test_corrections_preserve_input_rows():
    results = pd.DataFrame(
        {
            "signal": [
                "signal_a",
                "signal_b",
                "signal_c",
            ],
            "hac_t_stat": [
                2.5,
                -2.0,
                0.2,
            ],
        }
    )

    corrected = (
        apply_multiple_testing_corrections(
            results,
            family_name="aggregate_signals",
        )
    )

    assert len(corrected) == len(results)

    assert corrected[
        "signal"
    ].tolist() == results["signal"].tolist()

    assert corrected[
        "hypothesis_family_size"
    ].eq(3).all()

    assert corrected[
        "multiple_testing_status"
    ].eq("tested").all()

    assert corrected[
        "bh_fdr_adjusted_p_value"
    ].between(0.0, 1.0).all()

    assert corrected[
        "holm_adjusted_p_value"
    ].between(0.0, 1.0).all()


def test_all_ineligible_family_returns_guarded_output():
    results = pd.DataFrame(
        {
            "signal": [
                "signal_a",
                "signal_b",
            ],
            "hac_t_stat": [
                np.nan,
                np.nan,
            ],
            "inference_eligible": [
                False,
                False,
            ],
        }
    )

    corrected = (
        apply_multiple_testing_corrections(
            results,
            eligibility_col=(
                "inference_eligible"
            ),
            family_name=(
                "conditional_signal_regime"
            ),
        )
    )

    assert corrected[
        "hypothesis_family_size"
    ].eq(0).all()

    assert corrected[
        "multiple_testing_status"
    ].eq("not_tested_ineligible").all()

    assert corrected[
        [
            "raw_p_value",
            "bh_fdr_adjusted_p_value",
            "holm_adjusted_p_value",
        ]
    ].isna().all().all()

    assert not corrected[
        "bh_fdr_reject"
    ].any()

    assert not corrected[
        "holm_reject"
    ].any()
