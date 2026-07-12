from __future__ import annotations

import numpy as np

from src.workflow import SIGNAL_COLS, run_research_pipeline


def test_workflow_writes_robustness_outputs(tmp_path):
    outputs = run_research_pipeline(
        output_dir=tmp_path,
        data_mode="synthetic_smoke_test",
        seed=41,
        n_assets=30,
        n_days=260,
        n_sectors=5,
        n_regimes=3,
        pca_components=2,
        min_train_days=80,
        refit_frequency=20,
    )

    expected_outputs = {
        "aggregate_signal_rank_ic_hac",
        "signal_rank_ic_by_year",
        "non_overlapping_rank_ic_offsets",
    }

    assert expected_outputs.issubset(outputs)

    expected_files = {
        "aggregate_signal_rank_ic_hac.csv",
        "signal_rank_ic_by_year.csv",
        "non_overlapping_rank_ic_offsets.csv",
    }

    for filename in expected_files:
        assert (tmp_path / filename).exists()

    hac = outputs["aggregate_signal_rank_ic_hac"]

    assert len(hac) == len(SIGNAL_COLS)
    assert hac["hac_max_lag"].eq(9).all()
    assert hac["n_days"].gt(0).all()
    assert np.isfinite(hac["hac_se_mean"]).all()
    assert hac["hac_se_mean"].gt(0).all()

    offsets = outputs["non_overlapping_rank_ic_offsets"]

    assert len(offsets) == len(SIGNAL_COLS) * 10
    assert set(offsets["offset"]) == set(range(10))
    assert offsets["n_days"].gt(0).all()

    yearly = outputs["signal_rank_ic_by_year"]

    assert not yearly.empty
    assert set(yearly["signal"]) == set(SIGNAL_COLS)
    assert yearly["n_days"].gt(0).all()
