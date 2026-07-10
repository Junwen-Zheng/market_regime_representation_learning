import pandas as pd
import pytest

from src.alpha_signals import build_alpha_dataset
from src.evaluation import aggregate_rank_ic, conditional_rank_ic_by_regime
from src.market_state import build_market_state_features
from src.regime_representation import fit_regime_model, regime_transition_matrix
from src.synthetic_data import generate_synthetic_equity_panel
from src.workflow import SIGNAL_COLS, run_research_pipeline


def test_synthetic_panel_has_expected_columns():
    panel = generate_synthetic_equity_panel(
        n_assets=20,
        n_days=120,
        n_sectors=4,
        seed=1,
    )

    expected = {"date", "asset", "sector", "close", "volume", "dollar_volume", "return_1d"}

    assert expected.issubset(panel.columns)
    assert panel["asset"].nunique() == 20
    assert panel["date"].nunique() == 120


def test_market_state_and_regime_model_work():
    panel = generate_synthetic_equity_panel(
        n_assets=25,
        n_days=180,
        n_sectors=5,
        seed=2,
    )
    market_state = build_market_state_features(panel)

    result = fit_regime_model(
        market_state,
        n_regimes=3,
        pca_components=2,
        random_state=2,
    )

    assert {"date", "pc1", "pc2", "regime"}.issubset(result.assignments.columns)
    assert result.assignments["regime"].nunique() == 3

    transitions = regime_transition_matrix(result.assignments)
    assert "from_regime" in transitions.columns


def test_alpha_and_conditional_evaluation_work():
    panel = generate_synthetic_equity_panel(
        n_assets=30,
        n_days=220,
        n_sectors=5,
        seed=3,
    )
    market_state = build_market_state_features(panel)
    regimes = fit_regime_model(
        market_state,
        n_regimes=3,
        pca_components=2,
        random_state=3,
    ).assignments

    alpha = build_alpha_dataset(panel)

    aggregate = aggregate_rank_ic(alpha, SIGNAL_COLS)
    conditional = conditional_rank_ic_by_regime(alpha, regimes, SIGNAL_COLS)

    assert len(aggregate) == len(SIGNAL_COLS)
    assert not conditional.empty
    assert {"regime", "signal", "mean_rank_ic"}.issubset(conditional.columns)


def test_full_workflow_defaults_to_walk_forward_regimes(tmp_path):
    outputs = run_research_pipeline(
        output_dir=tmp_path,
        seed=4,
        n_assets=30,
        n_days=220,
        n_sectors=5,
        n_regimes=3,
        pca_components=2,
        min_train_days=80,
        refit_frequency=20,
    )

    expected_files = [
        "market_state_features.csv",
        "regime_assignments.csv",
        "regime_fit_windows.csv",
        "regime_transition_matrix.csv",
        "aggregate_signal_rank_ic.csv",
        "conditional_rank_ic_by_regime.csv",
        "regime_summary.csv",
        "research_summary.csv",
    ]

    for name in expected_files:
        assert (tmp_path / name).exists()

    assignments = outputs["regime_assignments"]
    fit_windows = outputs["regime_fit_windows"]

    assert isinstance(outputs["conditional_rank_ic_by_regime"], pd.DataFrame)
    assert not outputs["conditional_rank_ic_by_regime"].empty
    assert not fit_windows.empty
    assert "regime_model_id" in assignments.columns
    assert "model_fit_end_date" in assignments.columns

    assert (
        pd.to_datetime(assignments["date"])
        > pd.to_datetime(assignments["model_fit_end_date"])
    ).all()

    summary = outputs["research_summary"]
    regime_mode = summary.loc[summary["metric"] == "regime_mode", "value"].iloc[0]
    assert regime_mode == "walk_forward"


def test_full_workflow_still_supports_full_sample_diagnostic_mode(tmp_path):
    outputs = run_research_pipeline(
        output_dir=tmp_path,
        seed=5,
        n_assets=30,
        n_days=220,
        n_sectors=5,
        n_regimes=3,
        pca_components=2,
        regime_mode="full_sample_diagnostic",
    )

    assert "regime_pca_loadings" in outputs
    assert "regime_pca_explained_variance" in outputs
    assert "regime_fit_windows" in outputs
    assert (tmp_path / "regime_pca_loadings.csv").exists()
    assert (tmp_path / "regime_pca_explained_variance.csv").exists()

    summary = outputs["research_summary"]
    regime_mode = summary.loc[summary["metric"] == "regime_mode", "value"].iloc[0]
    assert regime_mode == "full_sample_diagnostic"


def test_workflow_rejects_unknown_regime_mode(tmp_path):
    with pytest.raises(ValueError, match="regime_mode"):
        run_research_pipeline(
            output_dir=tmp_path,
            seed=6,
            n_assets=30,
            n_days=220,
            n_sectors=5,
            regime_mode="invalid_mode",
        )
