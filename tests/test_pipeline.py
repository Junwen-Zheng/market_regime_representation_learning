import pandas as pd

from src.alpha_signals import build_alpha_dataset
from src.evaluation import aggregate_rank_ic, conditional_rank_ic_by_regime
from src.market_state import build_market_state_features
from src.regime_representation import fit_regime_model, regime_transition_matrix
from src.synthetic_data import generate_synthetic_equity_panel
from src.workflow import SIGNAL_COLS, run_research_pipeline


def test_synthetic_panel_has_expected_columns():
    panel = generate_synthetic_equity_panel(n_assets=20, n_days=120, n_sectors=4, seed=1)
    expected = {"date", "asset", "sector", "close", "volume", "dollar_volume", "return_1d"}
    assert expected.issubset(panel.columns)
    assert panel["asset"].nunique() == 20
    assert panel["date"].nunique() == 120


def test_market_state_and_regime_model_work():
    panel = generate_synthetic_equity_panel(n_assets=25, n_days=180, n_sectors=5, seed=2)
    market_state = build_market_state_features(panel)
    result = fit_regime_model(market_state, n_regimes=3, pca_components=2, random_state=2)
    assert set(["date", "pc1", "pc2", "regime"]).issubset(result.assignments.columns)
    assert result.assignments["regime"].nunique() == 3
    transitions = regime_transition_matrix(result.assignments)
    assert "from_regime" in transitions.columns


def test_alpha_and_conditional_evaluation_work():
    panel = generate_synthetic_equity_panel(n_assets=30, n_days=220, n_sectors=5, seed=3)
    market_state = build_market_state_features(panel)
    regimes = fit_regime_model(market_state, n_regimes=3, pca_components=2, random_state=3).assignments
    alpha = build_alpha_dataset(panel)
    aggregate = aggregate_rank_ic(alpha, SIGNAL_COLS)
    conditional = conditional_rank_ic_by_regime(alpha, regimes, SIGNAL_COLS)
    assert len(aggregate) == len(SIGNAL_COLS)
    assert not conditional.empty
    assert {"regime", "signal", "mean_rank_ic"}.issubset(conditional.columns)


def test_full_workflow_writes_outputs(tmp_path):
    outputs = run_research_pipeline(
        output_dir=tmp_path,
        seed=4,
        n_assets=30,
        n_days=220,
        n_sectors=5,
        n_regimes=3,
        pca_components=2,
    )
    expected_files = [
        "market_state_features.csv",
        "regime_assignments.csv",
        "regime_transition_matrix.csv",
        "aggregate_signal_rank_ic.csv",
        "conditional_rank_ic_by_regime.csv",
        "research_summary.csv",
    ]
    for name in expected_files:
        assert (tmp_path / name).exists()
    assert isinstance(outputs["conditional_rank_ic_by_regime"], pd.DataFrame)
