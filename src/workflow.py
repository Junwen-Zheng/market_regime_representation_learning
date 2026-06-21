from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.alpha_signals import build_alpha_dataset
from src.evaluation import aggregate_rank_ic, conditional_rank_ic_by_regime, summarise_regimes
from src.market_state import build_market_state_features
from src.regime_representation import fit_regime_model, regime_transition_matrix
from src.synthetic_data import generate_synthetic_equity_panel


SIGNAL_COLS = [
    "reversal_5d_z",
    "momentum_20d_z",
    "momentum_60d_z",
    "momentum_20d_vol_adj_z",
    "liquidity_adjusted_momentum_z",
]


def run_research_pipeline(
    output_dir: str | Path = "outputs",
    seed: int = 42,
    n_assets: int = 120,
    n_days: int = 760,
    n_sectors: int = 8,
    n_regimes: int = 4,
    pca_components: int = 3,
) -> dict[str, pd.DataFrame]:
    """Run the full research pipeline and write reproducible CSV outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    panel = generate_synthetic_equity_panel(n_assets=n_assets, n_days=n_days, n_sectors=n_sectors, seed=seed)
    market_state = build_market_state_features(panel)
    regime_result = fit_regime_model(market_state, n_regimes=n_regimes, pca_components=pca_components, random_state=seed)
    alpha_df = build_alpha_dataset(panel)

    aggregate_ic = aggregate_rank_ic(alpha_df, SIGNAL_COLS)
    conditional_ic = conditional_rank_ic_by_regime(alpha_df, regime_result.assignments, SIGNAL_COLS)
    transition = regime_transition_matrix(regime_result.assignments)
    regime_summary = summarise_regimes(market_state, regime_result.assignments)

    outputs = {
        "market_state_features": market_state,
        "regime_assignments": regime_result.assignments,
        "regime_pca_loadings": regime_result.loadings,
        "regime_pca_explained_variance": regime_result.explained_variance,
        "regime_transition_matrix": transition,
        "aggregate_signal_rank_ic": aggregate_ic,
        "conditional_rank_ic_by_regime": conditional_ic,
        "regime_summary": regime_summary,
    }
    for name, df in outputs.items():
        df.to_csv(output_path / f"{name}.csv", index=False)

    research_summary = pd.DataFrame(
        [
            {"metric": "n_assets", "value": n_assets},
            {"metric": "n_days", "value": n_days},
            {"metric": "n_regimes", "value": n_regimes},
            {"metric": "pca_components", "value": pca_components},
            {"metric": "best_aggregate_signal", "value": aggregate_ic.sort_values("mean_rank_ic", ascending=False).iloc[0]["signal"]},
            {"metric": "best_aggregate_mean_rank_ic", "value": float(aggregate_ic["mean_rank_ic"].max())},
        ]
    )
    research_summary.to_csv(output_path / "research_summary.csv", index=False)
    outputs["research_summary"] = research_summary
    return outputs
