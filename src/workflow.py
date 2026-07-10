from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.alpha_signals import build_alpha_dataset
from src.evaluation import aggregate_rank_ic, conditional_rank_ic_by_regime, summarise_regimes
from src.market_state import build_market_state_features
from src.regime_representation import (
    fit_predict_full_sample_regimes,
    regime_transition_matrix,
)
from src.synthetic_data import generate_synthetic_equity_panel
from src.walk_forward_regimes import build_walk_forward_regime_assignments

SIGNAL_COLS = [
    "reversal_5d_z",
    "momentum_20d_z",
    "momentum_60d_z",
    "momentum_20d_vol_adj_z",
    "liquidity_adjusted_momentum_z",
]


def _full_sample_fit_windows(regime_assignments: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "regime_model_id": 0,
                "fit_start_date": regime_assignments["model_fit_start_date"].min(),
                "fit_end_date": regime_assignments["model_fit_end_date"].max(),
                "assignment_start_date": regime_assignments["date"].min(),
                "assignment_end_date": regime_assignments["date"].max(),
                "n_train_observations": len(regime_assignments),
                "n_assignment_observations": len(regime_assignments),
                "expanding_window": False,
                "diagnostic_full_sample": True,
            }
        ]
    )


def run_research_pipeline(
    output_dir: str | Path = "outputs",
    seed: int = 42,
    n_assets: int = 120,
    n_days: int = 760,
    n_sectors: int = 8,
    n_regimes: int = 4,
    pca_components: int = 3,
    regime_mode: str = "walk_forward",
    min_train_days: int = 252,
    refit_frequency: int = 20,
    expanding_window: bool = True,
    train_window_days: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full research pipeline and write reproducible CSV outputs.

    By default, conditional regime results use walk-forward regime labels.
    The old full-sample regime fit is retained only as an explicit diagnostic
    mode because it can leak future market states into past labels.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    panel = generate_synthetic_equity_panel(
        n_assets=n_assets,
        n_days=n_days,
        n_sectors=n_sectors,
        seed=seed,
    )
    market_state = build_market_state_features(panel)

    regime_outputs: dict[str, pd.DataFrame] = {}

    if regime_mode == "walk_forward":
        walk_forward_result = build_walk_forward_regime_assignments(
            market_state=market_state,
            n_regimes=n_regimes,
            pca_components=pca_components,
            min_train_days=min_train_days,
            refit_frequency=refit_frequency,
            expanding_window=expanding_window,
            train_window_days=train_window_days,
            random_state=seed,
        )
        regime_assignments = walk_forward_result.assignments
        regime_outputs["regime_fit_windows"] = walk_forward_result.fit_windows

    elif regime_mode == "full_sample_diagnostic":
        regime_result = fit_predict_full_sample_regimes(
            market_state,
            n_regimes=n_regimes,
            pca_components=pca_components,
            random_state=seed,
        )
        regime_assignments = regime_result.assignments
        regime_outputs["regime_pca_loadings"] = regime_result.loadings
        regime_outputs["regime_pca_explained_variance"] = regime_result.explained_variance
        regime_outputs["regime_fit_windows"] = _full_sample_fit_windows(regime_assignments)

    else:
        raise ValueError(
            "regime_mode must be either 'walk_forward' or 'full_sample_diagnostic'"
        )

    alpha_df = build_alpha_dataset(panel)

    aggregate_ic = aggregate_rank_ic(alpha_df, SIGNAL_COLS)
    conditional_ic = conditional_rank_ic_by_regime(
        alpha_df,
        regime_assignments,
        SIGNAL_COLS,
    )
    transition = regime_transition_matrix(regime_assignments)
    regime_summary = summarise_regimes(market_state, regime_assignments)

    outputs = {
        "market_state_features": market_state,
        "regime_assignments": regime_assignments,
        **regime_outputs,
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
            {"metric": "regime_mode", "value": regime_mode},
            {"metric": "min_train_days", "value": min_train_days},
            {"metric": "refit_frequency", "value": refit_frequency},
            {
                "metric": "n_regime_assigned_days",
                "value": int(regime_assignments["date"].nunique()),
            },
            {
                "metric": "best_aggregate_signal",
                "value": aggregate_ic.sort_values("mean_rank_ic", ascending=False).iloc[0][
                    "signal"
                ],
            },
            {
                "metric": "best_aggregate_mean_rank_ic",
                "value": float(aggregate_ic["mean_rank_ic"].max()),
            },
        ]
    )

    research_summary.to_csv(output_path / "research_summary.csv", index=False)
    outputs["research_summary"] = research_summary

    return outputs
