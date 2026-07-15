from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.alpha_signals import build_alpha_dataset
from src.conditional_robustness import conditional_rank_ic_hac
from src.multiple_testing import apply_multiple_testing_corrections
from src.data_loader import load_equity_panel
from src.evaluation import (
    aggregate_rank_ic,
    conditional_rank_ic_by_regime,
    summarise_regimes,
)
from src.market_state import build_market_state_features
from src.regime_representation import (
    fit_predict_full_sample_regimes,
    regime_transition_matrix,
)
from src.robustness import (
    aggregate_rank_ic_hac,
    non_overlapping_rank_ic_offsets,
    rank_ic_by_year,
)
from src.walk_forward_regimes import build_walk_forward_regime_assignments


SIGNAL_COLS = [
    "reversal_5d_z",
    "momentum_20d_z",
    "momentum_60d_z",
    "momentum_20d_vol_adj_z",
    "liquidity_adjusted_momentum_z",
]


def _full_sample_fit_windows(
    regime_assignments: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "regime_model_id": 0,
                "fit_start_date": regime_assignments[
                    "model_fit_start_date"
                ].min(),
                "fit_end_date": regime_assignments[
                    "model_fit_end_date"
                ].max(),
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
    data_mode: str = "real",
    data_path: str | Path | None = None,
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
    """Run the research workflow using an explicit data source.

    Real data is the default. Synthetic data is available only through
    data_mode='synthetic_smoke_test'. Missing real data never triggers an
    implicit synthetic fallback.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    panel = load_equity_panel(
        data_mode=data_mode,
        data_path=data_path,
        n_assets=n_assets,
        n_days=n_days,
        n_sectors=n_sectors,
        seed=seed,
    )

    actual_n_assets = int(panel["asset"].nunique())
    actual_n_days = int(panel["date"].nunique())
    actual_n_sectors = int(panel["sector"].nunique())

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
        regime_outputs["regime_fit_windows"] = (
            walk_forward_result.fit_windows
        )
        regime_outputs["regime_label_mappings"] = (
            walk_forward_result.regime_mappings
        )
        regime_outputs["regime_aligned_centroids"] = (
            walk_forward_result.aligned_centroids
        )

    elif regime_mode == "full_sample_diagnostic":
        regime_result = fit_predict_full_sample_regimes(
            market_state,
            n_regimes=n_regimes,
            pca_components=pca_components,
            random_state=seed,
        )

        regime_assignments = regime_result.assignments
        regime_outputs["regime_pca_loadings"] = regime_result.loadings
        regime_outputs["regime_pca_explained_variance"] = (
            regime_result.explained_variance
        )
        regime_outputs["regime_fit_windows"] = (
            _full_sample_fit_windows(regime_assignments)
        )

    else:
        raise ValueError(
            "regime_mode must be either "
            "'walk_forward' or 'full_sample_diagnostic'"
        )

    alpha_df = build_alpha_dataset(panel)

    aggregate_ic = aggregate_rank_ic(
        alpha_df,
        SIGNAL_COLS,
    )

    aggregate_ic_hac = aggregate_rank_ic_hac(
        alpha_df,
        SIGNAL_COLS,
        max_lag=9,
    )

    aggregate_ic_hac_multiple_testing = (
        apply_multiple_testing_corrections(
            aggregate_ic_hac,
            statistic_col="hac_t_stat",
            family_name="aggregate_signals",
            alpha=0.05,
        )
    )

    yearly_ic = rank_ic_by_year(
        alpha_df,
        SIGNAL_COLS,
        max_lag=9,
    )

    non_overlapping_ic = non_overlapping_rank_ic_offsets(
        alpha_df,
        SIGNAL_COLS,
        horizon=10,
    )

    conditional_ic = conditional_rank_ic_by_regime(
        alpha_df,
        regime_assignments,
        SIGNAL_COLS,
    )

    conditional_ic_hac = conditional_rank_ic_hac(
        alpha_df=alpha_df,
        regime_assignments=regime_assignments,
        signal_cols=SIGNAL_COLS,
        max_lag=9,
        minimum_days=60,
    )

    conditional_ic_hac_multiple_testing = (
        apply_multiple_testing_corrections(
            conditional_ic_hac,
            statistic_col="hac_t_stat",
            eligibility_col="inference_eligible",
            family_name="conditional_signal_regime",
            alpha=0.05,
        )
    )

    transition = regime_transition_matrix(
        regime_assignments
    )

    regime_summary = summarise_regimes(
        market_state,
        regime_assignments,
    )

    outputs = {
        "market_state_features": market_state,
        "regime_assignments": regime_assignments,
        **regime_outputs,
        "regime_transition_matrix": transition,
        "aggregate_signal_rank_ic": aggregate_ic,
        "aggregate_signal_rank_ic_hac": aggregate_ic_hac,
        "aggregate_signal_rank_ic_hac_multiple_testing": aggregate_ic_hac_multiple_testing,
        "signal_rank_ic_by_year": yearly_ic,
        "non_overlapping_rank_ic_offsets": non_overlapping_ic,
        "conditional_rank_ic_by_regime": conditional_ic,
        "conditional_rank_ic_by_regime_hac": conditional_ic_hac,
        "conditional_rank_ic_by_regime_hac_multiple_testing": conditional_ic_hac_multiple_testing,
        "regime_summary": regime_summary,
    }

    for name, frame in outputs.items():
        frame.to_csv(
            output_path / f"{name}.csv",
            index=False,
        )

    research_summary = pd.DataFrame(
        [
            {
                "metric": "data_mode",
                "value": data_mode,
            },
            {
                "metric": "data_path",
                "value": (
                    str(data_path)
                    if data_path is not None
                    else ""
                ),
            },
            {
                "metric": "n_assets",
                "value": actual_n_assets,
            },
            {
                "metric": "n_days",
                "value": actual_n_days,
            },
            {
                "metric": "n_sectors",
                "value": actual_n_sectors,
            },
            {
                "metric": "n_regimes",
                "value": n_regimes,
            },
            {
                "metric": "pca_components",
                "value": pca_components,
            },
            {
                "metric": "regime_mode",
                "value": regime_mode,
            },
            {
                "metric": "min_train_days",
                "value": min_train_days,
            },
            {
                "metric": "refit_frequency",
                "value": refit_frequency,
            },
            {
                "metric": "n_regime_assigned_days",
                "value": int(
                    regime_assignments["date"].nunique()
                ),
            },
            {
                "metric": "best_aggregate_signal",
                "value": aggregate_ic.sort_values(
                    "mean_rank_ic",
                    ascending=False,
                ).iloc[0]["signal"],
            },
            {
                "metric": "best_aggregate_mean_rank_ic",
                "value": float(
                    aggregate_ic["mean_rank_ic"].max()
                ),
            },
        ]
    )

    research_summary.to_csv(
        output_path / "research_summary.csv",
        index=False,
    )

    outputs["research_summary"] = research_summary

    return outputs
