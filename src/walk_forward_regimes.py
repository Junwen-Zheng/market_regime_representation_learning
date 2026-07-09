from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.regime_representation import assign_regimes, fit_regime_estimator


@dataclass
class WalkForwardRegimeResult:
    """Walk-forward regime assignment outputs."""

    assignments: pd.DataFrame
    fit_windows: pd.DataFrame


def _validate_walk_forward_inputs(
    market_state: pd.DataFrame,
    min_train_days: int,
    refit_frequency: int,
    train_window_days: int | None,
    expanding_window: bool,
) -> None:
    if "date" not in market_state.columns:
        raise ValueError("market_state must include date column")

    if market_state["date"].duplicated().any():
        raise ValueError("market_state must contain one row per date")

    if min_train_days <= 0:
        raise ValueError("min_train_days must be positive")

    if refit_frequency <= 0:
        raise ValueError("refit_frequency must be positive")

    if len(market_state) <= min_train_days:
        raise ValueError(
            "market_state must contain more rows than min_train_days "
            "so there are dates to assign"
        )

    if not expanding_window:
        if train_window_days is None:
            raise ValueError("train_window_days is required when expanding_window=False")

        if train_window_days < min_train_days:
            raise ValueError("train_window_days must be at least min_train_days")


def build_walk_forward_regime_assignments(
    market_state: pd.DataFrame,
    n_regimes: int = 4,
    pca_components: int = 3,
    min_train_days: int = 252,
    refit_frequency: int = 20,
    expanding_window: bool = True,
    train_window_days: int | None = None,
    random_state: int = 42,
) -> WalkForwardRegimeResult:
    """Fit regimes on historical windows and assign labels to later dates.

    For each assignment window, the scaler, PCA, and KMeans model are fit only
    on rows strictly earlier than the dates being labelled.

    This function is the non-leaky regime-labelling foundation for later
    conditional IC evaluation.
    """

    ordered = market_state.copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values("date").reset_index(drop=True)

    _validate_walk_forward_inputs(
        market_state=ordered,
        min_train_days=min_train_days,
        refit_frequency=refit_frequency,
        train_window_days=train_window_days,
        expanding_window=expanding_window,
    )

    assignment_frames: list[pd.DataFrame] = []
    fit_window_records: list[dict[str, object]] = []

    assignment_start_idx = min_train_days
    regime_model_id = 0

    while assignment_start_idx < len(ordered):
        assignment_end_idx = min(assignment_start_idx + refit_frequency, len(ordered))

        if expanding_window:
            fit_start_idx = 0
        else:
            assert train_window_days is not None
            fit_start_idx = max(0, assignment_start_idx - train_window_days)

        train = ordered.iloc[fit_start_idx:assignment_start_idx].copy()
        to_assign = ordered.iloc[assignment_start_idx:assignment_end_idx].copy()

        model = fit_regime_estimator(
            market_state=train,
            n_regimes=n_regimes,
            pca_components=pca_components,
            random_state=random_state,
        )

        assignments = assign_regimes(model, to_assign)
        assignments["regime_model_id"] = regime_model_id
        assignments["assignment_window_start_date"] = to_assign["date"].min()
        assignments["assignment_window_end_date"] = to_assign["date"].max()

        assignment_frames.append(assignments)

        fit_window_records.append(
            {
                "regime_model_id": regime_model_id,
                "fit_start_date": train["date"].min(),
                "fit_end_date": train["date"].max(),
                "assignment_start_date": to_assign["date"].min(),
                "assignment_end_date": to_assign["date"].max(),
                "n_train_observations": len(train),
                "n_assignment_observations": len(to_assign),
                "n_regimes": n_regimes,
                "pca_components": pca_components,
                "random_state": random_state,
                "expanding_window": expanding_window,
            }
        )

        regime_model_id += 1
        assignment_start_idx = assignment_end_idx

    all_assignments = pd.concat(assignment_frames, ignore_index=True)
    fit_windows = pd.DataFrame(fit_window_records)

    return WalkForwardRegimeResult(
        assignments=all_assignments,
        fit_windows=fit_windows,
    )
