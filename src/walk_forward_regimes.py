from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.regime_alignment import (
    align_regime_centroids,
    canonical_regime_mapping,
    regime_centroids_in_feature_space,
)
from src.regime_representation import (
    assign_regimes,
    fit_regime_estimator,
)


@dataclass
class WalkForwardRegimeResult:
    """Walk-forward regime assignment and alignment outputs."""

    assignments: pd.DataFrame
    fit_windows: pd.DataFrame
    regime_mappings: pd.DataFrame
    aligned_centroids: pd.DataFrame


def _validate_walk_forward_inputs(
    market_state: pd.DataFrame,
    min_train_days: int,
    refit_frequency: int,
    train_window_days: int | None,
    expanding_window: bool,
) -> None:
    if "date" not in market_state.columns:
        raise ValueError(
            "market_state must include date column"
        )

    if market_state["date"].duplicated().any():
        raise ValueError(
            "market_state must contain one row per date"
        )

    if min_train_days <= 0:
        raise ValueError(
            "min_train_days must be positive"
        )

    if refit_frequency <= 0:
        raise ValueError(
            "refit_frequency must be positive"
        )

    if len(market_state) <= min_train_days:
        raise ValueError(
            "market_state must contain more rows than "
            "min_train_days so there are dates to assign"
        )

    if not expanding_window:
        if train_window_days is None:
            raise ValueError(
                "train_window_days is required when "
                "expanding_window=False"
            )

        if train_window_days < min_train_days:
            raise ValueError(
                "train_window_days must be at least "
                "min_train_days"
            )


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
    """Fit historical regime models and align labels across refits.

    Each model is fit only on rows earlier than its assignment dates.

    Raw KMeans labels are stored in `raw_regime`. The `regime` column
    contains aligned labels:

    - the first model is ordered deterministically by stress score
    - later models are matched one-to-one to the preceding aligned
      centroids using minimum-cost centroid assignment
    """

    ordered = market_state.copy()
    ordered["date"] = pd.to_datetime(
        ordered["date"]
    )
    ordered = ordered.sort_values(
        "date"
    ).reset_index(drop=True)

    _validate_walk_forward_inputs(
        market_state=ordered,
        min_train_days=min_train_days,
        refit_frequency=refit_frequency,
        train_window_days=train_window_days,
        expanding_window=expanding_window,
    )

    assignment_frames: list[pd.DataFrame] = []
    fit_window_records: list[dict[str, object]] = []
    mapping_frames: list[pd.DataFrame] = []
    centroid_frames: list[pd.DataFrame] = []

    assignment_start_idx = min_train_days
    regime_model_id = 0

    previous_model = None
    previous_aligned_centroids = None

    while assignment_start_idx < len(ordered):
        assignment_end_idx = min(
            assignment_start_idx + refit_frequency,
            len(ordered),
        )

        if expanding_window:
            fit_start_idx = 0
        else:
            assert train_window_days is not None
            fit_start_idx = max(
                0,
                assignment_start_idx - train_window_days,
            )

        train = ordered.iloc[
            fit_start_idx:assignment_start_idx
        ].copy()

        to_assign = ordered.iloc[
            assignment_start_idx:assignment_end_idx
        ].copy()

        model = fit_regime_estimator(
            market_state=train,
            n_regimes=n_regimes,
            pca_components=pca_components,
            random_state=random_state,
        )

        if previous_model is None:
            mapping = canonical_regime_mapping(
                model
            )
        else:
            assert previous_aligned_centroids is not None

            mapping = align_regime_centroids(
                previous_model=previous_model,
                previous_aligned_centroids=(
                    previous_aligned_centroids
                ),
                current_model=model,
            )

        mapping = mapping.copy()
        mapping["regime_model_id"] = regime_model_id
        mapping["fit_start_date"] = train["date"].min()
        mapping["fit_end_date"] = train["date"].max()
        mapping["assignment_start_date"] = (
            to_assign["date"].min()
        )
        mapping["assignment_end_date"] = (
            to_assign["date"].max()
        )

        raw_to_aligned = dict(
            zip(
                mapping["raw_regime"].astype(int),
                mapping["regime"].astype(int),
            )
        )

        assignments = assign_regimes(
            model,
            to_assign,
        ).rename(
            columns={"regime": "raw_regime"}
        )

        assignments["raw_regime"] = (
            assignments["raw_regime"].astype(int)
        )

        assignments["regime"] = (
            assignments["raw_regime"]
            .map(raw_to_aligned)
            .astype(int)
        )

        assignments["regime_model_id"] = (
            regime_model_id
        )
        assignments["assignment_window_start_date"] = (
            to_assign["date"].min()
        )
        assignments["assignment_window_end_date"] = (
            to_assign["date"].max()
        )

        centroids = (
            regime_centroids_in_feature_space(
                model
            )
            .merge(
                mapping[
                    [
                        "raw_regime",
                        "regime",
                        "mapping_method",
                        "match_distance",
                    ]
                ],
                on="raw_regime",
                how="inner",
                validate="one_to_one",
            )
        )

        centroids["regime_model_id"] = regime_model_id
        centroids["fit_start_date"] = (
            train["date"].min()
        )
        centroids["fit_end_date"] = (
            train["date"].max()
        )

        previous_aligned_centroids = centroids[
            [
                "regime",
                *model.feature_columns,
            ]
        ].copy()

        previous_model = model

        assignment_frames.append(assignments)
        mapping_frames.append(mapping)
        centroid_frames.append(centroids)

        fit_window_records.append(
            {
                "regime_model_id": regime_model_id,
                "fit_start_date": train["date"].min(),
                "fit_end_date": train["date"].max(),
                "assignment_start_date": (
                    to_assign["date"].min()
                ),
                "assignment_end_date": (
                    to_assign["date"].max()
                ),
                "n_train_observations": len(train),
                "n_assignment_observations": len(
                    to_assign
                ),
                "n_regimes": n_regimes,
                "pca_components": pca_components,
                "random_state": random_state,
                "expanding_window": expanding_window,
                "mean_match_distance": float(
                    mapping["match_distance"].mean()
                ),
                "max_match_distance": float(
                    mapping["match_distance"].max()
                ),
            }
        )

        regime_model_id += 1
        assignment_start_idx = assignment_end_idx

    all_assignments = pd.concat(
        assignment_frames,
        ignore_index=True,
    )

    fit_windows = pd.DataFrame(
        fit_window_records
    )

    regime_mappings = pd.concat(
        mapping_frames,
        ignore_index=True,
    )

    aligned_centroids = pd.concat(
        centroid_frames,
        ignore_index=True,
    )

    return WalkForwardRegimeResult(
        assignments=all_assignments,
        fit_windows=fit_windows,
        regime_mappings=regime_mappings,
        aligned_centroids=aligned_centroids,
    )
