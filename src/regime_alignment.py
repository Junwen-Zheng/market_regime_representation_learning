from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.regime_representation import RegimeModel


STRESS_WEIGHTS = {
    "market_return": -0.5,
    "cross_sectional_dispersion": 1.0,
    "breadth": -1.0,
    "realized_volatility_20d": 1.0,
    "trend_60d": -0.5,
    "liquidity_stress": 1.0,
    "sector_dispersion": 1.0,
    "sector_return_max_minus_min": 1.0,
    "market_move_to_dispersion": 0.5,
}


def _scaled_centroid_frame(
    model: RegimeModel,
) -> pd.DataFrame:
    """Return KMeans centroids in standardised feature coordinates."""

    pca = model.pipeline.named_steps["pca"]
    kmeans = model.pipeline.named_steps["kmeans"]

    scaled_centroids = pca.inverse_transform(
        kmeans.cluster_centers_
    )

    frame = pd.DataFrame(
        scaled_centroids,
        columns=model.feature_columns,
    )
    frame.insert(
        0,
        "raw_regime",
        np.arange(model.n_regimes, dtype=int),
    )

    return frame


def _stress_scores(
    scaled_centroids: pd.DataFrame,
) -> pd.Series:
    """Build a transparent heuristic stress score.

    Positive values indicate relatively higher statistical stress.
    The score is used only to order the first fitted model. Later
    models are matched to the preceding model by centroid distance.
    """

    feature_columns = [
        column
        for column in scaled_centroids.columns
        if column != "raw_regime"
    ]

    score = pd.Series(
        np.zeros(len(scaled_centroids), dtype=float),
        index=scaled_centroids.index,
    )

    recognised_features = 0

    for feature, weight in STRESS_WEIGHTS.items():
        if feature in scaled_centroids.columns:
            score = score + weight * scaled_centroids[feature]
            recognised_features += 1

    if recognised_features == 0:
        score = scaled_centroids[feature_columns].sum(axis=1)

    return score.astype(float)


def canonical_stress_mapping(
    scaled_centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Assign deterministic first-window labels by stress ordering.

    Regime zero is the lowest-stress centroid and the largest regime
    number is the highest-stress centroid under the documented score.
    """

    if "raw_regime" not in scaled_centroids.columns:
        raise ValueError(
            "scaled_centroids must include raw_regime"
        )

    feature_columns = [
        column
        for column in scaled_centroids.columns
        if column != "raw_regime"
    ]

    if not feature_columns:
        raise ValueError(
            "scaled_centroids must include feature columns"
        )

    working = scaled_centroids.copy()
    working["stress_score"] = _stress_scores(working)

    ordered = working.sort_values(
        [
            "stress_score",
            *feature_columns,
            "raw_regime",
        ]
    ).reset_index(drop=True)

    ordered["regime"] = np.arange(
        len(ordered),
        dtype=int,
    )
    ordered["mapping_method"] = "canonical_stress_order"
    ordered["match_distance"] = 0.0

    return ordered[
        [
            "raw_regime",
            "regime",
            "stress_score",
            "mapping_method",
            "match_distance",
        ]
    ].sort_values("raw_regime").reset_index(drop=True)


def canonical_regime_mapping(
    model: RegimeModel,
) -> pd.DataFrame:
    """Create deterministic aligned labels for the first fitted model."""

    return canonical_stress_mapping(
        _scaled_centroid_frame(model)
    )


def regime_centroids_in_feature_space(
    model: RegimeModel,
) -> pd.DataFrame:
    """Return cluster centroids in original market-feature units."""

    scaler = model.pipeline.named_steps["scaler"]
    scaled_centroids = _scaled_centroid_frame(model)

    original_centroids = scaler.inverse_transform(
        scaled_centroids[model.feature_columns]
    )

    result = pd.DataFrame(
        original_centroids,
        columns=model.feature_columns,
    )
    result.insert(
        0,
        "raw_regime",
        scaled_centroids["raw_regime"].to_numpy(),
    )
    result["stress_score"] = _stress_scores(
        scaled_centroids
    ).to_numpy()

    return result


def minimum_cost_label_mapping(
    previous_vectors: np.ndarray,
    current_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each current centroid to one previous centroid."""

    previous = np.asarray(
        previous_vectors,
        dtype=float,
    )
    current = np.asarray(
        current_vectors,
        dtype=float,
    )

    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError(
            "centroid vectors must be two-dimensional"
        )

    if previous.shape != current.shape:
        raise ValueError(
            "previous and current centroid arrays "
            "must have the same shape"
        )

    if not np.isfinite(previous).all():
        raise ValueError(
            "previous centroid vectors must be finite"
        )

    if not np.isfinite(current).all():
        raise ValueError(
            "current centroid vectors must be finite"
        )

    differences = (
        current[:, np.newaxis, :]
        - previous[np.newaxis, :, :]
    )

    cost_matrix = np.sqrt(
        np.mean(differences**2, axis=2)
    )

    current_rows, previous_columns = (
        linear_sum_assignment(cost_matrix)
    )

    mapping = np.full(
        len(current),
        fill_value=-1,
        dtype=int,
    )
    distances = np.full(
        len(current),
        fill_value=np.nan,
        dtype=float,
    )

    mapping[current_rows] = previous_columns
    distances[current_rows] = cost_matrix[
        current_rows,
        previous_columns,
    ]

    return mapping, distances


def align_regime_centroids(
    previous_model: RegimeModel,
    previous_aligned_centroids: pd.DataFrame,
    current_model: RegimeModel,
) -> pd.DataFrame:
    """Align current raw KMeans labels to prior aligned labels.

    Distances are measured after expressing both original-space
    centroid sets using the previous model's feature scaler.
    """

    if (
        previous_model.feature_columns
        != current_model.feature_columns
    ):
        raise ValueError(
            "previous and current models must use "
            "identical feature columns"
        )

    feature_columns = current_model.feature_columns

    required_previous = {
        "regime",
        *feature_columns,
    }
    missing = sorted(
        required_previous.difference(
            previous_aligned_centroids.columns
        )
    )

    if missing:
        raise ValueError(
            "previous_aligned_centroids is missing "
            f"columns: {missing}"
        )

    previous = (
        previous_aligned_centroids[
            ["regime", *feature_columns]
        ]
        .sort_values("regime")
        .reset_index(drop=True)
    )

    if previous["regime"].duplicated().any():
        raise ValueError(
            "previous aligned centroids must contain "
            "one row per regime"
        )

    current = (
        regime_centroids_in_feature_space(
            current_model
        )
        .sort_values("raw_regime")
        .reset_index(drop=True)
    )

    if len(previous) != len(current):
        raise ValueError(
            "previous and current models must have "
            "the same number of regimes"
        )

    previous_scaler = (
        previous_model.pipeline.named_steps["scaler"]
    )

    scale = np.asarray(
        previous_scaler.scale_,
        dtype=float,
    )
    scale = np.where(scale == 0.0, 1.0, scale)

    centre = np.asarray(
        previous_scaler.mean_,
        dtype=float,
    )

    previous_vectors = (
        previous[feature_columns].to_numpy(dtype=float)
        - centre
    ) / scale

    current_vectors = (
        current[feature_columns].to_numpy(dtype=float)
        - centre
    ) / scale

    matched_previous_rows, distances = (
        minimum_cost_label_mapping(
            previous_vectors=previous_vectors,
            current_vectors=current_vectors,
        )
    )

    aligned_regimes = [
        int(
            previous.iloc[
                previous_row
            ]["regime"]
        )
        for previous_row in matched_previous_rows
    ]

    result = pd.DataFrame(
        {
            "raw_regime": current["raw_regime"].astype(int),
            "regime": aligned_regimes,
            "stress_score": current["stress_score"],
            "mapping_method": (
                "sequential_centroid_match"
            ),
            "match_distance": distances,
        }
    )

    return result.sort_values(
        "raw_regime"
    ).reset_index(drop=True)
