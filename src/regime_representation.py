from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeModel:
    """Fitted PCA + KMeans regime model without assigned regime labels."""

    pipeline: Pipeline
    feature_columns: list[str]
    n_regimes: int
    pca_components: int
    random_state: int
    fit_start_date: pd.Timestamp
    fit_end_date: pd.Timestamp


@dataclass
class RegimeModelResult:
    """Full-sample diagnostic result kept for backward compatibility."""

    assignments: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    model: RegimeModel

    @property
    def pipeline(self) -> Pipeline:
        """Backward-compatible access to the fitted sklearn pipeline."""
        return self.model.pipeline


def _feature_columns(market_state: pd.DataFrame) -> list[str]:
    if "date" not in market_state.columns:
        raise ValueError("market_state must include date column")

    feature_cols = [col for col in market_state.columns if col != "date"]

    if not feature_cols:
        raise ValueError("market_state must include at least one feature column")

    return feature_cols


def _validate_market_state_for_model(
    market_state: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    if "date" not in market_state.columns:
        raise ValueError("market_state must include date column")

    missing = sorted(set(feature_columns) - set(market_state.columns))
    if missing:
        raise ValueError(f"market_state is missing model feature columns: {missing}")


def fit_regime_estimator(
    market_state: pd.DataFrame,
    n_regimes: int = 4,
    pca_components: int = 3,
    random_state: int = 42,
) -> RegimeModel:
    """Fit PCA + KMeans using only the provided market-state rows.

    This function intentionally does not assign labels. Splitting fit from
    assignment is the first step toward walk-forward, non-leaky regime labels.
    """

    feature_cols = _feature_columns(market_state)
    features = market_state[feature_cols]

    pca = PCA(n_components=pca_components, random_state=random_state)
    kmeans = KMeans(n_clusters=n_regimes, n_init=25, random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", pca),
            ("kmeans", kmeans),
        ]
    )

    pipeline.fit(features)

    ordered_dates = pd.to_datetime(market_state["date"])

    return RegimeModel(
        pipeline=pipeline,
        feature_columns=feature_cols,
        n_regimes=n_regimes,
        pca_components=pca_components,
        random_state=random_state,
        fit_start_date=ordered_dates.min(),
        fit_end_date=ordered_dates.max(),
    )


def assign_regimes(
    model: RegimeModel,
    market_state: pd.DataFrame,
) -> pd.DataFrame:
    """Assign regime labels using an already-fitted regime model."""

    _validate_market_state_for_model(market_state, model.feature_columns)

    features = market_state[model.feature_columns]
    scaler = model.pipeline.named_steps["scaler"]
    pca = model.pipeline.named_steps["pca"]
    kmeans = model.pipeline.named_steps["kmeans"]

    scaled = scaler.transform(features)
    components = pca.transform(scaled)
    labels = kmeans.predict(components)

    assignments = market_state[["date"]].copy()

    for idx in range(model.pca_components):
        assignments[f"pc{idx + 1}"] = components[:, idx]

    assignments["regime"] = labels
    assignments["model_fit_start_date"] = model.fit_start_date
    assignments["model_fit_end_date"] = model.fit_end_date

    return assignments


def regime_model_loadings(model: RegimeModel) -> pd.DataFrame:
    """Return PCA loadings for a fitted regime model."""

    pca = model.pipeline.named_steps["pca"]

    return pd.DataFrame(
        pca.components_.T,
        index=model.feature_columns,
        columns=[f"pc{i + 1}" for i in range(model.pca_components)],
    ).reset_index(names="feature")


def regime_model_explained_variance(model: RegimeModel) -> pd.DataFrame:
    """Return explained variance ratios for a fitted regime model."""

    pca = model.pipeline.named_steps["pca"]

    return pd.DataFrame(
        {
            "component": [f"pc{i + 1}" for i in range(model.pca_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )


def fit_regime_model(
    market_state: pd.DataFrame,
    n_regimes: int = 4,
    pca_components: int = 3,
    random_state: int = 42,
) -> RegimeModelResult:
    """Fit and assign regimes on the same sample.

    This function is kept for compatibility and for full-sample diagnostics.
    It should not be used as evidence of out-of-sample regime performance.
    Day 3 will add a walk-forward assignment path for the main research workflow.
    """

    model = fit_regime_estimator(
        market_state=market_state,
        n_regimes=n_regimes,
        pca_components=pca_components,
        random_state=random_state,
    )

    assignments = assign_regimes(model, market_state)
    loadings = regime_model_loadings(model)
    explained_variance = regime_model_explained_variance(model)

    return RegimeModelResult(
        assignments=assignments,
        loadings=loadings,
        explained_variance=explained_variance,
        model=model,
    )


def fit_predict_full_sample_regimes(
    market_state: pd.DataFrame,
    n_regimes: int = 4,
    pca_components: int = 3,
    random_state: int = 42,
) -> RegimeModelResult:
    """Explicit alias for the full-sample diagnostic regime workflow."""

    return fit_regime_model(
        market_state=market_state,
        n_regimes=n_regimes,
        pca_components=pca_components,
        random_state=random_state,
    )


def regime_transition_matrix(assignments: pd.DataFrame) -> pd.DataFrame:
    """Estimate empirical transition probabilities among learned regimes."""

    ordered = assignments.sort_values("date").copy()
    ordered["next_regime"] = ordered["regime"].shift(-1)
    transitions = ordered.dropna(subset=["next_regime"])

    counts = pd.crosstab(transitions["regime"], transitions["next_regime"])
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    probs.index.name = "from_regime"
    probs.columns.name = "to_regime"

    return probs.reset_index()
