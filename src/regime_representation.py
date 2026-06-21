from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeModelResult:
    assignments: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    pipeline: Pipeline


def fit_regime_model(
    market_state: pd.DataFrame,
    n_regimes: int = 4,
    pca_components: int = 3,
    random_state: int = 42,
) -> RegimeModelResult:
    """Fit PCA + KMeans regime representation model."""
    if "date" not in market_state.columns:
        raise ValueError("market_state must include date column")
    features = market_state.drop(columns=["date"])

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
    components = pipeline.named_steps["pca"].transform(
        pipeline.named_steps["scaler"].transform(features)
    )
    labels = pipeline.named_steps["kmeans"].labels_

    assignments = market_state[["date"]].copy()
    for idx in range(pca_components):
        assignments[f"pc{idx + 1}"] = components[:, idx]
    assignments["regime"] = labels

    loadings = pd.DataFrame(
        pipeline.named_steps["pca"].components_.T,
        index=features.columns,
        columns=[f"pc{i + 1}" for i in range(pca_components)],
    ).reset_index(names="feature")
    explained_variance = pd.DataFrame(
        {
            "component": [f"pc{i + 1}" for i in range(pca_components)],
            "explained_variance_ratio": pipeline.named_steps["pca"].explained_variance_ratio_,
        }
    )
    return RegimeModelResult(assignments, loadings, explained_variance, pipeline)


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
