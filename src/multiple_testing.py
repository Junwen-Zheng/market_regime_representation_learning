from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def two_sided_normal_p_values(
    statistics: np.ndarray | pd.Series,
) -> np.ndarray:
    """Convert finite large-sample test statistics to two-sided p-values."""

    values = np.asarray(statistics, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "statistics must contain only finite values"
        )

    return np.asarray(
        2.0 * norm.sf(np.abs(values)),
        dtype=float,
    )


def benjamini_hochberg_adjust(
    p_values: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values."""

    values = _validate_p_values(p_values)
    n_hypotheses = len(values)

    order = np.argsort(values)
    ordered = values[order]

    ranks = np.arange(
        1,
        n_hypotheses + 1,
        dtype=float,
    )

    adjusted_ordered = (
        ordered
        * n_hypotheses
        / ranks
    )

    adjusted_ordered = np.minimum.accumulate(
        adjusted_ordered[::-1]
    )[::-1]

    adjusted_ordered = np.clip(
        adjusted_ordered,
        0.0,
        1.0,
    )

    adjusted = np.empty(
        n_hypotheses,
        dtype=float,
    )
    adjusted[order] = adjusted_ordered

    return adjusted


def holm_adjust(
    p_values: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values."""

    values = _validate_p_values(p_values)
    n_hypotheses = len(values)

    order = np.argsort(values)
    ordered = values[order]

    multipliers = np.arange(
        n_hypotheses,
        0,
        -1,
        dtype=float,
    )

    adjusted_ordered = ordered * multipliers

    adjusted_ordered = np.maximum.accumulate(
        adjusted_ordered
    )

    adjusted_ordered = np.clip(
        adjusted_ordered,
        0.0,
        1.0,
    )

    adjusted = np.empty(
        n_hypotheses,
        dtype=float,
    )
    adjusted[order] = adjusted_ordered

    return adjusted


def apply_multiple_testing_corrections(
    results: pd.DataFrame,
    statistic_col: str = "hac_t_stat",
    eligibility_col: str | None = None,
    alpha: float = 0.05,
    family_name: str = "unspecified",
) -> pd.DataFrame:
    """Add raw and multiplicity-adjusted p-values to an inference table.

    Only eligible rows belong to the tested hypothesis family.
    Ineligible rows remain visible but receive blank p-values and
    rejection flags set to False.
    """

    if statistic_col not in results.columns:
        raise ValueError(
            f"results is missing column: {statistic_col}"
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must be strictly between zero and one"
        )

    output = results.copy()

    if eligibility_col is None:
        eligible = pd.Series(
            True,
            index=output.index,
            dtype=bool,
        )
    else:
        if eligibility_col not in output.columns:
            raise ValueError(
                f"results is missing column: {eligibility_col}"
            )

        eligible = output[
            eligibility_col
        ].astype(bool)

    eligible_statistics = output.loc[
        eligible,
        statistic_col,
    ].to_numpy(dtype=float)

    output["raw_p_value"] = np.nan
    output["bh_fdr_adjusted_p_value"] = np.nan
    output["holm_adjusted_p_value"] = np.nan

    output["bh_fdr_reject"] = False
    output["holm_reject"] = False

    if len(eligible_statistics) > 0:
        if not np.isfinite(
            eligible_statistics
        ).all():
            raise ValueError(
                "eligible test statistics must be finite"
            )

        raw_p_values = two_sided_normal_p_values(
            eligible_statistics
        )

        bh_adjusted = benjamini_hochberg_adjust(
            raw_p_values
        )

        holm_adjusted = holm_adjust(
            raw_p_values
        )

        output.loc[
            eligible,
            "raw_p_value",
        ] = raw_p_values

        output.loc[
            eligible,
            "bh_fdr_adjusted_p_value",
        ] = bh_adjusted

        output.loc[
            eligible,
            "holm_adjusted_p_value",
        ] = holm_adjusted

        output.loc[
            eligible,
            "bh_fdr_reject",
        ] = bh_adjusted <= alpha

        output.loc[
            eligible,
            "holm_reject",
        ] = holm_adjusted <= alpha

    output["multiple_testing_family"] = family_name
    output["hypothesis_family_size"] = int(
        eligible.sum()
    )
    output["multiple_testing_alpha"] = alpha

    output["multiple_testing_status"] = np.where(
        eligible,
        "tested",
        "not_tested_ineligible",
    )

    return output


def _validate_p_values(
    p_values: np.ndarray | pd.Series,
) -> np.ndarray:
    values = np.asarray(
        p_values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "p_values must be one-dimensional"
        )

    if len(values) == 0:
        raise ValueError(
            "p_values must not be empty"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "p_values must contain only finite values"
        )

    if (
        (values < 0.0).any()
        or (values > 1.0).any()
    ):
        raise ValueError(
            "p_values must lie between zero and one"
        )

    return values
