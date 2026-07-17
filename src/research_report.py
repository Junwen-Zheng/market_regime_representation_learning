from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_OUTPUTS = {
    "research_summary": "research_summary.csv",
    "data_quality": "data_quality_report.csv",
    "regime_summary": "regime_summary.csv",
    "regime_assignments": "regime_assignments.csv",
    "regime_fit_windows": "regime_fit_windows.csv",
    "regime_label_mappings": "regime_label_mappings.csv",
    "aggregate_multiple_testing": (
        "aggregate_signal_rank_ic_hac_"
        "multiple_testing.csv"
    ),
    "conditional_multiple_testing": (
        "conditional_rank_ic_by_regime_hac_"
        "multiple_testing.csv"
    ),
}


def _load_required_outputs(
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for key, filename in REQUIRED_OUTPUTS.items():
        path = output_dir / filename

        if not path.exists():
            missing.append(str(path))
            continue

        frames[key] = pd.read_csv(path)

    if missing:
        raise FileNotFoundError(
            "Missing required report inputs:\n"
            + "\n".join(missing)
        )

    return frames


def _metric_map(
    frame: pd.DataFrame,
) -> dict[str, str]:
    if not {"metric", "value"}.issubset(
        frame.columns
    ):
        raise ValueError(
            "Metric table must contain metric and value columns"
        )

    return {
        str(metric): str(value)
        for metric, value in zip(
            frame["metric"],
            frame["value"],
        )
    }


def _integer_value(
    value: Any,
) -> int:
    return int(float(value))


def _format_value(
    value: Any,
    digits: int = 4,
) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, bool):
        return "Yes" if value else "No"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return escape(str(value))

    return f"{numeric:.{digits}f}"


def _html_table(
    frame: pd.DataFrame,
) -> str:
    return frame.to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=True,
        na_rep="",
        float_format=lambda value: f"{value:.4f}",
    )


def build_research_report(
    output_dir: str | Path,
    destination: str | Path,
) -> dict[str, int | str]:
    """Build a deterministic reviewer-facing HTML research report."""

    output_path = Path(output_dir)
    destination_path = Path(destination)

    frames = _load_required_outputs(
        output_path
    )

    research_metrics = _metric_map(
        frames["research_summary"]
    )
    quality_metrics = _metric_map(
        frames["data_quality"]
    )

    regime_summary = (
        frames["regime_summary"]
        .sort_values("regime")
        .reset_index(drop=True)
    )

    assignments = frames[
        "regime_assignments"
    ]

    fit_windows = frames[
        "regime_fit_windows"
    ]

    mappings = frames[
        "regime_label_mappings"
    ]

    aggregate = frames[
        "aggregate_multiple_testing"
    ].sort_values(
        "raw_p_value"
    ).reset_index(drop=True)

    conditional = frames[
        "conditional_multiple_testing"
    ]

    conditional_tested = (
        conditional.loc[
            conditional[
                "multiple_testing_status"
            ]
            == "tested"
        ]
        .sort_values("raw_p_value")
        .reset_index(drop=True)
    )

    conditional_ineligible = (
        conditional.loc[
            conditional[
                "multiple_testing_status"
            ]
            != "tested"
        ]
        .sort_values(
            ["regime", "signal"]
        )
        .reset_index(drop=True)
    )

    n_models = int(
        mappings["regime_model_id"].nunique()
    )

    remapped_models = int(
        mappings.loc[
            mappings["raw_regime"]
            != mappings["regime"],
            "regime_model_id",
        ].nunique()
    )

    changed_assignments = int(
        (
            assignments["raw_regime"]
            != assignments["regime"]
        ).sum()
    )

    aggregate_family_size = int(
        aggregate[
            "hypothesis_family_size"
        ].max()
    )

    conditional_family_size = int(
        conditional[
            "hypothesis_family_size"
        ].max()
    )

    aggregate_bh_rejections = int(
        aggregate["bh_fdr_reject"].sum()
    )
    aggregate_holm_rejections = int(
        aggregate["holm_reject"].sum()
    )

    conditional_bh_rejections = int(
        conditional["bh_fdr_reject"].sum()
    )
    conditional_holm_rejections = int(
        conditional["holm_reject"].sum()
    )

    median_match_distance = float(
        mappings.loc[
            mappings["regime_model_id"] > 0,
            "match_distance",
        ].median()
    )

    maximum_match_distance = float(
        mappings.loc[
            mappings["regime_model_id"] > 0,
            "match_distance",
        ].max()
    )

    aggregate_view = aggregate[
        [
            "signal",
            "mean_rank_ic",
            "hac_t_stat",
            "raw_p_value",
            "bh_fdr_adjusted_p_value",
            "holm_adjusted_p_value",
            "bh_fdr_reject",
            "holm_reject",
        ]
    ].copy()

    conditional_view = conditional_tested[
        [
            "signal",
            "regime",
            "mean_rank_ic",
            "hac_t_stat",
            "raw_p_value",
            "bh_fdr_adjusted_p_value",
            "holm_adjusted_p_value",
        ]
    ].copy()

    ineligible_view = conditional_ineligible[
        [
            "signal",
            "regime",
            "mean_rank_ic",
            "n_ic_days",
            "minimum_days",
            "inference_status",
        ]
    ].copy()

    regime_columns = [
        column
        for column in [
            "regime",
            "n_days",
            "market_return",
            "breadth",
            "realized_volatility_20d",
            "trend_60d",
            "cross_sectional_dispersion",
        ]
        if column in regime_summary.columns
    ]

    regime_view = regime_summary[
        regime_columns
    ].copy()

    experiment_rows = [
        (
            "Data mode",
            research_metrics.get(
                "data_mode",
                "unknown",
            ),
        ),
        (
            "Assets",
            research_metrics.get(
                "n_assets",
                "unknown",
            ),
        ),
        (
            "Validated days",
            research_metrics.get(
                "n_days",
                "unknown",
            ),
        ),
        (
            "Sectors",
            research_metrics.get(
                "n_sectors",
                "unknown",
            ),
        ),
        (
            "Walk-forward regimes",
            research_metrics.get(
                "n_regimes",
                "unknown",
            ),
        ),
        (
            "Minimum training days",
            research_metrics.get(
                "min_train_days",
                "unknown",
            ),
        ),
        (
            "Refit frequency",
            research_metrics.get(
                "refit_frequency",
                "unknown",
            ),
        ),
        (
            "Assigned regime days",
            research_metrics.get(
                "n_regime_assigned_days",
                "unknown",
            ),
        ),
        (
            "Raw data rows",
            quality_metrics.get(
                "raw_rows",
                "unknown",
            ),
        ),
        (
            "Duplicate date-asset rows",
            quality_metrics.get(
                "duplicate_date_asset_rows",
                "unknown",
            ),
        ),
        (
            "Missing OHLCV values",
            quality_metrics.get(
                "missing_ohlcv_values",
                "unknown",
            ),
        ),
    ]

    experiment_table = pd.DataFrame(
        experiment_rows,
        columns=["Item", "Value"],
    )

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Regime Representation Learning — Research Report</title>
<style>
body {{
    max-width: 1120px;
    margin: 0 auto;
    padding: 32px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
        sans-serif;
    line-height: 1.55;
    color: #202124;
}}
h1, h2, h3 {{
    line-height: 1.2;
}}
h1 {{
    margin-bottom: 4px;
}}
.subtitle {{
    color: #5f6368;
    margin-top: 0;
}}
.summary-box {{
    border: 1px solid #dadce0;
    border-radius: 8px;
    padding: 18px;
    margin: 20px 0;
    background: #f8f9fa;
}}
.conclusion {{
    border-left: 5px solid #5f6368;
    padding: 14px 18px;
    background: #f8f9fa;
}}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 26px;
    font-size: 0.92rem;
}}
.data-table th,
.data-table td {{
    border: 1px solid #dadce0;
    padding: 7px 9px;
    text-align: right;
}}
.data-table th:first-child,
.data-table td:first-child {{
    text-align: left;
}}
.data-table th {{
    background: #f1f3f4;
}}
code {{
    background: #f1f3f4;
    padding: 2px 5px;
    border-radius: 4px;
}}
.small {{
    color: #5f6368;
    font-size: 0.9rem;
}}
</style>
</head>
<body>
<h1>Market Regime Representation Learning</h1>
<p class="subtitle">
Reviewer-facing real-data research report
</p>

<div class="summary-box">
<strong>Executive conclusion:</strong>
The current evidence does not support statistically reliable aggregate
or regime-dependent alpha. The strongest contribution is the research
process: explicit real-data execution, walk-forward regime fitting,
cross-refit label alignment, autocorrelation-aware inference,
sparse-sample safeguards, and multiple-testing control.
</div>

<h2>1. Research question</h2>
<p>
Can simple cross-sectional momentum, reversal, and
liquidity-adjusted signals exhibit economically meaningful and
statistically defensible performance conditional on market regimes?
</p>

<h2>2. Experiment configuration</h2>
{_html_table(experiment_table)}

<h2>3. Leakage and reproducibility controls</h2>
<ul>
<li>Regime estimators are fit only on dates preceding assignment dates.</li>
<li>The default workflow uses walk-forward rather than full-sample regimes.</li>
<li>Raw KMeans labels are preserved separately from aligned labels.</li>
<li>No silent synthetic fallback is used in real-data mode.</li>
<li>The raw downloaded price panel is excluded from Git.</li>
<li>Committed reports are generated from a fixed 24-asset universe.</li>
</ul>

<h2>4. Regime-label alignment</h2>
<p>
The real-data workflow fit <strong>{n_models}</strong> regime models.
Raw KMeans labels required remapping in
<strong>{remapped_models}</strong> models, and alignment changed the
numeric label of <strong>{changed_assignments}</strong> assigned dates.
</p>
<p>
The median sequential centroid-match distance was
<strong>{_format_value(median_match_distance)}</strong>, while the
maximum was <strong>{_format_value(maximum_match_distance)}</strong>.
The largest discontinuity occurred around the 2020 market shock.
</p>

<h3>Aligned regime profiles</h3>
{_html_table(regime_view)}

<h2>5. Aggregate signal inference</h2>
<p>
HAC inference accounts for overlap in the 10-day forward-return target.
The aggregate family contained
<strong>{aggregate_family_size}</strong> hypotheses.
</p>
{_html_table(aggregate_view)}

<h2>6. Regime-conditional inference</h2>
<p>
Formal conditional inference required at least 60 valid IC days.
The tested conditional family contained
<strong>{conditional_family_size}</strong> eligible hypotheses.
</p>
{_html_table(conditional_view)}

<h3>Sparse results excluded from formal testing</h3>
{_html_table(ineligible_view)}

<h2>7. Multiple-testing conclusion</h2>
<ul>
<li>
Aggregate Benjamini-Hochberg rejections:
<strong>{aggregate_bh_rejections}</strong>
</li>
<li>
Aggregate Holm rejections:
<strong>{aggregate_holm_rejections}</strong>
</li>
<li>
Conditional Benjamini-Hochberg rejections:
<strong>{conditional_bh_rejections}</strong>
</li>
<li>
Conditional Holm rejections:
<strong>{conditional_holm_rejections}</strong>
</li>
</ul>

<div class="conclusion">
<strong>Final interpretation:</strong>
No tested signal demonstrates statistically supported aggregate alpha
or regime-dependent alpha. Descriptive patterns, particularly in the
rare highest-stress regime, remain exploratory and should not be
presented as tradable evidence.
</div>

<h2>8. Main limitations</h2>
<ul>
<li>The universe contains 24 large US equities across eight sectors.</li>
<li>The analysis uses one fixed regime specification.</li>
<li>Intermediate aligned regimes remain partially ambiguous.</li>
<li>The highest-stress regime has a small sample.</li>
<li>HAC p-values use a large-sample normal approximation.</li>
<li>Transaction costs, turnover, and portfolio implementation are not evaluated.</li>
</ul>

<h2>9. Reproducibility artifacts</h2>
<p>
The report is generated from committed CSV outputs under
<code>outputs/real/</code>. Detailed stage logs are stored under
<code>docs/research_log/</code>.
</p>

<p class="small">
This report intentionally distinguishes methodological validity from
evidence of profitable alpha.
</p>
</body>
</html>
"""

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_path.write_text(
        report_html,
        encoding="utf-8",
    )

    return {
        "report_path": str(destination_path),
        "regime_models": n_models,
        "remapped_models": remapped_models,
        "changed_assignments": changed_assignments,
        "aggregate_hypotheses": (
            aggregate_family_size
        ),
        "conditional_hypotheses": (
            conditional_family_size
        ),
        "total_rejections": (
            aggregate_bh_rejections
            + aggregate_holm_rejections
            + conditional_bh_rejections
            + conditional_holm_rejections
        ),
    }
