from __future__ import annotations

from pathlib import Path

import pytest

from src.research_report import build_research_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REAL_OUTPUTS = REPOSITORY_ROOT / "outputs" / "real"


def test_build_research_report_from_committed_outputs(
    tmp_path,
):
    destination = tmp_path / "research_report.html"

    summary = build_research_report(
        output_dir=REAL_OUTPUTS,
        destination=destination,
    )

    assert destination.exists()

    assert summary["regime_models"] == 92
    assert summary["remapped_models"] == 86
    assert summary["changed_assignments"] == 1315
    assert summary["aggregate_hypotheses"] == 5
    assert summary["conditional_hypotheses"] == 15
    assert summary["total_rejections"] == 0

    html = destination.read_text(
        encoding="utf-8",
    )

    expected_fragments = [
        "Market Regime Representation Learning",
        "Executive conclusion",
        "Leakage and reproducibility controls",
        "Regime-label alignment",
        "Aggregate signal inference",
        "Regime-conditional inference",
        "Multiple-testing conclusion",
        "No tested signal demonstrates statistically supported",
    ]

    for fragment in expected_fragments:
        assert fragment in html


def test_report_generation_is_deterministic(
    tmp_path,
):
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"

    build_research_report(
        output_dir=REAL_OUTPUTS,
        destination=first,
    )

    build_research_report(
        output_dir=REAL_OUTPUTS,
        destination=second,
    )

    assert first.read_bytes() == second.read_bytes()


def test_missing_required_output_is_rejected(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Missing required report inputs",
    ):
        build_research_report(
            output_dir=tmp_path,
            destination=tmp_path / "report.html",
        )
