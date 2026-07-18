from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_direct_dependency_lock_is_fully_pinned():
    path = REPOSITORY_ROOT / "requirements-lock.txt"
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    expected_packages = {
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pytest",
    }

    package_names = {
        line.split("==", maxsplit=1)[0]
        for line in lines
    }

    assert package_names == expected_packages
    assert all(line.count("==") == 1 for line in lines)


def test_ci_contract_covers_supported_versions_and_report():
    workflow = (
        REPOSITORY_ROOT
        / ".github"
        / "workflows"
        / "ci.yml"
    ).read_text()

    required_fragments = [
        'python-version:',
        '"3.9"',
        '"3.11"',
        "requirements-lock.txt",
        "pytest -q",
        "python scripts/build_research_report.py",
        "git diff --exit-code",
        "docs/reports/market_regime_research_report.html",
    ]

    for fragment in required_fragments:
        assert fragment in workflow


def test_reproducibility_guide_has_no_machine_paths():
    guide = (
        REPOSITORY_ROOT
        / "docs"
        / "REPRODUCIBILITY.md"
    ).read_text()

    required_fragments = [
        "Clean-clone verification",
        "Synthetic smoke test",
        "Real-data workflow",
        "Continuous integration",
        "requirements-lock.txt",
        "pytest -q",
        "scripts/build_research_report.py",
    ]

    for fragment in required_fragments:
        assert fragment in guide

    forbidden_fragments = [
        "/Users/",
        "junwenzheng",
        "file://",
    ]

    for fragment in forbidden_fragments:
        assert fragment not in guide
