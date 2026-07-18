# Reproducibility Guide

## Supported environments

GitHub Actions verifies the repository on:

- Python 3.9
- Python 3.11

`requirements.txt` contains compatible development ranges.

`requirements-lock.txt` pins the direct dependency versions used by CI and committed-report verification. It is not a complete transitive lock file.

## Clean-clone verification

From a clean checkout:

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements-lock.txt
    pytest -q
    python scripts/build_research_report.py
    git diff --exit-code -- docs/reports/market_regime_research_report.html

The final `git diff` command should produce no output.

## Synthetic smoke test

The synthetic path verifies the complete workflow without downloading external data:

    python scripts/run_research.py \
      --data-mode synthetic_smoke_test \
      --output-dir outputs/synthetic_smoke_test

Synthetic data is used only for smoke testing and automated verification. It is not presented as empirical market evidence.

## Real-data workflow

The committed real-data outputs were generated from a fixed 24-asset US equity panel.

The optional downloader requires Python 3.11 or newer:

    python3.11 -m venv .venv-download
    source .venv-download/bin/activate
    python -m pip install -r requirements-data-download.txt
    python scripts/download_yahoo_data.py \
      --start 2018-01-01 \
      --end 2026-07-02

The downloaded file is written to:

    data/raw/prices.csv

That raw file is excluded from Git.

Run the real-data research pipeline with:

    python scripts/run_research.py \
      --data-mode real \
      --data-path data/raw/prices.csv \
      --output-dir outputs/real \
      --n-regimes 4 \
      --pca-components 3 \
      --min-train-days 252 \
      --refit-frequency 20

Then regenerate the reviewer report:

    python scripts/build_research_report.py

## Continuous integration

The workflow in `.github/workflows/ci.yml` performs:

1. dependency installation from `requirements-lock.txt`
2. the complete pytest suite
3. reviewer-report regeneration
4. a Git diff check proving the committed HTML report is deterministic

The real market-data file is not required in CI because the tests and report builder operate on committed reproducibility artifacts.
