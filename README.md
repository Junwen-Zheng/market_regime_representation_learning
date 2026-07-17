# Market Regime Representation Learning

This repository studies whether simple cross-sectional equity signals behave differently across learned market regimes.

It is a research case study, not a production trading strategy, and it does not claim tradable alpha.

## Current research design

The primary workflow uses a real multi-asset OHLCV CSV.

Synthetic data is available only through the explicitly selected `synthetic_smoke_test` mode. Missing or invalid real data never triggers a synthetic fallback.

Regimes are learned using market-state features, standardisation, PCA, KMeans, and walk-forward fitting. Conditional signal rank IC uses walk-forward regime labels by default.

## Setup

Run:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Real-data workflow

Place a valid CSV at:

    data/raw/prices.csv

Required fields:

    date,asset,sector,open,high,low,close,volume

Run:

    python scripts/run_research.py --data-mode real --data-path data/raw/prices.csv

There is no synthetic fallback if the file is missing or invalid.

## Synthetic smoke test

Run:

    python scripts/run_research.py --data-mode synthetic_smoke_test --output-dir /tmp/market_regime_smoke

Synthetic results demonstrate pipeline execution only.

## Tests

Run:

    pytest -q

## Main modules

- `src/data_loader.py`: explicit real and synthetic data selection
- `src/market_state.py`: market-state feature construction
- `src/regime_representation.py`: regime fitting and assignment
- `src/walk_forward_regimes.py`: walk-forward regime labels
- `src/alpha_signals.py`: cross-sectional signals
- `src/evaluation.py`: aggregate and conditional rank IC
- `src/workflow.py`: end-to-end workflow
- `scripts/run_research.py`: command-line runner

## Important limitations

- The planned public-data universe contains current survivors.
- Delisted securities and delisting returns are not included.
- Public OHLCV data is not point-in-time institutional data.
- Transaction costs, borrow constraints, and execution effects are not modelled.
- Statistical regime labels are not stable economic truths.

## Robustness findings

Because the target is a 10-day forward return, daily rank-IC observations overlap.

The real-data workflow therefore includes Newey-West HAC uncertainty estimates, non-overlapping 10-day offsets, and calendar-year stability checks.

The strongest signal, 60-day momentum, had:

- mean rank IC of approximately 0.0123
- HAC t-statistic of approximately 0.73
- a 95 percent HAC confidence interval that included zero
- positive mean IC in 9 of 10 non-overlapping offsets
- positive mean IC in 6 of 9 calendar years

These results are directionally interesting but statistically weak. The repository does not claim reliable or tradable alpha.

## Walk-forward regime-label alignment

KMeans cluster numbers are arbitrary across separate model fits. The walk-forward workflow therefore preserves each model's `raw_regime` and creates an aligned `regime` through sequential minimum-cost centroid matching.

In the real-data experiment:

- 86 of 92 fitted models required label remapping
- 1,315 of 1,823 assigned dates changed label number
- regimes 0 and 3 remained the lowest- and highest-stress states across all refits
- regimes 1 and 2 exchanged relative stress rank in 16 models
- the largest centroid discontinuity occurred around the 2020 COVID shock

The aligned labels support more defensible conditional analysis, but they remain statistical states rather than permanent economic categories. The rare highest-stress regime contained only 44 evaluation days, so its conditional signal results are treated as descriptive.

## Regime-conditional inference safeguards

The workflow applies position-aware Newey-West HAC inference to regime-conditioned rank IC. Lagged covariance is based on original market-day spacing rather than adjacency after filtering to a regime.

Formal inference requires at least 60 valid IC days. In the real-data experiment, regimes 0 through 2 were eligible, while the highest-stress regime had only 44 days and was marked `insufficient_sample`.

None of the 15 eligible signal-regime confidence intervals excluded zero. Conditional results are therefore reported as weak or descriptive rather than evidence of reliable regime-dependent alpha.

## Multiple-testing control

The workflow converts HAC t-statistics into two-sided large-sample p-values and applies both Benjamini-Hochberg false-discovery-rate adjustment and Holm family-wise-error adjustment.

The real-data analysis tested:

- 5 aggregate signal hypotheses
- 15 eligible signal-regime hypotheses

Five sparse regime-3 rows were excluded from formal testing under the 60-day minimum-sample safeguard.

Neither family produced any Benjamini-Hochberg or Holm rejection at the 0.05 level. The results therefore provide no statistically supported aggregate or regime-dependent alpha evidence.

## Reviewer-facing research report

The consolidated real-data report is available at:

[`docs/reports/market_regime_research_report.html`](docs/reports/market_regime_research_report.html)

It summarizes:

- real-data coverage and quality checks
- walk-forward leakage controls
- regime-label alignment across refits
- aggregate and regime-conditional HAC inference
- sparse-regime safeguards
- Benjamini-Hochberg and Holm multiple-testing control
- limitations and the final negative-alpha conclusion

Regenerate it deterministically with:

    python scripts/build_research_report.py
