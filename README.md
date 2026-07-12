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
