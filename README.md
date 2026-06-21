# Market Regime Representation Learning for Conditional Equity Alpha Robustness

This repository is a reproducible quant research case study focused on **learning market-regime representations** and testing whether equity alpha signals remain robust across different market states.

The project is designed as a second, standalone financial research project for a systematic investing / quant research portfolio. It is not a production trading strategy and does not claim tradable alpha. The goal is to show the full research workflow: hypothesis design, data construction, regime representation, conditional signal evaluation, failure analysis, and reproducible outputs.

## Research Question

Do simple cross-sectional equity alpha signals behave differently across learned market regimes?

More specifically:

1. Can market-level features such as volatility, liquidity stress, cross-sectional dispersion, trend, breadth, and sector dispersion be compressed into a lower-dimensional regime representation?
2. Do momentum/reversal/volatility-adjusted alpha signals show different rank-IC behavior across those regimes?
3. Are any apparent results robust enough to justify further research, or are they mainly artifacts of regime sampling, public-data limitations, and transaction-cost assumptions?

## Why This Project Exists

Most toy quant projects stop at a simple backtest. This project focuses on a more research-oriented question: **conditional robustness**. A signal that appears useful on average may fail under certain volatility, liquidity, or trend regimes.

The project therefore evaluates alpha signals by regime rather than only reporting aggregate metrics.

## Project Structure

```text
config/                     Experiment configuration
src/                        Research code
  synthetic_data.py          Reproducible panel-data generator
  market_state.py            Market-state feature construction
  regime_representation.py   Standardisation, PCA, KMeans regime learning
  alpha_signals.py           Cross-sectional alpha signal construction
  evaluation.py              Rank IC, regime slicing, transition diagnostics
  workflow.py                End-to-end pipeline
scripts/
  run_research.py            Main runnable script
reports/
  research_report.md         Research write-up
  failure_analysis.md        Honest limitations and failure modes
docs/research_log/           Timestamped research notes
outputs/                     Generated CSV outputs
tests/                       Unit tests
resume_snippet.md            Resume-ready project bullet options
```

## Methodology

The project builds a daily stock panel with synthetic but finance-shaped data. The synthetic dataset is used so the repository is reproducible without paid market data or private vendor APIs. The pipeline is structured so real OHLCV data can be swapped in later.

The workflow is:

1. Generate or load a multi-asset equity panel.
2. Construct market-state features:
   - market return
   - realised volatility
   - cross-sectional return dispersion
   - return breadth
   - dollar-volume liquidity stress
   - sector return dispersion
   - trend strength
3. Learn market-regime representations:
   - standardise market-state features
   - reduce dimension with PCA
   - cluster market states with KMeans
4. Construct alpha signals:
   - 5-day reversal
   - 20-day momentum
   - 60-day momentum
   - volatility-adjusted momentum
   - liquidity-adjusted momentum
5. Evaluate forward relative returns using daily cross-sectional rank IC.
6. Slice alpha performance by learned regime.
7. Generate transition matrix and regime diagnostics.
8. Produce reproducible CSV outputs and research notes.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_research.py
pytest -q
```

Expected generated outputs:

```text
outputs/market_state_features.csv
outputs/regime_assignments.csv
outputs/regime_transition_matrix.csv
outputs/conditional_rank_ic_by_regime.csv
outputs/aggregate_signal_rank_ic.csv
outputs/research_summary.csv
```

## What This Demonstrates

- Market-state feature engineering
- Representation learning for regime discovery
- PCA + clustering workflow for interpretable regime modelling
- Cross-sectional alpha-signal construction
- Rank-IC based signal evaluation
- Conditional alpha robustness analysis
- Transition-matrix diagnostics
- Reproducible synthetic-data research design
- Honest limitation reporting

## Current Status

This is a research case study, not a trading system. The current version uses a reproducible synthetic dataset to demonstrate methodology and evaluation discipline. A production-grade version would require survivorship-bias-free equity data, point-in-time fundamentals/events, corporate-action handling, borrow/cost assumptions, and stricter execution modelling.

## Main Limitation

The project is useful as evidence of research process, not as evidence of tradable alpha. Synthetic data can test whether the pipeline is logically coherent, but not whether a signal works in live markets.
