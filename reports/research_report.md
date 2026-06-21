# Market Regime Representation Learning for Conditional Equity Alpha Robustness

## Executive Summary

This project studies whether cross-sectional equity alpha signals behave differently across learned market regimes. The central research idea is that average signal performance is incomplete: a signal may look acceptable in aggregate while failing in high-volatility, low-liquidity, or trend-reversal regimes.

The pipeline learns regimes from market-state features using PCA and KMeans, then evaluates signal rank IC by regime. The result is a reproducible research framework for conditional alpha robustness rather than a claim of production-ready tradable alpha.

## Hypothesis

Simple momentum and reversal signals should not be treated as uniformly stable. Their predictive behavior should depend on market state. For example:

- momentum may behave better in calm trend regimes;
- short-term reversal may improve during stressed or overextended regimes;
- volatility-adjusted signals may be more stable than raw signals in high-dispersion periods;
- liquidity-adjusted signals may degrade when liquidity stress rises.

## Data Design

The repository currently uses a reproducible synthetic equity panel with finance-shaped return dynamics. This is intentional: the project can be run by any reviewer without paid data access. The synthetic data generator includes hidden regimes, sector effects, market shocks, volatility differences, and weak conditional signal behavior.

This design proves pipeline discipline, not live-market alpha.

## Market-State Features

The regime model uses daily features such as:

- market return
- cross-sectional return dispersion
- return breadth
- realised volatility
- 60-day trend
- liquidity stress proxy
- sector dispersion
- sector max-minus-min return spread
- market-move-to-dispersion ratio

These features are standardised and compressed with PCA before clustering.

## Regime Learning

The project uses PCA to build a lower-dimensional market-state representation, followed by KMeans clustering. The goal is interpretability and reproducibility rather than model complexity. The outputs include:

- regime assignments by date
- PCA loadings
- explained variance ratios
- regime transition matrix
- regime summary statistics

## Alpha Signals

The project evaluates five cross-sectional signal families:

- 5-day reversal
- 20-day momentum
- 60-day momentum
- 20-day volatility-adjusted momentum
- liquidity-adjusted momentum

Signals are z-scored cross-sectionally by date and evaluated against 10-day forward relative returns.

## Evaluation Method

The main evaluation metric is daily cross-sectional Spearman rank IC. The project reports:

- aggregate rank IC by signal
- rank IC by learned regime
- IC information ratio
- regime transition probabilities
- regime-specific market-state summaries

This design tests whether a signal is conditionally useful rather than merely positive on average.

## Research Interpretation

A strong result would not be a single high average IC. A stronger result would be:

1. consistent signal behavior inside clearly interpretable regimes;
2. sensible degradation in regimes where the signal should not work;
3. robustness to horizon, universe, and cost assumptions;
4. no leakage through forward-looking features;
5. stable results after replacing synthetic data with real survivorship-bias-controlled equity data.

## Current Limitations

This project is not evidence of tradable alpha. The current dataset is synthetic, and therefore cannot validate real market performance. A production-grade version would require:

- survivorship-bias-free equity universe;
- point-in-time corporate actions;
- robust delisting handling;
- borrow and financing assumptions;
- real transaction-cost modelling;
- more rigorous train/test separation;
- stability checks across multiple historical periods.

## Next Research Steps

The next meaningful extension would be to replace the synthetic panel with a real public-data or vendor-data panel and run the same regime-conditioned evaluation. A second extension would be to add graph-based peer-relative equity features and test whether regime conditioning improves or weakens those signals.
