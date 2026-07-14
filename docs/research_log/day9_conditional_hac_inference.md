# Day 9 — Regime-Conditional HAC Inference

## Goal

Day 9 adds uncertainty estimates and minimum-sample safeguards to the regime-conditioned rank-IC analysis.

The prediction target is a 10-day forward relative return. Daily IC observations overlap, so conventional independent-observation standard errors would understate uncertainty.

Regime filtering creates another complication: two observations from the same regime may be separated by many market days. Treating them as adjacent after filtering would introduce incorrect autocovariance relationships.

## Method

The conditional robustness implementation adds:

- position-aware Newey-West HAC standard errors
- maximum lag of 9 market days
- HAC t-statistics
- 95 percent confidence intervals
- assigned-day and valid-IC-day counts
- IC-day coverage
- a minimum inference threshold of 60 valid IC days
- explicit `eligible` and `insufficient_sample` statuses

The position-aware HAC estimator uses each observation's original market-day position. It includes a lagged covariance term only when two observations are separated by the corresponding number of actual market days.

## Real-data sample status

The aligned regimes produced:

- regime 0: 915 valid IC days from 915 assigned days
- regime 1: 364 valid IC days from 365 assigned days
- regime 2: 490 valid IC days from 499 assigned days
- regime 3: 44 valid IC days from 44 assigned days

Regimes 0, 1, and 2 passed the 60-day minimum.

Regime 3 remained visible descriptively but was marked `insufficient_sample`. Its HAC standard errors, t-statistics, and confidence intervals were intentionally left blank.

## Eligible conditional results

There were 15 eligible signal-regime combinations.

No eligible 95 percent HAC confidence interval excluded zero.

The largest absolute HAC t-statistics were:

- regime 0, `reversal_5d_z`: approximately -1.15
- regime 0, `momentum_60d_z`: approximately 1.10
- regime 2, `reversal_5d_z`: approximately 0.87
- regime 1, `momentum_60d_z`: approximately 0.78

These values do not provide strong evidence of regime-dependent predictive performance.

## Sparse high-stress regime

Regime 3 previously showed a descriptive reversal mean rank IC of approximately 0.0965.

However, that estimate is based on only 44 valid IC days and therefore fails the minimum-sample safeguard.

It should not be interpreted as statistically supported evidence.

## Interpretation

The regime-conditioned results remain weak after autocorrelation-aware inference.

The project therefore does not claim that any tested signal has reliable conditional alpha within the aligned regimes.

Day 9 improves the research process by:

- preserving sparse-regime descriptive results
- suppressing unsupported inferential statistics
- respecting actual market-day spacing
- making sample eligibility explicit
- preventing rare-state estimates from being overstated

## Limitations

The 60-day threshold is a transparent research safeguard rather than a mathematically optimal cutoff.

The analysis does not adjust for multiple testing across signals and regimes.

Regime identity is aligned across refits, but intermediate regimes remain partially ambiguous.

The results are based on one fixed public-data universe and one signal specification set.
