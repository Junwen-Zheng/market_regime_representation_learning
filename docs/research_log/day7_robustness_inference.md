# Day 7 — Robustness and Autocorrelation-Aware Inference

## Goal

Day 7 tests whether the weak aggregate real-data rank-IC results remain persuasive after accounting for overlapping forward-return horizons and calendar instability.

The prediction target is a 10-day forward relative return. Daily IC observations therefore overlap, so treating them as independent would understate uncertainty.

## Methods

The robustness layer adds:

- Newey-West HAC standard errors
- maximum HAC lag of 9
- HAC t-statistics
- 95 percent confidence intervals
- ten non-overlapping 10-day offset samples
- calendar-year IC summaries

## HAC results

The strongest aggregate signal remained `momentum_60d_z`:

- mean rank IC: 0.012318
- HAC standard error: 0.016939
- HAC t-statistic: 0.727215
- 95 percent confidence interval: -0.020882 to 0.045519
- evaluated days: 2,064

All five signals had 95 percent HAC confidence intervals that included zero.

No signal had an absolute HAC t-statistic above 1.

## Non-overlapping offsets

`momentum_60d_z` was positive in 9 of 10 offsets.

Its offset mean IC ranged from approximately -0.0012 to 0.0324.

`reversal_5d_z` and `liquidity_adjusted_momentum_z` were each positive in 8 of 10 offsets, but their offset ranges were wider and included negative values.

The offset results show some directional consistency, but not enough magnitude or stability to establish strong evidence.

## Calendar-year stability

`momentum_60d_z` was positive in 6 of 9 calendar years.

Its yearly mean IC ranged from approximately -0.0317 to 0.0481.

`reversal_5d_z` was also positive in 6 of 9 years, with a wider range from approximately -0.0649 to 0.0549.

The remaining signals were positive in only 4 or 5 of 9 years.

## Interpretation

The real-data results remain weak and noisy after more defensible inference.

The positive aggregate mean IC for 60-day momentum is not statistically distinguishable from zero under the HAC estimate.

The project should therefore not claim a reliable or tradable alpha effect.

The main contribution remains methodological:

- real-data execution
- explicit data-quality auditing
- walk-forward regime assignment
- no silent synthetic fallback
- autocorrelation-aware inference
- transparent reporting of negative and inconclusive evidence

## Next step

The next stage should examine regime-label interpretability and stability across model refits.

Numeric KMeans labels should not automatically be interpreted as consistent economic states across separate fitting windows.
