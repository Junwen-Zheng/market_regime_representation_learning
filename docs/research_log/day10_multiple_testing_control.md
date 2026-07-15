# Day 10 — Multiple-Testing Control

## Goal

Day 10 controls for the number of statistical hypotheses examined across aggregate signals and regime-conditioned signals.

Without multiplicity adjustment, repeatedly testing signals and regimes increases the probability of observing apparently interesting results by chance.

## Methods

The workflow now calculates two-sided large-sample p-values from HAC t-statistics and applies:

- Benjamini-Hochberg false-discovery-rate adjustment
- Holm family-wise-error-rate adjustment
- a significance level of 0.05
- explicit hypothesis-family names and sizes
- explicit tested and not-tested statuses

Two separate hypothesis families are used:

- aggregate signals
- eligible conditional signal-regime combinations

Rows that fail the Day 9 minimum-sample requirement remain visible but are excluded from the tested family. Their p-values and adjusted p-values remain blank.

A valid zero-eligible-hypothesis family is also supported. In that case, the workflow returns guarded output rather than failing.

## Aggregate family

The aggregate family contained five signal hypotheses.

The smallest raw p-values were:

- `momentum_60d_z`: approximately 0.4671
- `reversal_5d_z`: approximately 0.4841

The smallest Benjamini-Hochberg adjusted p-value was approximately 0.9862.

All Holm adjusted p-values were 1.0.

There were:

- zero Benjamini-Hochberg rejections
- zero Holm rejections

## Conditional family

The conditional family contained 15 eligible signal-regime hypotheses.

Five regime-3 rows were excluded because the regime had only 44 valid IC days and failed the 60-day minimum-sample safeguard.

The smallest raw conditional p-values were:

- regime 0, `reversal_5d_z`: approximately 0.2496
- regime 0, `momentum_60d_z`: approximately 0.2714
- regime 2, `reversal_5d_z`: approximately 0.3854
- regime 1, `momentum_60d_z`: approximately 0.4382

The smallest Benjamini-Hochberg adjusted p-value was approximately 0.9859.

All Holm adjusted p-values were 1.0.

There were:

- zero Benjamini-Hochberg rejections
- zero Holm rejections

## Interpretation

The tested aggregate and regime-conditioned results remain statistically weak even before multiple-testing adjustment.

Multiplicity correction reinforces the conclusion that none of the tested signals demonstrates reliable predictive performance.

The project therefore does not claim:

- statistically significant aggregate alpha
- statistically significant regime-dependent alpha
- evidence that would survive false-discovery-rate control
- evidence that would survive family-wise-error-rate control

## Research-process contribution

Day 10 improves the project by making the inferential hierarchy explicit:

1. compute descriptive rank IC
2. account for overlapping forward returns using HAC inference
3. exclude sparse regimes from formal inference
4. control for multiple hypotheses
5. report negative results transparently

## Limitations

The p-values use a large-sample normal approximation to the HAC t-statistics.

The hypothesis families are defined separately for aggregate and conditional analyses.

The analysis does not test every possible model specification, signal transformation, universe, or regime configuration.

The absence of statistical significance does not prove that all underlying effects are exactly zero. It indicates that the current evidence is insufficient.
