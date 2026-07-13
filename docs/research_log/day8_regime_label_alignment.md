# Day 8 — Walk-Forward Regime Label Alignment

## Goal

Day 8 addresses a subtle interpretability problem in walk-forward clustering.

Each walk-forward refit produces a new KMeans model. Integer cluster labels are arbitrary, so raw label zero from one fitted model is not inherently comparable with raw label zero from the next model.

Concatenating those raw labels would mix economically different states and make conditional signal results difficult to interpret.

## Alignment method

The implementation now preserves two labels:

- `raw_regime`: the original KMeans cluster number
- `regime`: the aligned walk-forward regime identity

For the first fitted model, clusters are ordered using a documented statistical-stress score based on standardised market-state features.

For every later refit:

1. cluster centroids are transformed back into original market-feature units
2. previous and current centroids are expressed using the previous model's scale
3. a one-to-one minimum-cost assignment matches current centroids to the preceding aligned centroids
4. the aligned labels are propagated into the next assignment window

The workflow exports:

- `regime_label_mappings.csv`
- `regime_aligned_centroids.csv`
- aligned and raw labels in `regime_assignments.csv`
- mean and maximum matching distance in `regime_fit_windows.csv`

## Structural validation

The real-data experiment produced:

- 92 fitted regime models
- 4 regimes per model
- 1,823 assigned dates
- 368 mapping rows
- 368 centroid rows

All mappings were one-to-one.

All assignment labels agreed with their corresponding mapping table.

All assigned dates occurred strictly after their model's fitting window.

All numeric centroid outputs were finite.

Appending future synthetic data also left historical raw and aligned assignments unchanged in the integration test.

## Effect of alignment

Raw KMeans labels were not stable across refits:

- 86 of 92 models required at least one label remapping
- 1,315 of 1,823 assigned dates had different raw and aligned label numbers

This confirms that directly concatenating raw KMeans labels would have produced misleading regime-conditioned results.

## Matching-distance diagnostics

Across models after the initial fit:

- median centroid matching distance: approximately 0.0122
- 95th percentile matching distance: approximately 0.3868
- maximum matching distance: approximately 4.0870

Most adjacent refits were close.

The largest discontinuity occurred for the model assigning dates from 2020-04-13 through 2020-05-08, during the exceptional COVID-era market-state change.

Other relatively large distances occurred during 2019 and the first half of 2020.

Matching distance is a diagnostic measure rather than a formal statistical confidence score.

## Economic coherence

Aligned regime zero was the lowest-stress state in all 92 fitted models.

Aligned regime three was the highest-stress state in all 92 fitted models.

The complete zero-to-three stress ordering held in 76 of 92 models.

Regimes one and two exchanged their relative within-model stress rank in 16 models. They should therefore be interpreted as intermediate states with less stable separation rather than permanent economic categories.

Median aligned centroid profiles showed:

- regime 0: positive returns, broad participation, low volatility, and low dispersion
- regime 1: negative returns and weak breadth with moderate volatility
- regime 2: positive returns with elevated cross-sectional and sector dispersion
- regime 3: the highest volatility and dispersion, negative trend, and poor returns

## Assigned regime counts

The aligned real-data assignments contained:

- regime 0: 915 days
- regime 1: 365 days
- regime 2: 499 days
- regime 3: 44 days

Regime three is therefore a rare extreme state.

## Conditional signal results

The best descriptive signal within each aligned regime was:

- regime 0: `momentum_60d_z`, mean rank IC approximately 0.0221
- regime 1: `momentum_60d_z`, mean rank IC approximately 0.0199
- regime 2: `reversal_5d_z`, mean rank IC approximately 0.0187
- regime 3: `reversal_5d_z`, mean rank IC approximately 0.0965

These results are descriptive rather than conclusive.

In particular, regime three has only 44 valid IC dates. Its apparently large reversal result may reflect a small number of extreme observations and should not be treated as reliable evidence without regime-specific uncertainty estimation.

Conditional IC day counts can also be below assignment counts because evaluation requires valid signal and forward-return observations.

## Interpretation

Day 8 materially improves the validity of the regime-conditioned analysis.

The aligned labels provide locally continuous walk-forward identities and stable low- and high-stress endpoints.

However:

- intermediate regimes remain partially ambiguous
- some refits have large matching distances
- regime three is sparse
- sequential alignment does not prove that regime semantics are globally stationary

The repository should describe the regimes as aligned statistical market states, not immutable economic categories.

## Next step

A later robustness stage should apply uncertainty estimates and minimum-sample safeguards to regime-conditioned IC results, particularly for rare regimes.
