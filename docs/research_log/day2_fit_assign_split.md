# Day 2 — Split Regime Fitting from Regime Assignment

## Goal

Day 2 separates regime model fitting from regime label assignment.

The previous implementation fit the scaler, PCA, and KMeans model and assigned labels in one full-sample function. That design made it hard to build a walk-forward pipeline because regime assignment could not be performed using a model trained only on historical data.

## Change

The regime module now exposes three layers:

1. `fit_regime_estimator(...)`
   - fits scaler, PCA, and KMeans on the provided training window
   - returns a fitted model object
   - does not assign labels

2. `assign_regimes(...)`
   - uses an already-fitted model
   - assigns regimes to a provided market-state frame
   - records the model fit start and end dates

3. `fit_regime_model(...)`
   - remains as a backward-compatible full-sample diagnostic wrapper
   - should not be treated as out-of-sample evidence

## Why this matters

This is the required foundation for walk-forward regime assignment.

After this change, Day 3 can repeatedly fit the regime model on historical windows and assign labels to later dates without refitting on future observations.

## Current status

The project still has the intentional xfailed leakage test from Day 1.

That test should remain xfailed until the main research workflow uses walk-forward regime labels.
