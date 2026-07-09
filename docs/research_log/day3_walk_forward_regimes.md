# Day 3 — Walk-Forward Regime Assignment

## Goal

Day 3 adds a walk-forward regime assignment engine.

The model now repeatedly fits scaler, PCA, and KMeans on historical market-state windows, then assigns regime labels only to later dates.

## Why this matters

This directly addresses the look-ahead issue identified in Day 1.

A regime label for a given date should not depend on future observations. The walk-forward path enforces this by making every assigned date strictly later than the model fit end date.

## Implementation

The new module is:

`src/walk_forward_regimes.py`

It provides:

- `build_walk_forward_regime_assignments(...)`
- `WalkForwardRegimeResult`
- assignment output with model fit metadata
- fit-window output for auditability

Each assignment row records:

- assigned date
- PCA coordinates
- regime label
- regime model id
- model fit start date
- model fit end date
- assignment window start date
- assignment window end date

## Testing

Day 3 adds tests for:

1. assigned dates occur strictly after the model fit window
2. multiple refit windows are created
3. future shock rows do not change prior walk-forward assignments
4. duplicate dates are rejected
5. rolling-window configuration is validated
6. fixed-length rolling training windows work as expected

## Current status

The old full-sample regime function remains available as a diagnostic compatibility path.

The Day 1 xfailed leakage audit test remains intentional because it covers the old full-sample diagnostic function, not the new walk-forward path.

Day 4 should wire walk-forward assignments into the main research workflow and make them the default for conditional IC evaluation.
