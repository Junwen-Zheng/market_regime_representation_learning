# Day 4 — Use Walk-Forward Regimes in the Main Workflow

## Goal

Day 4 wires the walk-forward regime assignment engine into the main research workflow.

Before this change, the end-to-end pipeline still used the full-sample regime fit by default. That meant the research outputs could still slice conditional rank IC by regime labels learned with future information.

## Change

`run_research_pipeline(...)` now defaults to:

`regime_mode="walk_forward"`

The workflow now:

1. generates the synthetic equity panel
2. builds market-state features
3. fits regime models on historical windows
4. assigns labels only to later dates
5. builds alpha signals
6. evaluates aggregate rank IC
7. evaluates conditional rank IC using walk-forward regime labels
8. writes regime fit-window metadata for auditability

## Diagnostic mode

The old full-sample path is still available as:

`regime_mode="full_sample_diagnostic"`

This mode is useful for debugging and PCA diagnostics, but it should not be treated as out-of-sample evidence.

## New workflow outputs

The default workflow now writes:

- `regime_assignments.csv`
- `regime_fit_windows.csv`
- `conditional_rank_ic_by_regime.csv`
- `regime_transition_matrix.csv`
- `regime_summary.csv`
- `research_summary.csv`

`regime_fit_windows.csv` records the training window used for each regime model.

## Current interpretation

Conditional IC by regime is now based on walk-forward regime labels in the default workflow.

The Day 1 xfailed test remains intentional because it documents that the old full-sample diagnostic wrapper is leaky.
