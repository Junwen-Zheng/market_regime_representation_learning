# Day 5 — Explicit Real-Data Path

## Goal

Day 5 addresses the synthetic-only limitation identified in the research audit.

The project now supports an explicit real multi-asset OHLCV CSV path. Synthetic data remains available only through the explicitly named `synthetic_smoke_test` mode.

## Data modes

The workflow supports:

1. `real`
2. `synthetic_smoke_test`

Real mode is the default. Missing or invalid real data raises an exception, with no silent synthetic fallback.

## Real-data schema

Required columns:

- date
- asset
- sector
- open
- high
- low
- close
- volume

The loader validates schema, dates, numeric values, prices, volume, duplicate date/asset rows, and minimum asset and sector counts.

It derives daily returns and dollar volume.

## Verification

Tests cover valid real CSV loading, invalid-input rejection, explicit synthetic selection, no fallback, and end-to-end real-format workflow execution.

Day 6 will run the first real public-data experiment and produce a data-quality report.
