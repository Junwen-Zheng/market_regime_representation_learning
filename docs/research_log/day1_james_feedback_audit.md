# Day 1 — James Feedback Audit: Regime Look-Ahead and Synthetic-Only Workflow

## Context

James reviewed this project and flagged two core research problems:

1. The regime model is currently fit in-sample.
2. The project currently uses synthetic data only, with no real-data path.

These are valid criticisms. This note documents the issues directly so the next changes can be made deliberately rather than hidden behind new modelling complexity.

## Issue 1 — In-sample regime fitting creates look-ahead risk

The current regime workflow fits the scaler, PCA, and KMeans model using the full market-state feature history.

That means regime labels for earlier dates can be influenced by later market states. Any conditional rank-IC result sliced by these labels may therefore contain look-ahead bias.

This is especially important because the output currently presents conditional alpha behavior by learned regime. If the regime labels themselves are learned using future observations, the conditional results are not valid out-of-sample evidence.

## Why this matters

A realistic research process should only use information available at the time a regime label is assigned.

For example, if assigning a regime label for date T, the model should be fit only on data available up to T or up to a prior training cutoff. Future observations after T should not affect:

- feature standardisation
- PCA loadings
- KMeans cluster centroids
- regime labels for past dates

The current full-sample approach violates this principle.

## Issue 2 — Synthetic-only data path

The current workflow uses synthetic finance-shaped data. Synthetic data is useful for testing whether the pipeline runs, but it cannot support claims about real market behavior.

Synthetic data should be treated as a smoke test only.

A stronger version of this project needs an explicit real-data path with no silent fallback to synthetic data.

## Day 1 decision

For now, I will not add more complex models. The priority is to make the research process correct.

The next stages should be:

1. Split regime model fitting from regime assignment.
2. Build walk-forward regime assignment.
3. Make walk-forward regime labels the default for conditional IC evaluation.
4. Add a real OHLCV data path.
5. Keep synthetic data only as a reproducible smoke test.

## Current interpretation of existing results

Existing conditional regime results should be treated as diagnostic only.

They are useful for checking pipeline mechanics, but they should not be presented as non-leaky evidence of alpha robustness.
