# Failure Analysis

## What Could Be Wrong

This project can fail in several ways.

First, learned regimes may not be economically meaningful. PCA and KMeans can cluster statistical patterns that look stable but do not correspond to tradable market states.

Second, the alpha signals may be too simple. Momentum and reversal are heavily studied, and public OHLCV-only versions may have little remaining edge after realistic costs.

Third, synthetic data can overstate the quality of a research pipeline. Even if the pipeline detects conditional behavior, that may only reflect the assumptions embedded in the data generator.

Fourth, regime slicing reduces sample size. A signal can appear strong in one regime simply because that regime has fewer observations.

Fifth, transaction costs and turnover are not yet modelled in sufficient detail. A positive rank IC does not imply a profitable implementable portfolio.

## Why The Project Still Has Value

The value is not the result. The value is the research structure:

- define a hypothesis;
- construct interpretable market-state features;
- learn regime representations;
- test alpha conditionally;
- report failure modes honestly;
- preserve reproducible code and outputs.

This is closer to quant research process than a simple backtest screenshot.
