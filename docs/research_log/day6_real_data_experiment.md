# Day 6 — First Real-Data Walk-Forward Experiment

## Goal

Day 6 runs the first end-to-end experiment using real public OHLCV data rather than synthetic observations.

The experiment uses the explicit real-data path introduced on Day 5 and the walk-forward regime assignment introduced on Days 3 and 4.

## Dataset

The fixed research universe contains:

- 24 liquid US equities
- 8 sectors
- 2,135 trading dates per asset
- date range from 2018-01-02 through 2026-07-01
- 51,240 raw OHLCV rows
- 51,216 validated rows after removing each asset's first undefined return

The panel is balanced across the selected universe.

## Data-quality checks

The report found:

- zero duplicate date/asset rows
- zero missing OHLCV values
- zero non-finite OHLCV values
- zero nonpositive prices
- zero negative-volume rows
- zero rows where high was below low
- zero missing rows relative to the balanced panel

There were 12 absolute daily returns above 20 percent. These observations should be reviewed as potential event-driven outliers, but they are not automatically classified as data errors.

## Regime experiment

Configuration:

- regime model: standardisation, PCA, and KMeans
- principal components: 3
- regimes: 4
- initial training history: 252 days
- refit frequency: 20 days
- regime mode: walk-forward
- regime-assigned days: 1,823
- model-fit windows: 92

Every assigned date occurs strictly after the corresponding model fit end date.

## Aggregate signal results

The strongest aggregate result was:

- signal: `momentum_60d_z`
- mean rank IC: 0.012318
- rank-IC standard deviation: 0.322452
- IC information ratio: 0.038202
- evaluated days: 2,064

Other signals had smaller or slightly negative mean rank IC.

## Interpretation

The experiment does not provide strong evidence of tradable alpha.

The mean IC values are small relative to their variation. The main contribution of this stage is methodological:

- a real-data path exists
- the dataset is auditable
- regime assignments are walk-forward
- conditional evaluation no longer uses future-fitted regime labels
- results are reported without overstating statistical or economic significance

## Limitations

- The universe consists of current surviving securities.
- The sample therefore contains survivorship and selection bias.
- The cross-section is small at 24 assets.
- Public Yahoo data is not institutional point-in-time data.
- Delisted securities and delisting returns are absent.
- Transaction costs, borrow availability, and execution effects are not modelled.
- Regime identifiers may not represent stable economic states across refits.
