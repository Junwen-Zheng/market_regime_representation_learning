# Data

The primary research workflow expects a real multi-asset OHLCV CSV at:

`data/raw/prices.csv`

Required columns:

- `date`
- `asset`
- `sector`
- `open`
- `high`
- `low`
- `close`
- `volume`

The loader derives daily close-to-close returns and dollar volume.

Download the configured public-data universe with:

    python scripts/download_yahoo_data.py --start 2018-01-01 --end 2026-07-01

Downloaded raw CSV files are excluded from Git.

The configured universe is a convenient research sample, not a survivorship-bias-free historical universe. Yahoo/yfinance data is suitable for this public research exercise but is not treated as institutional point-in-time market data.
