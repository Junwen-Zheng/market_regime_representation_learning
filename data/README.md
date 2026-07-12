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

## Optional Yahoo downloader

The public-data download helper requires a separate Python 3.11-or-newer environment:

    python3.11 -m venv .venv-download
    source .venv-download/bin/activate
    pip install -r requirements-data-download.txt
    python scripts/download_yahoo_data.py --start 2018-01-01 --end 2026-07-02

The downloaded `data/raw/prices.csv` file is excluded from Git. Committed reports and research outputs are derived from the fixed downloaded panel.
