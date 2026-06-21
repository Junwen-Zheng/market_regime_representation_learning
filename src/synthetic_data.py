from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_equity_panel(
    n_assets: int = 120,
    n_days: int = 760,
    n_sectors: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a reproducible synthetic OHLCV-style equity panel.

    The purpose is not to create a profitable strategy. It is to create a stable,
    finance-shaped dataset where the research pipeline can be run end-to-end
    without vendor data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    assets = [f"STK{i:03d}" for i in range(n_assets)]
    sectors = [f"Sector_{i % n_sectors}" for i in range(n_assets)]
    sector_map = dict(zip(assets, sectors))

    # Hidden regime sequence with persistence. The model will not observe this.
    hidden_regime = np.zeros(n_days, dtype=int)
    transition = np.array(
        [
            [0.94, 0.03, 0.02, 0.01],
            [0.04, 0.90, 0.04, 0.02],
            [0.03, 0.05, 0.88, 0.04],
            [0.06, 0.04, 0.05, 0.85],
        ]
    )
    for t in range(1, n_days):
        hidden_regime[t] = rng.choice(4, p=transition[hidden_regime[t - 1]])

    # Regime-specific market return and volatility assumptions.
    regime_mu = np.array([0.00045, -0.00025, 0.00005, 0.00015])
    regime_sigma = np.array([0.006, 0.014, 0.010, 0.020])
    market_returns = rng.normal(regime_mu[hidden_regime], regime_sigma[hidden_regime])

    sector_shocks = rng.normal(0, 0.006, size=(n_days, n_sectors))
    idio_vol = rng.uniform(0.008, 0.018, size=n_assets)
    betas = rng.normal(1.0, 0.20, size=n_assets)

    rows = []
    prices = rng.uniform(30, 160, size=n_assets)
    base_volume = rng.lognormal(mean=12.0, sigma=0.7, size=n_assets)

    # A weak synthetic alpha effect: in calm regimes momentum has modest persistence;
    # in stressed regimes short-term reversal is stronger. This lets the evaluation
    # pipeline detect conditional behavior without claiming real-world alpha.
    trailing_returns = np.zeros((n_days, n_assets))

    for t, date in enumerate(dates):
        reg = hidden_regime[t]
        for i, asset in enumerate(assets):
            sector_idx = i % n_sectors
            base_ret = (
                betas[i] * market_returns[t]
                + 0.45 * sector_shocks[t, sector_idx]
                + rng.normal(0, idio_vol[i])
            )
            if t >= 20:
                mom_20 = trailing_returns[t - 20 : t, i].sum()
                rev_5 = -trailing_returns[t - 5 : t, i].sum()
                if reg == 0:  # calmer trend regime
                    base_ret += 0.025 * mom_20
                elif reg == 3:  # stressed liquidity regime
                    base_ret += 0.030 * rev_5
            ret = np.clip(base_ret, -0.15, 0.15)
            trailing_returns[t, i] = ret
            prev_price = prices[i]
            close = max(1.0, prev_price * (1.0 + ret))
            high = max(prev_price, close) * (1.0 + abs(rng.normal(0, 0.004)))
            low = min(prev_price, close) * (1.0 - abs(rng.normal(0, 0.004)))
            volume_noise = rng.lognormal(0, 0.25)
            regime_volume_boost = 1.0 + (0.5 if reg == 3 else 0.0)
            volume = base_volume[i] * volume_noise * regime_volume_boost
            dollar_volume = close * volume
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "sector": sector_map[asset],
                    "open": prev_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "dollar_volume": dollar_volume,
                    "return_1d": ret,
                    "hidden_regime": reg,
                }
            )
            prices[i] = close
    return pd.DataFrame(rows)
