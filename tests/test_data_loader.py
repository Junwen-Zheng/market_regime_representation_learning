from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_equity_panel, load_real_equity_panel


def _raw_real_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=6)
    assets = [
        ("AAA", "Technology"),
        ("BBB", "Financials"),
        ("CCC", "Healthcare"),
    ]

    rows: list[dict[str, object]] = []

    for asset_index, (asset, sector) in enumerate(assets):
        for day_index, date in enumerate(dates):
            open_price = 100.0 + 10.0 * asset_index + day_index
            close_price = open_price * (
                1.0 + 0.001 * (asset_index + 1)
            )

            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "sector": sector,
                    "open": open_price,
                    "high": max(open_price, close_price) * 1.01,
                    "low": min(open_price, close_price) * 0.99,
                    "close": close_price,
                    "volume": 1_000_000 + 1_000 * day_index,
                }
            )

    return pd.DataFrame(rows)


def test_load_real_panel_derives_returns_and_dollar_volume(
    tmp_path: Path,
):
    path = tmp_path / "prices.csv"
    _raw_real_panel().to_csv(path, index=False)

    panel = load_real_equity_panel(path)

    expected_columns = {
        "date",
        "asset",
        "sector",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "return_1d",
    }

    assert set(panel.columns) == expected_columns
    assert len(panel) == 15
    assert panel["date"].nunique() == 5
    assert panel["asset"].nunique() == 3
    assert np.isfinite(panel["return_1d"]).all()

    assert np.allclose(
        panel["dollar_volume"],
        panel["close"] * panel["volume"],
    )


def test_real_loader_rejects_missing_file(tmp_path: Path):
    with pytest.raises(
        FileNotFoundError,
        match="No synthetic fallback",
    ):
        load_real_equity_panel(tmp_path / "missing.csv")


def test_real_loader_rejects_missing_required_columns(
    tmp_path: Path,
):
    path = tmp_path / "prices.csv"

    _raw_real_panel().drop(columns=["sector"]).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        load_real_equity_panel(path)


def test_real_loader_rejects_duplicate_date_asset_rows(
    tmp_path: Path,
):
    path = tmp_path / "prices.csv"
    raw = _raw_real_panel()

    duplicated = pd.concat(
        [raw, raw.iloc[[0]]],
        ignore_index=True,
    )
    duplicated.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="duplicate date/asset",
    ):
        load_real_equity_panel(path)


def test_real_loader_rejects_nonpositive_prices(
    tmp_path: Path,
):
    path = tmp_path / "prices.csv"
    raw = _raw_real_panel()
    raw.loc[0, "close"] = 0.0
    raw.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        load_real_equity_panel(path)


def test_real_mode_requires_a_path():
    with pytest.raises(
        ValueError,
        match="data_path is required",
    ):
        load_equity_panel(data_mode="real")


def test_real_mode_never_falls_back_to_synthetic(
    tmp_path: Path,
):
    with pytest.raises(
        FileNotFoundError,
        match="No synthetic fallback",
    ):
        load_equity_panel(
            data_mode="real",
            data_path=tmp_path / "missing.csv",
            n_assets=10,
            n_days=100,
            n_sectors=3,
            seed=99,
        )


def test_synthetic_mode_must_be_selected_explicitly():
    panel = load_equity_panel(
        data_mode="synthetic_smoke_test",
        n_assets=5,
        n_days=20,
        n_sectors=2,
        seed=8,
    )

    assert panel["asset"].nunique() == 5
    assert panel["date"].nunique() == 20
    assert "hidden_regime" in panel.columns


def test_unknown_data_mode_is_rejected():
    with pytest.raises(ValueError, match="data_mode"):
        load_equity_panel(
            data_mode="automatic_fallback"
        )
