# -*- coding: utf-8 -*-
"""Tests for the observation-only cross-signal trade quality ledger."""

import pathlib
import sys
import types

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))


class FakeLoader:
    def __init__(self, frames):
        self.frames = frames

    def load_daily_frame(self, code, trade_date):
        year = pd.Timestamp(trade_date).year
        key = (str(code).split(".")[0], year)
        if key not in self.frames:
            raise FileNotFoundError(key)
        return self.frames[key].copy()


def _trade(**overrides):
    from cross_signal_strategy.research.trade_diagnostics import ClosedTradeDiagnostic

    values = {
        "code": "513100",
        "buy_date": "2020-12-30",
        "sell_date": "2021-01-05",
        "sell_reason": "signal_sell",
        "amount": 100,
        "buy_price": 10.0,
        "sell_price": 10.5,
        "pnl": 45.0,
        "return_pct": 5.0,
        "entry_score": {"atr": 1.0, "signal_date": "2020-12-29"},
    }
    values.update(overrides)
    return ClosedTradeDiagnostic(**values)


def test_trade_quality_ledger_uses_actual_fill_and_combines_training_years():
    from cross_signal_strategy.research.trade_quality_ledger import build_trade_quality_ledger

    loader = FakeLoader({
        ("513100", 2020): pd.DataFrame({
            "date": ["2020-12-29", "2020-12-30", "2020-12-31"],
            "close": [9.8, 10.4, 11.1],
        }),
        ("513100", 2021): pd.DataFrame({
            "date": ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"],
            "close": [8.8, 10.6, 11.2, 9.5],
        }),
    })

    row = build_trade_quality_ledger(
        [_trade()], loader, post_sell_horizons=(1, 2, 5)
    )[0]

    assert row.market_group == "qdii"
    assert row.holding_trade_days == 4
    assert row.holding_mfe == pytest.approx(0.11)
    assert row.holding_mae == pytest.approx(-0.12)
    assert row.entry_mfe[5] == pytest.approx(0.12)
    assert row.entry_mae[5] == pytest.approx(-0.12)
    assert row.first_profitable_close_offset == 0
    assert row.first_atr_barrier == "up_first"
    assert row.post_sell_returns[1] == pytest.approx(11.2 / 10.5 - 1.0)
    assert row.post_sell_returns[2] == pytest.approx(9.5 / 10.5 - 1.0)
    assert row.post_sell_returns[5] is None


def test_trade_quality_ledger_labels_down_barrier_without_changing_trade_result():
    from cross_signal_strategy.research.trade_quality_ledger import build_trade_quality_ledger

    loader = FakeLoader({
        ("510300", 2021): pd.DataFrame({
            "date": ["2021-06-01", "2021-06-02", "2021-06-03", "2021-06-04"],
            "close": [9.6, 8.9, 10.1, 11.2],
        }),
    })
    trade = _trade(
        code="510300",
        buy_date="2021-06-01",
        sell_date="2021-06-04",
        sell_reason="atr_stop",
        buy_price=10.0,
        sell_price=11.0,
        return_pct=10.0,
        entry_score={"atr": 1.0, "signal_date": "2021-05-31"},
    )

    row = build_trade_quality_ledger([trade], loader)[0]

    assert row.market_group == "non_qdii"
    assert row.first_atr_barrier == "down_first"
    assert row.realized_return_pct == pytest.approx(10.0)


def test_hang_seng_etf_is_classified_as_qdii():
    from cross_signal_strategy.research.trade_quality_ledger import build_trade_quality_ledger

    loader = FakeLoader({
        ("159920", 2021): pd.DataFrame({
            "date": ["2021-06-01", "2021-06-02"],
            "close": [10.0, 10.1],
        }),
    })
    trade = _trade(
        code="159920", buy_date="2021-06-01", sell_date="2021-06-02",
        entry_score={"atr": 1.0, "signal_date": "2021-05-31"},
    )

    row = build_trade_quality_ledger([trade], loader)[0]

    assert row.market_group == "qdii"


def test_holding_excursion_excludes_close_after_the_0935_exit():
    from cross_signal_strategy.research.trade_quality_ledger import build_trade_quality_ledger

    loader = FakeLoader({
        ("510300", 2021): pd.DataFrame({
            "date": ["2021-06-01", "2021-06-02", "2021-06-03"],
            "close": [10.0, 9.0, 20.0],
        }),
    })
    trade = _trade(
        code="510300", buy_date="2021-06-01", sell_date="2021-06-03",
        buy_price=10.0, sell_price=9.5,
        entry_score={"atr": 2.0, "signal_date": "2021-05-31"},
    )

    row = build_trade_quality_ledger([trade], loader)[0]

    assert row.holding_mfe == pytest.approx(0.0)
    assert row.holding_mae == pytest.approx(-0.10)


def test_incomplete_atr_barrier_window_is_unavailable_when_no_barrier_was_hit():
    from cross_signal_strategy.research.trade_quality_ledger import build_trade_quality_ledger

    loader = FakeLoader({
        ("510300", 2021): pd.DataFrame({
            "date": ["2021-12-27", "2021-12-28", "2021-12-29", "2021-12-30", "2021-12-31"],
            "close": [10.0, 10.1, 9.9, 10.2, 10.0],
        }),
    })
    trade = _trade(
        code="510300", buy_date="2021-12-27", sell_date="2021-12-31",
        buy_price=10.0, sell_price=10.0,
        entry_score={"atr": 1.0, "signal_date": "2021-12-24"},
    )

    row = build_trade_quality_ledger([trade], loader)[0]

    assert row.first_atr_barrier == "unavailable"


@pytest.mark.parametrize(
    "trade, error",
    [
        (_trade(buy_date="2022-01-04", sell_date="2022-01-05"), "training window"),
        (_trade(entry_score={"atr": 1.0, "signal_date": "2020-12-30"}), "before buy date"),
    ],
)
def test_trade_quality_ledger_rejects_leakage_boundaries(trade, error):
    from cross_signal_strategy.research.trade_quality_ledger import build_trade_quality_ledger

    with pytest.raises(ValueError, match=error):
        build_trade_quality_ledger([trade], FakeLoader({}))


def test_trade_quality_ledger_accepts_proven_1445_same_day_signal_only():
    from cross_signal_strategy.research.trade_quality_ledger import (
        build_trade_quality_ledger,
    )

    loader = FakeLoader({
        ("510300", 2021): pd.DataFrame({
            "date": ["2021-06-01", "2021-06-02", "2021-06-03"],
            "close": [10.2, 10.4, 10.5],
        }),
    })
    trade = _trade(
        code="510300",
        buy_date="2021-06-01",
        sell_date="2021-06-03",
        buy_price=10.0,
        sell_price=10.5,
        entry_score={
            "atr": 0.2,
            "signal_date": "2021-06-01",
            "decision_time": "14:45",
            "data_cutoff": "14:44",
        },
    )

    row = build_trade_quality_ledger([trade], loader)[0]

    assert row.buy_date == "2021-06-01"
    assert row.realized_return_pct == pytest.approx(5.0)


def test_trade_quality_summary_groups_by_year_market_and_reason():
    from cross_signal_strategy.research.trade_quality_ledger import (
        TradeQualityRow,
        summarize_trade_quality,
    )

    rows = [
        TradeQualityRow(
            code="513100", buy_date="2020-01-02", sell_date="2020-01-10",
            sell_reason="signal_sell", market_group="qdii", realized_return_pct=8.0,
            holding_trade_days=7, holding_mfe=0.12, holding_mae=-0.03,
            entry_mfe={5: 0.10, 10: 0.12}, entry_mae={5: -0.02, 10: -0.03},
            first_profitable_close_offset=0, first_atr_barrier="up_first",
            post_sell_returns={5: 0.02, 10: 0.03},
        ),
        TradeQualityRow(
            code="510300", buy_date="2020-02-03", sell_date="2020-02-10",
            sell_reason="atr_stop", market_group="non_qdii", realized_return_pct=-4.0,
            holding_trade_days=6, holding_mfe=0.01, holding_mae=-0.06,
            entry_mfe={5: 0.01, 10: 0.02}, entry_mae={5: -0.05, 10: -0.06},
            first_profitable_close_offset=None, first_atr_barrier="down_first",
            post_sell_returns={5: -0.01, 10: 0.01},
        ),
    ]

    summary = summarize_trade_quality(rows)

    assert summary["all"].count == 2
    assert summary["all"].win_rate == pytest.approx(0.5)
    assert summary["year:2020"].count == 2
    assert summary["market:qdii"].up_first_rate == pytest.approx(1.0)
    assert summary["reason:atr_stop"].mean_return_pct == pytest.approx(-4.0)
