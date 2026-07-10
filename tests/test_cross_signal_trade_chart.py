# -*- coding: utf-8 -*-
"""Tests for segmented cross-signal K-line trade charts."""

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SAMPLE_LOG = """\
2019-02-11 09:35:00 - INFO  - [buy] 159915.XSHE buy=70 rev=35 loc=17 trend=14 vol=4 target=6333
2019-02-11 09:35:00 - INFO  - order StockOrder(entrust_id=1 security=159915.XSHE mode=OrderTargetValue: _value=6333 style=MarketOrderStyle: _limit_price=0.0 side=long action=open margin=False) trade price: 1.227, amount:5100, commission: 5.0
2019-02-20 09:35:00 - INFO  - [sell] 159915.XSHE reason=sell_score 34 amount=5100
2019-02-20 09:35:00 - INFO  - order StockOrder(entrust_id=2 security=159915.XSHE mode=OrderTargetAmount: _amount=0 style=MarketOrderStyle: _limit_price=0.0 side=long action=close margin=False) trade price: 1.300, amount:5100, commission: 5.0
"""


def daily_frame():
    return pd.DataFrame({
        "code": ["159915"] * 6,
        "symbol": ["创业板ETF易方达"] * 6,
        "date": pd.date_range("2019-02-11", periods=6, freq="B").strftime("%Y-%m-%d"),
        "open": [1.20, 1.22, 1.24, 1.25, 1.27, 1.29],
        "high": [1.24, 1.25, 1.27, 1.28, 1.30, 1.32],
        "low": [1.19, 1.21, 1.23, 1.24, 1.26, 1.28],
        "close": [1.22, 1.24, 1.25, 1.27, 1.29, 1.30],
        "volume": [100, 110, 120, 130, 140, 150],
    })


def test_default_periods_match_the_five_joinquant_backtests():
    from cross_signal_strategy.trade_chart import DEFAULT_PERIODS

    assert [(item.key, item.start, item.end) for item in DEFAULT_PERIODS] == [
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2018", "2015-01-01", "2018-12-31"),
        ("2019-2021", "2019-01-01", "2021-12-31"),
        ("2022-2023", "2022-01-01", "2023-12-31"),
        ("2024-latest", "2024-01-01", "2026-07-08"),
    ]


def test_parse_joinquant_log_extracts_fills_scores_and_sell_reason():
    from cross_signal_strategy.trade_chart import parse_joinquant_trade_log

    markers = parse_joinquant_trade_log(SAMPLE_LOG)

    assert len(markers) == 2
    assert markers[0].side == "buy"
    assert markers[0].code == "159915"
    assert markers[0].price == pytest.approx(1.227)
    assert markers[0].amount == 5100
    assert markers[0].buy_score == pytest.approx(70.0)
    assert markers[0].trend_score == pytest.approx(14.0)
    assert markers[1].side == "sell"
    assert markers[1].reason == "sell_score 34"


def test_parse_joinquant_log_handles_wrapped_fill_line():
    from cross_signal_strategy.trade_chart import parse_joinquant_trade_log

    wrapped = """\
2025-08-25 09:35:00 - INFO  - [buy] 513050.XSHG buy=65 rev=35 loc=15 trend=9 vol=6 target=9537
2025-08-25 09:35:00 - INFO  - order StockOrder(entrust_id=1 security=513050.XSHG mode=OrderTargetValue: _value=9537.43 style=MarketOrderStyle: _limit_price=0.0 side=long action=open margin=False comment= error=因为资金有限，下单数量调整为 6122
开仓数量必须是 100 的整数倍，调整为 6100) trade price: 1.523, amount:6100, commission: 5.0
"""

    markers = parse_joinquant_trade_log(wrapped)

    assert len(markers) == 1
    assert markers[0].code == "513050"
    assert markers[0].price == pytest.approx(1.523)
    assert markers[0].amount == 6100


def test_pair_trade_outcomes_adds_realized_pnl_to_sell_marker():
    from cross_signal_strategy.trade_chart import (
        pair_trade_outcomes,
        parse_joinquant_trade_log,
    )

    markers = pair_trade_outcomes(parse_joinquant_trade_log(SAMPLE_LOG))
    sell = markers[1]

    assert sell.realized_pnl == pytest.approx(5100 * (1.300 - 1.227) - 10.0)
    assert sell.return_pct == pytest.approx((1.300 / 1.227 - 1.0) * 100.0)
    assert sell.hold_days == 9


def test_build_symbol_dataset_contains_kline_ma_volume_and_trade_markers():
    from cross_signal_strategy.trade_chart import (
        build_symbol_dataset,
        pair_trade_outcomes,
        parse_joinquant_trade_log,
    )

    dataset = build_symbol_dataset(
        daily_frame(),
        pair_trade_outcomes(parse_joinquant_trade_log(SAMPLE_LOG)),
    )

    assert dataset["code"] == "159915"
    assert dataset["name"] == "创业板ETF易方达"
    assert len(dataset["dates"]) == 6
    assert dataset["ma5"][:4] == [None, None, None, None]
    assert dataset["ma5"][4] == pytest.approx(1.254)
    assert dataset["volume"][-1] == pytest.approx(150.0)
    assert dataset["buys"][0]["date"] == "2019-02-11"
    assert dataset["sells"][0]["date"] == "2019-02-20"
    assert dataset["sells"][0]["realized_pnl"] == pytest.approx(362.3)


def test_build_symbol_dataset_rejects_dates_outside_declared_period():
    from cross_signal_strategy.trade_chart import build_symbol_dataset

    with pytest.raises(ValueError, match="outside period"):
        build_symbol_dataset(
            daily_frame(),
            [],
            period_start="2020-01-01",
            period_end="2020-12-31",
        )


def test_render_period_page_has_etf_selector_kline_and_source_caveat():
    from cross_signal_strategy.trade_chart import render_period_page

    html = render_period_page(
        period_key="2019-2021",
        datasets={"159915": build_minimal_dataset()},
    )

    assert "plotly.min.js" in html
    assert 'id="etf-select"' in html
    assert "candlestick" in html
    assert "MA20" in html and "MA60" in html
    assert "聚宽成交" in html
    assert "本地日线" in html
    assert "159915" in html
    assert "rangeslider:{visible:false" in html


def test_render_index_links_each_period_and_reports_fill_count():
    from cross_signal_strategy.trade_chart import render_index_page

    html = render_index_page({
        "2010-2014": {"fills": 83, "symbols": 4, "last_kline": "2014-12-31"},
        "2015-2018": {"fills": 235, "symbols": 8, "last_kline": "2018-12-28"},
    })

    assert 'href="2010-2014.html"' in html
    assert 'href="2015-2018.html"' in html
    assert "83" in html and "235" in html


def build_minimal_dataset():
    return {
        "code": "159915",
        "name": "创业板ETF易方达",
        "dates": ["2019-02-11"],
        "open": [1.2],
        "high": [1.3],
        "low": [1.1],
        "close": [1.25],
        "volume": [100.0],
        "ma5": [None],
        "ma10": [None],
        "ma20": [None],
        "ma60": [None],
        "buys": [{"date": "2019-02-11", "price": 1.227, "amount": 5100}],
        "sells": [],
    }
