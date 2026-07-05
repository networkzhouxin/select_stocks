# -*- coding: utf-8 -*-
"""Tests for JoinQuant/local order-path diagnostics."""

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_parse_joinquant_order_events_normalizes_buy_and_sell_logs():
    from cross_signal_strategy.order_path_diagnostics import parse_joinquant_order_events

    text = """
2019-10-18 09:35:00 - INFO  - [buy] 513880.XSHG buy=80 rev=25 loc=17 trend=9 vol=4 target=5000
2019-11-18 09:35:00 - INFO  - [sell] 159928.XSHE reason=signal_sell amount=1300
"""

    events = parse_joinquant_order_events(text)

    assert [event.as_key() for event in events] == [
        ("2019-10-18", "BUY", "513880"),
        ("2019-11-18", "SELL", "159928"),
    ]
    assert events[0].target_value == 5000.0
    assert events[1].reason == "signal_sell"
    assert events[1].amount == 1300


def test_parse_joinquant_sell_reason_can_contain_score_text():
    from cross_signal_strategy.order_path_diagnostics import parse_joinquant_order_events

    text = "2019-02-28 09:35:00 - INFO  - [sell] 159915.XSHE reason=sell_score 31 amount=3900"

    events = parse_joinquant_order_events(text)

    assert len(events) == 1
    assert events[0].as_key() == ("2019-02-28", "SELL", "159915")
    assert events[0].reason == "sell_score 31"
    assert events[0].amount == 3900


def test_parse_joinquant_filled_order_events_ignores_cancelled_intent():
    from cross_signal_strategy.order_path_diagnostics import parse_joinquant_filled_order_events

    text = """
2019-12-12 09:35:00 - INFO  - [sell] 513880.XSHG reason=sell_score 45 amount=5700
2019-12-12 09:35:00 - WARNING - 该标的截至到目前成交量为 0 ，暂时无法成交：StockOrder(entrust_id=1 security=513880.XSHG mode=OrderTargetAmount: _amount=0 style=MarketOrderStyle: _limit_price=0.0 side=long action=close margin=False)
2019-12-12 09:35:00 - INFO  - 订单取消完成：StockOrder(entrust_id=1 security=513880.XSHG mode=OrderTargetAmount: _amount=0 style=MarketOrderStyle: _limit_price=0.0 side=long action=close margin=False)
2019-12-30 09:35:00 - INFO  - order StockOrder(entrust_id=2 security=513880.XSHG mode=OrderTargetAmount: _amount=0 style=MarketOrderStyle: _limit_price=0.0 side=long action=close margin=False) trade price: 1.098, amount:5700, commission: 5.0
"""

    events = parse_joinquant_filled_order_events(text)

    assert [event.as_key() for event in events] == [("2019-12-30", "SELL", "513880")]
    assert events[0].amount == 5700
    assert events[0].price == 1.098


def test_parse_joinquant_filled_order_events_parses_open_as_buy():
    from cross_signal_strategy.order_path_diagnostics import parse_joinquant_filled_order_events

    text = (
        "2019-10-18 09:35:00 - INFO  - order StockOrder(entrust_id=3 security=513880.XSHG "
        "mode=OrderTargetValue: _value=6106.62 style=MarketOrderStyle: _limit_price=0.0 "
        "side=long action=open margin=False) trade price: 1.063, amount:5700, commission: 5.0"
    )

    events = parse_joinquant_filled_order_events(text)

    assert [event.as_key() for event in events] == [("2019-10-18", "BUY", "513880")]
    assert events[0].amount == 5700
    assert events[0].price == 1.063


def test_extract_local_order_events_uses_filled_orders_only():
    from cross_signal_strategy.local_backtester import DayResult, OrderResult, Position
    from cross_signal_strategy.order_path_diagnostics import extract_local_order_events

    results = [
        DayResult(
            date="2019-10-18",
            previous_date="2019-10-17",
            orders=[
                OrderResult("513880", 1000, 1.001, 5.0, "2019-10-18 09:35", True),
                OrderResult("510300", 0, 3.0, 0.0, "2019-10-18 09:35", False, "no change"),
            ],
            cash=1000.0,
            positions={"513880": Position("513880", 1000, 1.001)},
            marks={"513880": 1.0},
            total_value=2000.0,
        ),
        DayResult(
            date="2019-11-18",
            previous_date="2019-11-15",
            orders=[OrderResult("159928", -1300, 3.0, 5.0, "2019-11-18 09:35", True)],
            cash=4900.0,
            positions={},
            marks={},
            total_value=4900.0,
        ),
    ]

    events = extract_local_order_events(results)

    assert [event.as_key() for event in events] == [
        ("2019-10-18", "BUY", "513880"),
        ("2019-11-18", "SELL", "159928"),
    ]
    assert events[0].amount == 1000
    assert events[1].amount == 1300


def test_find_first_order_divergence_reports_missing_or_different_event():
    from cross_signal_strategy.order_path_diagnostics import (
        OrderPathEvent,
        find_first_order_divergence,
    )

    jq_events = [
        OrderPathEvent("2019-10-18", "BUY", "513880"),
        OrderPathEvent("2020-09-22", "BUY", "512100"),
    ]
    local_events = [
        OrderPathEvent("2019-10-18", "BUY", "513880"),
        OrderPathEvent("2020-09-29", "BUY", "512100"),
    ]

    divergence = find_first_order_divergence(jq_events, local_events)

    assert divergence is not None
    assert divergence.index == 1
    assert divergence.expected.as_key() == ("2020-09-22", "BUY", "512100")
    assert divergence.actual.as_key() == ("2020-09-29", "BUY", "512100")
    assert "first mismatch at order index 1" in divergence.message
