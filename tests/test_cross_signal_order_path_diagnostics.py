# -*- coding: utf-8 -*-
"""Tests for JoinQuant/local order-path diagnostics."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_parse_joinquant_order_events_normalizes_buy_and_sell_logs():
    from cross_signal_strategy.research.order_path_diagnostics import parse_joinquant_order_events

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
    from cross_signal_strategy.research.order_path_diagnostics import parse_joinquant_order_events

    text = "2019-02-28 09:35:00 - INFO  - [sell] 159915.XSHE reason=sell_score 31 amount=3900"

    events = parse_joinquant_order_events(text)

    assert len(events) == 1
    assert events[0].as_key() == ("2019-02-28", "SELL", "159915")
    assert events[0].reason == "sell_score 31"
    assert events[0].amount == 3900


def test_parse_joinquant_filled_order_events_ignores_cancelled_intent():
    from cross_signal_strategy.research.order_path_diagnostics import parse_joinquant_filled_order_events

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
    from cross_signal_strategy.research.order_path_diagnostics import parse_joinquant_filled_order_events

    text = (
        "2019-10-18 09:35:00 - INFO  - order StockOrder(entrust_id=3 security=513880.XSHG "
        "mode=OrderTargetValue: _value=6106.62 style=MarketOrderStyle: _limit_price=0.0 "
        "side=long action=open margin=False) trade price: 1.063, amount:5700, commission: 5.0"
    )

    events = parse_joinquant_filled_order_events(text)

    assert [event.as_key() for event in events] == [("2019-10-18", "BUY", "513880")]
    assert events[0].amount == 5700
    assert events[0].price == 1.063
    assert events[0].commission == 5.0


def test_parse_joinquant_transaction_csv_reads_filled_rows(tmp_path):
    from cross_signal_strategy.research.order_path_diagnostics import parse_joinquant_transaction_csv

    csv_path = tmp_path / "transaction.csv"
    csv_path.write_text(
        "\n".join(
            [
                "日期,委托时间,品种,标的,交易类型,下单类型,成交数量,成交价,成交额,委托数量,委托价格,平仓盈亏,手续费,状态,最后更新时间",
                "2020-09-22,09:35:00,基金,南方中证1000ETF(512100.XSHG),买,市价单,7700股,0.954,7345.8,7700股,--,0,5,全部成交,2020-09-22 09:35:00",
                "2020-09-23,09:35:00,基金,南方中证1000ETF(512100.XSHG),卖,市价单,7700股,0.947,-7291.9,7700股,--,-53.9,5,全部成交,2020-09-23 09:35:00",
            ]
        ),
        encoding="gbk",
    )

    events = parse_joinquant_transaction_csv(csv_path)

    assert [event.as_key() for event in events] == [
        ("2020-09-22", "BUY", "512100"),
        ("2020-09-23", "SELL", "512100"),
    ]
    assert events[0].amount == 7700
    assert events[0].price == 0.954
    assert events[0].trade_value == 7345.8
    assert events[0].commission == 5.0
    assert events[0].status == "全部成交"
    assert events[1].trade_value == -7291.9


def test_parse_joinquant_transaction_csv_ignores_cancelled_rows_by_default(tmp_path):
    from cross_signal_strategy.research.order_path_diagnostics import parse_joinquant_transaction_csv

    csv_path = tmp_path / "transaction.csv"
    csv_path.write_text(
        "\n".join(
            [
                "日期,委托时间,品种,标的,交易类型,下单类型,成交数量,成交价,成交额,委托数量,委托价格,平仓盈亏,手续费,状态,最后更新时间",
                "2019-12-12,09:35:00,基金,日经ETF(513880.XSHG),卖,市价单,0股,--,0,5700股,--,0,0,已撤单,2019-12-12 09:35:00",
            ]
        ),
        encoding="gbk",
    )

    assert parse_joinquant_transaction_csv(csv_path) == []


def test_extract_local_order_events_uses_filled_orders_only():
    from cross_signal_strategy.local.local_backtester import DayResult, OrderResult, Position
    from cross_signal_strategy.research.order_path_diagnostics import extract_local_order_events

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
    from cross_signal_strategy.research.order_path_diagnostics import (
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


def test_compare_order_execution_fields_reports_amount_price_fee_and_value_diffs():
    from cross_signal_strategy.research.order_path_diagnostics import (
        OrderPathEvent,
        compare_order_execution_fields,
    )

    expected = [OrderPathEvent("2020-09-22", "BUY", "512100", amount=7700, price=0.954, trade_value=7345.8, commission=5.0)]
    actual = [OrderPathEvent("2020-09-22", "BUY", "512100", amount=7600, price=0.954954, trade_value=7257.6504, commission=5.0)]

    diffs = compare_order_execution_fields(expected, actual)

    assert len(diffs) == 1
    assert diffs[0].key == ("2020-09-22", "BUY", "512100")
    assert diffs[0].amount_diff == -100
    assert diffs[0].price_diff == pytest.approx(0.000954)
    assert diffs[0].commission_diff == 0.0
    assert diffs[0].trade_value_diff == pytest.approx(-88.1496)
