# -*- coding: utf-8 -*-
"""Tests for observation-only attribution of changed KDJ candidate days."""

from __future__ import annotations

from copy import deepcopy
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ScoreSource:
    def __init__(self, scores):
        self.scores = scores

    def score(self, code, current_date, return_reason=False):
        value = self.scores.get((str(current_date), str(code).split(".")[0]))
        result = deepcopy(value) if value is not None else None
        reason = None if result is not None else "no_data"
        return (result, reason) if return_reason else result


class PriceLoader:
    def __init__(self, minute_prices, daily_closes):
        self.minute_prices = minute_prices
        self.daily_closes = daily_closes

    def get_minute_bar(self, code, trade_date, trade_time="09:35"):
        return {"close": self.minute_prices[(str(trade_date), str(code))]}

    def load_daily_frame(self, code, trade_date):
        rows = [
            {"date": date, "close": close}
            for (date, item_code), close in self.daily_closes.items()
            if item_code == str(code)
        ]
        return pd.DataFrame(rows)


def _module():
    try:
        from cross_signal_strategy.research import (
            kdj_tiered_changed_day_attribution,
        )
    except ImportError as exc:  # pragma: no cover - exercised in TDD red phase
        pytest.fail("changed-day attribution is not implemented: %s" % exc)
    return kdj_tiered_changed_day_attribution


def _day(date, orders, total_value=20000.0):
    from cross_signal_strategy.local.local_backtester import DayResult

    return DayResult(
        date=date,
        previous_date=None,
        orders=orders,
        cash=total_value,
        positions={},
        marks={},
        total_value=total_value,
    )


def _order(code, amount, reason="buy_signal"):
    from cross_signal_strategy.local.local_backtester import OrderResult

    return OrderResult(
        code=code,
        amount_delta=amount,
        exec_price=10.0,
        commission=5.0,
        side_time="2019-01-02 09:35",
        filled=True,
        reason=reason,
    )


def _score(code, **overrides):
    values = {
        "code": code,
        "k": 50.0,
        "buy_score": 40.0,
        "sell_score": 0.0,
        "buy_extreme_zone_score": 0.0,
        "sell_extreme_zone_score": 0.0,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "ma20_slope_non_negative": True,
        "adx": 10.0,
        "plus_di": 20.0,
        "minus_di": 10.0,
    }
    values.update(overrides)
    return values


def _loader(dates, codes):
    minute = {(date, code): 10.0 for date in dates for code in codes}
    daily = {
        (date, code): 10.0 + index
        for index, date in enumerate(dates)
        for code in codes
    }
    return PriceLoader(minute, daily)


def test_candidate_only_bonus_buy_is_direct_and_displaced_buy_is_same_day_chain():
    module = _module()
    date = "2019-01-02"
    official = ScoreSource({
        (date, "AAA"): _score("AAA", k=19.0, buy_score=55.0),
        (date, "BBB"): _score("BBB", buy_score=65.0),
    })
    candidate = ScoreSource({
        (date, "AAA"): _score(
            "AAA", k=19.0, buy_score=75.0, buy_extreme_zone_score=20.0
        ),
        (date, "BBB"): _score("BBB", buy_score=65.0),
    })

    rows = module.build_changed_order_attributions(
        [_day(date, [_order("BBB", 100)])],
        [_day(date, [_order("AAA", 100)])],
        official,
        candidate,
        _loader((date,), ("AAA", "BBB")),
        (date,),
    )

    by_key = {(row.path, row.code): row for row in rows}
    assert by_key[("candidate_only", "AAA")].origin == "buy_bonus_direct"
    assert by_key[("candidate_only", "AAA")].buy_threshold_crossed
    assert by_key[("baseline_only", "BBB")].origin == "same_day_portfolio_chain"


def test_candidate_only_threshold_sell_records_confirmation_and_adx_state():
    module = _module()
    date = "2019-01-09"
    official = ScoreSource({
        (date, "AAA"): _score("AAA", k=82.0, sell_score=25.0),
    })
    candidate = ScoreSource({
        (date, "AAA"): _score(
            "AAA",
            k=82.0,
            sell_score=35.0,
            sell_extreme_zone_score=10.0,
            close_below_ma20=True,
            adx=35.0,
            plus_di=30.0,
            minus_di=10.0,
        ),
    })

    rows = module.build_changed_order_attributions(
        [_day(date, [])],
        [_day(date, [_order("AAA", -100, reason="signal_sell")])],
        official,
        candidate,
        _loader((date,), ("AAA",)),
        (date,),
    )

    row = rows[0]
    assert row.origin == "sell_bonus_direct"
    assert row.sell_threshold_crossed
    assert row.price_confirmed
    assert not row.adx_protected


def test_candidate_only_order_without_active_bonus_is_portfolio_chain():
    module = _module()
    date = "2019-01-10"
    official = ScoreSource({(date, "AAA"): _score("AAA", buy_score=65.0)})
    candidate = ScoreSource({(date, "AAA"): _score("AAA", buy_score=65.0)})

    rows = module.build_changed_order_attributions(
        [_day(date, [])],
        [_day(date, [_order("AAA", 100)])],
        official,
        candidate,
        _loader((date,), ("AAA",)),
        (date,),
    )

    assert rows[0].origin == "portfolio_chain"


def test_forward_returns_are_ex_post_next_session_labels():
    module = _module()
    dates = ("2019-01-02", "2019-01-03", "2019-01-04")
    loader = PriceLoader(
        {(date, "AAA"): 10.0 for date in dates},
        {
            (dates[0], "AAA"): 10.0,
            (dates[1], "AAA"): 11.0,
            (dates[2], "AAA"): 9.0,
        },
    )
    official = ScoreSource({(dates[0], "AAA"): _score("AAA")})
    candidate = ScoreSource({
        (dates[0], "AAA"): _score(
            "AAA", k=19.0, buy_score=65.0, buy_extreme_zone_score=20.0
        )
    })

    row = module.build_changed_order_attributions(
        [_day(dates[0], [])],
        [_day(dates[0], [_order("AAA", 100)])],
        official,
        candidate,
        loader,
        dates,
        horizons=(1, 2, 3),
    )[0]

    assert row.forward_returns[1] == pytest.approx(0.10)
    assert row.forward_returns[2] == pytest.approx(-0.10)
    assert row.forward_returns[3] is None


def _trade(code, buy_date, sell_date, pnl, amount=100):
    from cross_signal_strategy.research.trade_diagnostics import (
        ClosedTradeDiagnostic,
    )

    return ClosedTradeDiagnostic(
        code=code,
        buy_date=buy_date,
        sell_date=sell_date,
        sell_reason="signal_sell",
        amount=amount,
        buy_price=10.0,
        sell_price=11.0,
        pnl=pnl,
        return_pct=10.0,
    )


def test_changed_trade_comparison_keeps_path_only_and_pnl_changed_round_trips():
    module = _module()
    baseline = [_trade("AAA", "2019-01-02", "2019-01-09", 100.0)]
    candidate = [
        _trade("AAA", "2019-01-02", "2019-01-09", 80.0),
        _trade("BBB", "2019-01-03", "2019-01-10", -30.0),
    ]

    rows = module.compare_closed_trade_paths(baseline, candidate)

    assert [(row.path, row.code, row.pnl_delta) for row in rows] == [
        ("matched_changed", "AAA", pytest.approx(-20.0)),
        ("candidate_only", "BBB", pytest.approx(-30.0)),
    ]


def test_training_runner_rejects_unapproved_data_roots():
    module = _module()

    with pytest.raises(ValueError, match="approved training data root"):
        module.run_kdj_tiered_changed_day_attribution(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )
