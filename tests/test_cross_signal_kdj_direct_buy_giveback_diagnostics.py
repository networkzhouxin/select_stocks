# -*- coding: utf-8 -*-
"""Tests for the observation-only direct-KDJ-buy giveback diagnostic."""

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
        code_text = str(code).split(".")[0]
        return pd.DataFrame([
            {"date": date, "close": close}
            for (date, item_code), close in self.daily_closes.items()
            if item_code == code_text
        ])


def _module():
    try:
        from cross_signal_strategy.research import (
            kdj_direct_buy_giveback_diagnostics,
        )
    except ImportError as exc:  # pragma: no cover - exercised in TDD red phase
        pytest.fail("giveback diagnostic is not implemented: %s" % exc)
    return kdj_direct_buy_giveback_diagnostics


def _score(signal_date, **overrides):
    values = {
        "signal_date": signal_date,
        "max_data_date": signal_date,
        "sell_score": 0.0,
        "sell_reversal_score": 0.0,
        "sell_risk_score": 0.0,
        "sell_extreme_zone_score": 0.0,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "macd_cross_down": False,
        "kdj_k_cross_down": False,
        "kdj_j_cross_down": False,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "ma20_slope_non_negative": False,
        "adx": 10.0,
        "plus_di": 10.0,
        "minus_di": 20.0,
    }
    values.update(overrides)
    return values


def _trade(buy_date, sell_date, buy_price=10.0, sell_price=9.8):
    from cross_signal_strategy.research.trade_diagnostics import (
        ClosedTradeDiagnostic,
    )

    return ClosedTradeDiagnostic(
        code="AAA",
        buy_date=buy_date,
        sell_date=sell_date,
        sell_reason="signal_sell",
        amount=100,
        buy_price=buy_price,
        sell_price=sell_price,
        pnl=-30.0,
        return_pct=(sell_price / buy_price - 1.0) * 100.0,
    )


def test_giveback_uses_actual_fill_and_excludes_post_exit_close_from_peak():
    module = _module()
    dates = pd.bdate_range("2019-01-02", periods=6).strftime("%Y-%m-%d").tolist()
    scores = {
        (date, "AAA"): _score(dates[index - 1])
        for index, date in enumerate(dates[1:], start=1)
    }
    daily_closes = {
        (date, "AAA"): close
        for date, close in zip(dates, [10.0, 10.5, 11.0, 12.0, 11.5, 15.0])
    }
    minute_prices = {(date, "AAA"): 10.0 for date in dates[1:]}

    item = module.build_trade_giveback_diagnostic(
        _trade(dates[0], dates[-1]),
        ScoreSource(scores),
        PriceLoader(minute_prices, daily_closes),
        dates,
        params={**module.strategy.get_default_params(), "min_signal_hold_days": 2},
    )

    assert item.peak_date == dates[3]
    assert item.peak_close == pytest.approx(12.0)
    assert item.peak_return == pytest.approx(0.20)
    assert item.realized_return == pytest.approx(-0.02)
    assert item.giveback_from_peak == pytest.approx(0.22)
    assert item.best_decision_date == dates[1]
    assert item.best_decision_price == pytest.approx(10.0)
    assert item.best_decision_return == pytest.approx(0.0)
    assert item.giveback_from_best_decision == pytest.approx(0.02)


def test_daily_states_separate_score_hold_confirmation_and_adx_bottlenecks():
    module = _module()
    dates = pd.bdate_range("2019-02-01", periods=6).strftime("%Y-%m-%d").tolist()
    scores = {
        (dates[1], "AAA"): _score(
            dates[0], sell_score=35.0, sell_reversal_score=35.0
        ),
        (dates[2], "AAA"): _score(
            dates[1],
            sell_score=35.0,
            sell_reversal_score=35.0,
            close_below_boll_mid=True,
            adx=35.0,
            plus_di=30.0,
            minus_di=10.0,
            ma20_slope_non_negative=True,
        ),
        (dates[3], "AAA"): _score(
            dates[2], sell_score=25.0, close_below_ma20=True
        ),
        (dates[4], "AAA"): _score(dates[3], sell_score=20.0),
        (dates[5], "AAA"): _score(
            dates[4],
            k=82.0,
            sell_score=35.0,
            sell_reversal_score=30.0,
            sell_extreme_zone_score=5.0,
            rsi6_cross_rsi12_down=True,
            macd_cross_down=True,
            kdj_k_cross_down=True,
            close_below_ma20=True,
        ),
    }
    loader = PriceLoader(
        {(date, "AAA"): 10.0 for date in dates[1:]},
        {(date, "AAA"): 10.0 for date in dates},
    )

    item = module.build_trade_giveback_diagnostic(
        _trade(dates[0], dates[-1]),
        ScoreSource(scores),
        loader,
        dates,
        params={**module.strategy.get_default_params(), "min_signal_hold_days": 2},
    )

    assert item.first_score_30_date == dates[1]
    assert item.first_eligible_score_30_date == dates[2]
    assert item.first_confirmation_date == dates[2]
    assert item.first_official_sell_date == dates[5]
    by_date = {state.decision_date: state for state in item.states}
    assert not by_date[dates[1]].minimum_hold_eligible
    assert by_date[dates[2]].minimum_hold_eligible
    assert by_date[dates[2]].price_confirmed
    assert by_date[dates[2]].adx_protected
    assert not by_date[dates[2]].official_signal_sell
    assert by_date[dates[5]].official_signal_sell
    assert by_date[dates[5]].rsi_down_cross
    assert by_date[dates[5]].macd_down_cross
    assert by_date[dates[5]].kdj_k_down_cross
    assert by_date[dates[5]].k_value == pytest.approx(82.0)
    assert by_date[dates[5]].sell_extreme_zone_score == pytest.approx(5.0)


@pytest.mark.parametrize("field", ["signal_date", "max_data_date"])
def test_daily_state_rejects_same_day_or_future_signal_data(field):
    module = _module()
    dates = pd.bdate_range("2019-03-01", periods=3).strftime("%Y-%m-%d").tolist()
    score = _score(dates[1])
    score[field] = dates[2]
    source = ScoreSource({(dates[2], "AAA"): score})
    loader = PriceLoader(
        {(dates[2], "AAA"): 10.0},
        {(date, "AAA"): 10.0 for date in dates},
    )

    with pytest.raises(ValueError, match="must precede decision date"):
        module.build_trade_giveback_diagnostic(
            _trade(dates[1], dates[2]),
            source,
            loader,
            dates,
        )


def test_training_runner_rejects_unapproved_data_root():
    module = _module()

    with pytest.raises(ValueError, match="approved training data root"):
        module.run_direct_kdj_buy_giveback_diagnostics(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )
