# -*- coding: utf-8 -*-
"""Tests for the isolated dimension-capped candidate order planner."""

from copy import deepcopy
from dataclasses import replace
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cross_signal_strategy.local.local_backtester import LocalBroker, OrderResult, Position


class FakeSignalAdapter:
    def __init__(self, scores):
        self.scores = scores

    def score(self, code, current_date, return_reason=False):
        value = deepcopy(self.scores.get(code))
        reason = None if value is not None else "no_data"
        return (value, reason) if return_reason else value


def _training_module():
    from cross_signal_strategy.research import dimension_capped_training_ab
    return dimension_capped_training_ab


def _real_local_adapter():
    from cross_signal_strategy.local.local_data_loader import (
        CrossSignalTrainingDataLoader,
    )
    from cross_signal_strategy.local_training_run import build_training_signal_adapter

    return build_training_signal_adapter(CrossSignalTrainingDataLoader())


def _candidate_score(code, **overrides):
    values = {
        "code": code,
        "buy_allowed": True,
        "buy_score": 40.0,
        "reversal_score": 18.0,
        "location_score": 10.0,
        "trend_score": 12.0,
        "volume_rank_score": 0.0,
        "sell_score": 0.0,
        "sell_weakness_score": 0.0,
        "sell_damage_score": 0.0,
        "close_far_above_ma20": False,
        "downside_continuation": False,
        "weak_repair_blocked": False,
        "adx": 10.0,
        "plus_di": 20.0,
        "minus_di": 10.0,
        "ma20_slope_non_negative": True,
        "atr": 0.1,
    }
    values.update(overrides)
    return values


def _six_trade_dates():
    return [
        "2019-07-01", "2019-07-02", "2019-07-03",
        "2019-07-04", "2019-07-05", "2019-07-08",
    ]


def _held_severe_sell_fixture(buy_date, params=None):
    planner = _training_module().DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score(
                "510300", buy_score=10.0,
                sell_score=24.0, sell_weakness_score=6.0, sell_damage_score=18.0,
            )
        }),
        etf_pool=["510300"],
        params=params,
        buy_dates={"510300": buy_date},
        trade_dates=_six_trade_dates(),
    )
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)
    return planner, broker


def _held_atr_stop_fixture(buy_date):
    planner = _training_module().DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score("510300", buy_score=44.0),
            "159915": _candidate_score("159915", buy_score=42.0),
        }),
        etf_pool=["510300", "159915"],
        buy_dates={"510300": buy_date},
        trade_dates=_six_trade_dates(),
    )
    planner.highest_since_buy["510300"] = 10.0
    planner.entry_atr["510300"] = 1.0
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 9.0)
    return planner, broker


def test_dimension_capped_training_configuration_is_exact():
    config = _training_module().dimension_capped_training_config()
    assert config.candidate_name == "cross-v0.4.0-dimension-capped-candidate"
    assert config.training_start == "2019-01-01"
    assert config.training_end == "2021-12-31"
    assert config.initial_cash == pytest.approx(20000.0)
    assert config.execution_time == "09:35"
    assert config.buy_threshold == pytest.approx(40.0)
    assert config.ordinary_sell_threshold == pytest.approx(24.0)
    assert config.min_signal_hold_days == 5
    assert config.max_hold == 3
    assert config.base_ratio == pytest.approx(0.95)
    assert config.candidate_variants == 1
    assert config.training_root == pathlib.Path(
        r"G:\financial\history_data\cross_signal_train_2019_2021"
    )
    assert config.warmup_root == pathlib.Path(
        r"G:\financial\history_data\cross_signal_warmup_2018"
    )


def _performance(**overrides):
    module = _training_module()
    values = {
        "total_return": 1.0,
        "annualized_return": 0.25,
        "max_drawdown": 0.10,
        "sharpe_ratio": 2.0,
        "sortino_ratio": 3.0,
        "win_rate": 0.55,
        "profit_loss_ratio": 4.0,
        "buy_count": 100,
        "sell_count": 100,
        "closed_trade_count": 100,
        "annual_returns": {2019: 0.20, 2020: 0.30, 2021: 0.15},
    }
    values.update(overrides)
    return module.DimensionCappedPerformance(**values)


def _passing_inputs():
    module = _training_module()
    return module.DimensionCappedGateInputs(
        baseline=_performance(),
        candidate=_performance(
            total_return=0.96,
            annualized_return=0.24,
            max_drawdown=0.104,
            sharpe_ratio=1.90,
            sortino_ratio=2.85,
            win_rate=0.56,
            profit_loss_ratio=3.80,
            closed_trade_count=80,
            annual_returns={2019: 0.19, 2020: 0.29, 2021: 0.14},
        ),
        baseline_double_friction=_performance(total_return=0.80, win_rate=0.50),
        candidate_double_friction=_performance(total_return=0.76, win_rate=0.50),
        changed_order_days=10,
        changed_days_by_year={2019: 4, 2020: 3, 2021: 3},
    )


def _failed_inputs(name):
    item = _passing_inputs()
    if name == "changed_total_9":
        return replace(
            item,
            changed_order_days=9,
            changed_days_by_year={2019: 3, 2020: 3, 2021: 3},
        )
    if name == "changed_2019_1":
        return replace(
            item,
            changed_order_days=11,
            changed_days_by_year={2019: 1, 2020: 5, 2021: 5},
        )
    if name == "closed_trade_79pct":
        return replace(item, candidate=replace(item.candidate, closed_trade_count=79))
    if name == "win_rate_equal":
        return replace(item, candidate=replace(item.candidate, win_rate=0.55))
    if name == "return_949pct":
        return replace(item, candidate=replace(item.candidate, total_return=0.949))
    if name == "drawdown_plus_051pp":
        return replace(item, candidate=replace(item.candidate, max_drawdown=0.1051))
    if name == "sharpe_949pct":
        return replace(item, candidate=replace(item.candidate, sharpe_ratio=1.898))
    if name == "sortino_949pct":
        return replace(item, candidate=replace(item.candidate, sortino_ratio=2.847))
    if name == "pl_949pct":
        return replace(item, candidate=replace(item.candidate, profit_loss_ratio=3.796))
    if name == "positive_year_to_zero":
        return replace(
            item,
            candidate=replace(
                item.candidate,
                annual_returns={2019: 0.19, 2020: 0.29, 2021: 0.0},
            ),
        )
    if name == "x2_return_949pct":
        return replace(
            item,
            candidate_double_friction=replace(
                item.candidate_double_friction,
                total_return=0.759,
            ),
        )
    if name == "x2_win_lower":
        return replace(
            item,
            candidate_double_friction=replace(
                item.candidate_double_friction,
                win_rate=0.499,
            ),
        )
    raise AssertionError("unknown mutation: %s" % name)


def test_training_gate_accepts_every_frozen_boundary():
    decision = _training_module().evaluate_dimension_capped_gate(_passing_inputs())
    assert decision == _training_module().DimensionCappedGateDecision(True, ())


@pytest.mark.parametrize("mutation, reason", [
    ("changed_total_9", "fewer than 10 changed filled-order days"),
    ("changed_2019_1", "2019 has fewer than 2 changed filled-order days"),
    ("closed_trade_79pct", "candidate retains fewer than 80% of closed trades"),
    ("win_rate_equal", "candidate win rate does not strictly improve"),
    ("return_949pct", "candidate retains less than 95% of baseline return"),
    ("drawdown_plus_051pp", "candidate maximum drawdown worsens by more than 0.5pp"),
    ("sharpe_949pct", "candidate Sharpe ratio retains less than 95%"),
    ("sortino_949pct", "candidate Sortino ratio retains less than 95%"),
    ("pl_949pct", "candidate profit/loss ratio retains less than 95%"),
    ("positive_year_to_zero", "a positive baseline year turns non-positive"),
    ("x2_return_949pct", "doubled-friction return retains less than 95%"),
    ("x2_win_lower", "doubled-friction win rate is below baseline"),
])
def test_training_gate_rejects_each_frozen_failure(mutation, reason):
    comparison = _failed_inputs(mutation)
    decision = _training_module().evaluate_dimension_capped_gate(comparison)
    assert not decision.passed
    assert reason in decision.reasons


@pytest.mark.parametrize("field, label", [
    ("sharpe_ratio", "Sharpe ratio"),
    ("sortino_ratio", "Sortino ratio"),
    ("profit_loss_ratio", "profit/loss ratio"),
])
def test_training_gate_reports_missing_candidate_ratio(field, label):
    inputs = _passing_inputs()
    inputs = replace(inputs, candidate=replace(inputs.candidate, **{field: None}))
    decision = _training_module().evaluate_dimension_capped_gate(inputs)
    assert not decision.passed
    assert f"candidate {label} metric is missing" in decision.reasons


@pytest.mark.parametrize("field", [
    "sharpe_ratio", "sortino_ratio", "profit_loss_ratio",
])
def test_training_gate_allows_mutually_undefined_ratio(field):
    inputs = _passing_inputs()
    inputs = replace(
        inputs,
        baseline=replace(inputs.baseline, **{field: None}),
        candidate=replace(inputs.candidate, **{field: None}),
    )
    assert _training_module().evaluate_dimension_capped_gate(inputs).passed


def _replay_day(date, total_value, orders=()):
    return SimpleNamespace(
        date=date,
        total_value=total_value,
        orders=list(orders),
        positions={},
        marks={},
    )


def _filled_order(date, code, amount, reason):
    return OrderResult(
        code=code,
        amount_delta=amount,
        exec_price=10.0,
        commission=5.0,
        side_time=f"{date} 09:35",
        filled=True,
        reason=reason,
    )


def test_filled_order_materiality_excludes_reason_and_includes_absolute_amount():
    module = _training_module()
    baseline = _replay_day(
        "2019-07-01",
        20000.0,
        [_filled_order("2019-07-01", "510300.XSHG", 600, "official_buy")],
    )
    renamed = _replay_day(
        "2019-07-01",
        20000.0,
        [_filled_order("2019-07-01", "510300", 600, "dimension_capped_buy")],
    )
    resized = _replay_day(
        "2019-07-01",
        20000.0,
        [_filled_order("2019-07-01", "510300", 500, "dimension_capped_buy")],
    )

    assert module._filled_order_signature(baseline) == (
        ("2019-07-01", "510300", "buy", 600),
    )
    assert module._filled_order_changes([baseline], [renamed]) == (
        0,
        {2019: 0, 2020: 0, 2021: 0},
    )
    assert module._filled_order_changes([baseline], [resized]) == (
        1,
        {2019: 1, 2020: 0, 2021: 0},
    )


def test_filled_order_comparison_requires_identical_training_dates():
    module = _training_module()
    with pytest.raises(ValueError, match="identical trading dates"):
        module._filled_order_changes(
            [_replay_day("2019-07-01", 20000.0)],
            [_replay_day("2019-07-02", 20000.0)],
        )
    with pytest.raises(ValueError, match="outside 2019-2021"):
        module._assert_training_dates(["2022-01-04"])


def test_performance_extraction_uses_baseline_report_and_chained_calendar_years():
    module = _training_module()
    days = [
        _replay_day(
            "2019-12-31",
            110.0,
            [_filled_order("2019-12-31", "510300", 100, "buy")],
        ),
        _replay_day(
            "2020-12-31",
            99.0,
            [_filled_order("2020-12-31", "510300", -100, "sell")],
        ),
        _replay_day("2021-12-31", 118.8),
    ]
    performance = module._performance(days, initial_cash=100.0)

    assert performance.total_return == pytest.approx(0.188)
    assert performance.buy_count == 1
    assert performance.sell_count == 1
    assert performance.closed_trade_count == 1
    assert performance.annual_returns == pytest.approx({
        2019: 0.10,
        2020: -0.10,
        2021: 0.20,
    })


def test_runner_rejects_unapproved_root_and_preserves_t_minus_one_metadata():
    module = _training_module()
    with pytest.raises(ValueError, match="approved training data root"):
        module.run_dimension_capped_training_ab(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )

    adapter = module.DimensionCappedScoreAdapter(_real_local_adapter())
    score, reason = adapter.score("510300", "2019-07-01", return_reason=True)
    assert reason is None
    assert score["signal_date"] == "2019-06-28"
    assert score["max_data_date"] == "2019-06-28"


def test_runner_builds_one_official_cache_and_four_independent_arms(monkeypatch):
    module = _training_module()
    trade_dates = ["2019-01-02", "2020-01-02", "2021-01-04"]
    official_source = object()
    cached_source = object()
    cache_calls = []
    arm_calls = []
    official_defaults = {
        "buy_threshold": 45.0,
        "sell_threshold": 28.0,
        "min_signal_hold_days": 3,
        "max_hold": 3,
        "base_ratio": 0.95,
        "stop_atr_multiplier": 2.5,
    }

    class FakePrecomputedSignalAdapter:
        @classmethod
        def from_source(cls, source, trade_dates, codes):
            cache_calls.append((source, list(trade_dates), list(codes)))
            return cached_source

    def fake_run_arm(
        loader,
        signal_adapter,
        planner_class,
        params,
        pool,
        dates,
        initial_cash,
        friction,
    ):
        planner = SimpleNamespace(
            decision_audits=["nominal_candidate_audit"]
            if planner_class is module.DimensionCappedOrderPlanner and friction is None
            else ["ignored_audit"]
        )
        arm_calls.append((
            signal_adapter,
            planner_class,
            dict(params),
            list(pool),
            list(dates),
            initial_cash,
            friction,
            planner,
        ))
        return [_replay_day(date, initial_cash) for date in dates], planner

    monkeypatch.setattr(module, "get_training_trade_dates", lambda loader: trade_dates)
    monkeypatch.setattr(
        module,
        "build_training_signal_adapter",
        lambda loader, warmup_root: official_source,
    )
    monkeypatch.setattr(module, "PrecomputedSignalAdapter", FakePrecomputedSignalAdapter)
    monkeypatch.setattr(module, "_run_arm", fake_run_arm)
    monkeypatch.setattr(module, "_build_gate_inputs", lambda *args: _passing_inputs())
    monkeypatch.setattr(
        module.strategy,
        "get_default_params",
        lambda: deepcopy(official_defaults),
    )
    monkeypatch.setattr(
        module.strategy,
        "get_default_etf_pool",
        lambda: ["510300.XSHG", "513100.XSHG"],
    )

    report = module.run_dimension_capped_training_ab(
        loader=SimpleNamespace(root=module.APPROVED_TRAINING_ROOT),
        initial_cash=12345.0,
    )

    assert cache_calls == [
        (official_source, trade_dates, ["510300", "513100"]),
    ]
    assert len(arm_calls) == 4
    assert arm_calls[0][0] is cached_source
    assert isinstance(arm_calls[1][0], module.DimensionCappedScoreAdapter)
    assert arm_calls[1][0] is arm_calls[3][0]
    assert arm_calls[1][0].source is cached_source
    assert [call[1] for call in arm_calls] == [
        module.LocalCrossSignalOrderPlanner,
        module.DimensionCappedOrderPlanner,
        module.LocalCrossSignalOrderPlanner,
        module.DimensionCappedOrderPlanner,
    ]
    assert [call[6] for call in arm_calls] == [
        None, None, module.DOUBLE_FRICTION, module.DOUBLE_FRICTION,
    ]
    assert arm_calls[0][2] == official_defaults
    assert arm_calls[1][2] == {
        **official_defaults,
        "buy_threshold": 40.0,
        "sell_threshold": 24.0,
        "min_signal_hold_days": 5,
    }
    assert official_defaults["buy_threshold"] == 45.0
    assert report.decision_audits == ("nominal_candidate_audit",)


def test_run_arm_applies_doubled_friction_to_an_independent_engine():
    module = _training_module()

    class OneDayLoader:
        def get_minute_bar(self, code, date, time):
            return {"close": 10.0, "volume": 1000.0}

        def load_daily_frame(self, code, end_date):
            return pd.DataFrame([{"date": end_date, "close": 10.0}])

    class BuyOncePlanner:
        def __init__(self, adapter, etf_pool, params, trade_dates):
            self.adapter = adapter
            self.etf_pool = list(etf_pool)
            self.params = dict(params)
            self.trade_dates = list(trade_dates)

        def plan_orders(self, current_date, previous_date, broker, current_prices=None):
            if broker.positions:
                return []
            return [{"code": "510300", "target_value": 10000.0, "reason": "buy"}]

    arguments = (
        OneDayLoader(), object(), BuyOncePlanner,
        {"max_hold": 3}, ["510300"], ["2019-01-02"], 20000.0,
    )
    nominal_days, nominal_planner = module._run_arm(*arguments, None)
    doubled_days, doubled_planner = module._run_arm(
        *arguments,
        module.DOUBLE_FRICTION,
    )

    assert nominal_planner is not doubled_planner
    assert nominal_days[-1].total_value == pytest.approx(19985.0)
    assert doubled_days[-1].total_value == pytest.approx(19970.0)


def test_gate_inputs_aggregate_four_identical_date_replays():
    module = _training_module()
    baseline = [_replay_day(
        "2019-07-01",
        110.0,
        [_filled_order("2019-07-01", "510300", 100, "official_buy")],
    )]
    candidate = [_replay_day(
        "2019-07-01",
        108.0,
        [_filled_order("2019-07-01", "510300", 200, "candidate_buy")],
    )]
    baseline_x2 = [_replay_day("2019-07-01", 105.0)]
    candidate_x2 = [_replay_day("2019-07-01", 104.0)]

    inputs = module._build_gate_inputs(
        baseline,
        candidate,
        baseline_x2,
        candidate_x2,
        100.0,
    )

    assert inputs.changed_order_days == 1
    assert inputs.changed_days_by_year == {2019: 1, 2020: 0, 2021: 0}
    assert inputs.baseline.total_return == pytest.approx(0.10)
    assert inputs.candidate.total_return == pytest.approx(0.08)
    assert inputs.baseline_double_friction.total_return == pytest.approx(0.05)
    assert inputs.candidate_double_friction.total_return == pytest.approx(0.04)
    with pytest.raises(ValueError, match="identical trading dates"):
        module._build_gate_inputs(
            baseline,
            candidate,
            baseline_x2,
            [_replay_day("2019-07-02", 104.0)],
            100.0,
        )


def test_planner_audits_buy_adx_protection_severe_sell_and_atr_stop_causally():
    module = _training_module()
    metadata = {
        "signal_date": "2019-07-05",
        "max_data_date": "2019-07-05",
        "k": 50.0,
        "macd_cross_up": False,
        "macd_cross_down": False,
        "raw_buy_reversal_contributions": {
            "rsi_group": 12.0,
            "kdj_group": 6.0,
        },
        "raw_location_contributions": {"between_boll_lower_mid": 10.0},
        "raw_trend_contributions": {"ma5_gt_ma10": 6.0},
        "raw_sell_weakness_contributions": {"rsi_group": 0.0},
        "raw_sell_damage_contributions": {"below_ma20": 0.0},
    }
    planner = module.DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score("510300", **metadata),
            "513100": _candidate_score(
                "513100",
                **metadata,
                buy_score=10.0,
                sell_score=24.0,
                sell_weakness_score=10.0,
                sell_damage_score=8.0,
                adx=30.0,
            ),
            "513500": _candidate_score(
                "513500",
                **metadata,
                buy_score=10.0,
                sell_score=24.0,
                sell_weakness_score=6.0,
                sell_damage_score=18.0,
                adx=30.0,
            ),
            "159915": _candidate_score("159915", **metadata, buy_score=44.0),
        }),
        etf_pool=["510300", "513100", "513500", "159915"],
        buy_dates={
            "513100": "2019-06-20",
            "513500": "2019-06-20",
            "159915": "2019-07-08",
        },
        trade_dates=_six_trade_dates(),
    )
    planner.highest_since_buy["159915"] = 10.0
    planner.entry_atr["159915"] = 1.0
    broker = LocalBroker(initial_cash=20000.0)
    for code in ("513100", "513500", "159915"):
        broker.positions[code] = Position(code, 100, 9.0)

    orders = planner.plan_orders(
        "2019-07-08",
        "2019-07-05",
        broker,
        current_prices={"159915": 8.0},
    )
    reasons_by_code = {order["code"]: order["reason"] for order in orders}
    audits = {audit.code: audit for audit in planner.decision_audits}

    assert len(audits) == 4
    assert reasons_by_code["510300"] == "dimension_capped_buy"
    assert audits["510300"].order_reason == "dimension_capped_buy"
    assert ("buy_rsi_group", 12.0) in audits["510300"].raw_contributions
    assert audits["513100"].adx_protected
    assert audits["513100"].order_reason is None
    assert reasons_by_code["513500"] == "dimension_capped_signal_sell"
    assert audits["513500"].order_reason == "dimension_capped_signal_sell"
    assert audits["159915"].atr_stop
    assert audits["159915"].min_hold_blocked
    assert audits["159915"].order_reason == "atr_stop"
    for audit in audits.values():
        assert audit.decision_date == "2019-07-08"
        assert audit.signal_date == "2019-07-05"
        assert audit.max_data_date == "2019-07-05"
        assert not hasattr(audit, "mfe")
        assert not hasattr(audit, "mae")
        assert not hasattr(audit, "post_sell_return")
        assert not hasattr(audit, "gate_result")


def _comparison_report(passed):
    module = _training_module()
    audit = module.DimensionCappedDecisionAudit(
        decision_date="2019-07-01",
        signal_date="2019-06-28",
        max_data_date="2019-06-28",
        code="510300",
        held=False,
        buy_reversal=18.0,
        buy_location=10.0,
        buy_trend=12.0,
        volume_rank=6.0,
        buy_total=40.0,
        sell_weakness=0.0,
        sell_damage=0.0,
        sell_total=0.0,
        kdj_tier="neutral",
        macd_confirmation="none",
        raw_contributions=(("buy_rsi_group", 12.0), ("buy_kdj_group", 6.0)),
        adx_protected=False,
        atr_stop=False,
        min_hold_blocked=False,
        hard_block_reasons=(),
        order_reason="dimension_capped_buy",
    )
    reasons = () if passed else (
        "candidate win rate does not strictly improve",
        "candidate Sortino ratio metric is missing",
    )
    return module.DimensionCappedComparisonReport(
        config=module.dimension_capped_training_config(),
        inputs=_passing_inputs(),
        gate=module.DimensionCappedGateDecision(passed, reasons),
        decision_audits=(audit,),
    )


def test_formatter_emits_one_terminal_action_and_causal_audit():
    module = _training_module()
    passed = module.format_dimension_capped_comparison(_comparison_report(True))
    failed = module.format_dimension_capped_comparison(_comparison_report(False))
    assert "ELIGIBLE_FOR_JOINQUANT_PLAN" in passed
    assert "STOP" not in passed
    assert "STOP" in failed
    assert "ELIGIBLE_FOR_JOINQUANT_PLAN" not in failed
    for token in (
        "2019", "2020", "2021", "BASELINE", "CANDIDATE",
        "BASELINE_X2_FRICTION", "CANDIDATE_X2_FRICTION",
        "2019-06-28", "510300", "buy_rsi_group", "dimension_capped_buy",
        "authority=local_screen_only",
    ):
        assert token in passed


def test_formatter_includes_frozen_gate_evidence_and_every_causal_audit_field():
    module = _training_module()
    report = _comparison_report(False)
    rendered = module.format_dimension_capped_comparison(report)

    for token in (
        "cross-v0.4.0-dimension-capped-candidate",
        "hypothesis=",
        "changed_order_days=10",
        "changed_days_by_year=2019:4,2020:3,2021:3",
        "closed_trade_retention=80.00%",
        "candidate win rate does not strictly improve",
        "candidate Sortino ratio metric is missing",
        "future_function_audit=T-1_only",
        "decision_date=2019-07-01",
        "signal_date=2019-06-28",
        "max_data_date=2019-06-28",
        "code=510300",
        "held=false",
        "buy_reversal=18.000",
        "buy_location=10.000",
        "buy_trend=12.000",
        "volume_rank=6.000",
        "buy_total=40.000",
        "sell_weakness=0.000",
        "sell_damage=0.000",
        "sell_total=0.000",
        "kdj_tier=neutral",
        "macd_confirmation=none",
        "buy_rsi_group:12.000",
        "buy_kdj_group:6.000",
        "adx_protected=false",
        "atr_stop=false",
        "min_hold_blocked=false",
        "hard_block_reasons=none",
        "order_reason=dimension_capped_buy",
    ):
        assert token in rendered
    assert rendered.count("terminal_action=") == 1
    assert rendered.rstrip().endswith("terminal_action=STOP")
    for forbidden in ("MFE", "MAE", "post_sell_return", "gate_result"):
        assert forbidden not in rendered


def test_formatter_reports_mutually_undefined_ratios_as_not_applicable():
    module = _training_module()
    inputs = _passing_inputs()
    inputs = replace(
        inputs,
        baseline=replace(
            inputs.baseline,
            sharpe_ratio=None,
            sortino_ratio=None,
            profit_loss_ratio=None,
        ),
        candidate=replace(
            inputs.candidate,
            sharpe_ratio=None,
            sortino_ratio=None,
            profit_loss_ratio=None,
        ),
    )
    report = replace(_comparison_report(True), inputs=inputs)
    rendered = module.format_dimension_capped_comparison(report)
    assert rendered.count("not_applicable") >= 6


def test_candidate_planner_sells_first_then_buys_ranked_empty_slots():
    module = _training_module()
    planner = module.DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score(
                "510300", buy_score=10, sell_score=24,
                sell_weakness_score=10, sell_damage_score=14,
            ),
            "513100": _candidate_score("513100", buy_score=44, location_score=10),
            "159915": _candidate_score("159915", buy_score=42, location_score=8, volume_rank_score=6),
        }),
        etf_pool=["510300", "513100", "159915"],
        buy_dates={"510300": "2019-06-20"},
        trade_dates=_six_trade_dates(),
    )
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)

    orders = planner.plan_orders("2019-07-08", "2019-07-05", broker)
    assert orders[0] == {"code": "510300", "target_value": 0.0, "reason": "dimension_capped_signal_sell"}
    assert [item["code"] for item in orders[1:]] == ["513100", "159915"]


def test_candidate_target_is_equal_weight_and_volume_only_breaks_rank_ties():
    planner = _training_module().DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "159915": _candidate_score("159915", volume_rank_score=10.0),
            "513100": _candidate_score("513100", volume_rank_score=0.0),
        }),
        etf_pool=["159915", "513100"],
        trade_dates=_six_trade_dates(),
    )
    broker = LocalBroker(initial_cash=20000.0)
    orders = planner.plan_orders("2019-07-01", None, broker)
    assert [item["code"] for item in orders] == ["159915", "513100"]
    assert orders[0]["target_value"] == pytest.approx(20000.0 * 0.95 / 3)
    assert orders[1]["target_value"] == pytest.approx(20000.0 * 0.95 / 3)


def test_candidate_signal_sell_waits_five_trading_days():
    planner, broker = _held_severe_sell_fixture(buy_date="2019-07-01")
    assert planner.plan_orders("2019-07-05", "2019-07-04", broker) == []
    assert planner.plan_orders("2019-07-08", "2019-07-05", broker)[0]["reason"] == "dimension_capped_signal_sell"


def test_candidate_freezes_five_day_signal_hold_when_params_request_one_day():
    module = _training_module()
    params = module.strategy.get_default_params()
    params["min_signal_hold_days"] = 1
    planner, broker = _held_severe_sell_fixture("2019-07-01", params=params)

    assert planner.plan_orders("2019-07-02", "2019-07-01", broker) == []


def test_candidate_atr_stop_ignores_five_day_signal_hold_and_blocks_same_day_rebuy():
    planner, broker = _held_atr_stop_fixture(buy_date="2019-07-01")
    orders = planner.plan_orders(
        "2019-07-02", "2019-07-01", broker,
        current_prices={"510300": 8.0, "159915": 4.0},
    )
    assert orders[0] == {"code": "510300", "target_value": 0.0, "reason": "atr_stop"}
    assert "510300" not in [item["code"] for item in orders[1:]]
