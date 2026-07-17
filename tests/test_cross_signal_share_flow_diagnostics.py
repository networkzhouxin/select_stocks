# -*- coding: utf-8 -*-
"""Tests for the training-only ETF share-flow shadow diagnostic."""

from __future__ import annotations

import math
import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _frame(dates, shares, code="159915"):
    return pd.DataFrame({
        "code": [code] * len(dates),
        "trade_date": list(dates),
        "total_share_wan": list(shares),
    })


def _write_flow_csv(root, partition, year, code, frame):
    path = pathlib.Path(root) / partition / str(year) / (str(code) + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _candidate_score(code="159915", signal_date="2020-01-09"):
    return {
        "code": code,
        "signal_date": signal_date,
        "buy_score": 70,
        "sell_score": 0,
        "reversal_score": 40,
        "volume_score": 0,
        "buy_allowed": True,
        "close_between_boll_lower_mid": True,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "ma20_slope_non_negative": True,
        "adx": 10,
        "plus_di": 20,
        "minus_di": 10,
        "atr": 0.1,
    }


class _FakeSignalSource:
    def __init__(self, score, frame_signal_date=None):
        self.base_score = score
        self.frame_signal_date = frame_signal_date or score.get("signal_date")
        self.score_calls = []
        self.frame_calls = []

    def score(self, code, current_date, return_reason=False):
        self.score_calls.append((code, current_date, return_reason))
        result = self.base_score
        return (result, None) if return_reason else result

    def load_signal_frame(self, code, current_date):
        self.frame_calls.append((code, current_date))
        return pd.DataFrame({"date": [self.frame_signal_date]}), self.frame_signal_date


class _FakeFlowLoader:
    def __init__(self, frame, actions=()):
        self.frame = frame
        self.actions = tuple(actions)
        self.history_calls = []
        self.action_calls = 0

    def load_history(self, code, signal_date):
        self.history_calls.append((code, signal_date))
        return self.frame.copy()

    def load_corporate_actions(self):
        self.action_calls += 1
        return self.actions


def test_share_flow_loader_rejects_non_approved_root(tmp_path):
    from cross_signal_strategy.research.share_flow_diagnostics import ShareFlowDataLoader

    with pytest.raises(ValueError, match="approved share-flow data root"):
        ShareFlowDataLoader(tmp_path)


def test_share_flow_loader_combines_cross_year_history_filters_future_and_copies(
    tmp_path,
    monkeypatch,
):
    import cross_signal_strategy.research.share_flow_diagnostics as module

    warmup = _frame(
        ["2018-12-26", "2018-12-27", "2018-12-28"],
        [97.0, 98.0, 99.0],
    )
    training = _frame(
        ["2019-01-02", "2019-01-03", "2019-01-04"],
        [100.0, 101.0, 999.0],
    )
    _write_flow_csv(tmp_path, "warmup", 2018, "159915", warmup)
    _write_flow_csv(tmp_path, "training", 2019, "159915", training)
    monkeypatch.setattr(module, "APPROVED_SHARE_FLOW_ROOT", tmp_path)

    reads = []
    original_read_csv = pd.read_csv

    def tracking_read_csv(*args, **kwargs):
        reads.append(pathlib.Path(args[0]).name)
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", tracking_read_csv)
    loader = module.ShareFlowDataLoader(tmp_path)

    first = loader.load_history("159915", "2019-01-03")
    first.loc[first.index[0], "total_share_wan"] = -1.0
    second = loader.load_history("159915", "2019-01-03")

    assert list(second["trade_date"].dt.strftime("%Y-%m-%d")) == [
        "2018-12-26",
        "2018-12-27",
        "2018-12-28",
        "2019-01-02",
        "2019-01-03",
    ]
    assert second["trade_date"].max() == pd.Timestamp("2019-01-03")
    assert second.iloc[0]["total_share_wan"] == pytest.approx(97.0)
    assert reads == ["159915.csv", "159915.csv"]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"code": ["159915"]}), "missing columns"),
        (_frame(["bad-date"], [100.0]), "invalid trade_date"),
        (_frame(["2022-01-04"], [100.0]), "outside approved share-flow dates"),
        (_frame(["2019-01-02"], [0.0]), "positive total_share_wan"),
        (
            _frame(["2019-01-02", "2019-01-02"], [100.0, 101.0]),
            "duplicate trade_date",
        ),
        (_frame(["2019-01-02"], [100.0], code="512100"), "code mismatch"),
    ],
)
def test_validate_share_flow_frame_rejects_bad_source_rows(frame, message):
    from cross_signal_strategy.research.share_flow_diagnostics import validate_share_frame

    with pytest.raises(ValueError, match=message):
        validate_share_frame(frame, expected_code="159915")


def test_calculate_share_flow_uses_fixed_five_observation_log_change():
    from cross_signal_strategy.research.share_flow_diagnostics import calculate_share_flow

    dates = [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
        "2020-01-09",
    ]

    positive = calculate_share_flow(
        _frame(dates, [100.0, 101.0, 103.0, 106.0, 108.0, 110.0]),
        code="159915",
        decision_date="2020-01-10",
        signal_date="2020-01-09",
        corporate_actions=(),
    )
    negative = calculate_share_flow(
        _frame(dates, [100.0, 99.0, 98.0, 96.0, 94.0, 90.0]),
        code="159915",
        decision_date="2020-01-10",
        signal_date="2020-01-09",
        corporate_actions=(),
    )
    flat = calculate_share_flow(
        _frame(dates, [100.0, 99.0, 101.0, 100.0, 102.0, 100.0]),
        code="159915",
        decision_date="2020-01-10",
        signal_date="2020-01-09",
        corporate_actions=(),
    )

    assert positive.value == pytest.approx(math.log(110.0 / 100.0))
    assert positive.baseline_date == "2020-01-02"
    assert positive.raw_state == "net_creation"
    assert positive.comparison_group == "positive"
    assert negative.value == pytest.approx(math.log(90.0 / 100.0))
    assert negative.raw_state == "net_redemption"
    assert negative.comparison_group == "non_positive"
    assert flat.value == pytest.approx(0.0)
    assert flat.raw_state == "flat"
    assert flat.comparison_group == "non_positive"


def test_calculate_share_flow_requires_t_minus_one_endpoint_and_rejects_future_rows():
    from cross_signal_strategy.research.share_flow_diagnostics import calculate_share_flow

    dates = [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
        "2020-01-09",
    ]
    shares = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    missing_endpoint = calculate_share_flow(
        _frame(dates[:-1], shares[:-1]),
        code="159915",
        decision_date="2020-01-10",
        signal_date="2020-01-09",
        corporate_actions=(),
    )

    assert missing_endpoint.value is None
    assert missing_endpoint.raw_state == "insufficient_history"
    assert missing_endpoint.comparison_group == "excluded"

    with pytest.raises(ValueError, match="after signal_date"):
        calculate_share_flow(
            _frame(dates + ["2020-01-10"], shares + [106.0]),
            code="159915",
            decision_date="2020-01-10",
            signal_date="2020-01-09",
            corporate_actions=(),
        )
    with pytest.raises(ValueError, match="strictly before decision_date"):
        calculate_share_flow(
            _frame(dates, shares),
            code="159915",
            decision_date="2020-01-09",
            signal_date="2020-01-09",
            corporate_actions=(),
        )


def test_calculate_share_flow_blocks_qdii_without_using_history():
    from cross_signal_strategy.research.share_flow_diagnostics import calculate_share_flow

    observation = calculate_share_flow(
        pd.DataFrame(),
        code="513100",
        decision_date="2020-01-10",
        signal_date="2020-01-09",
        corporate_actions=(),
    )

    assert observation.value is None
    assert observation.raw_state == "blocked_qdii"
    assert observation.comparison_group == "excluded"


def test_calculate_share_flow_neutralizes_split_crossing_and_resets_at_split_baseline():
    from cross_signal_strategy.research.share_flow_diagnostics import (
        CorporateAction,
        calculate_share_flow,
    )

    action = CorporateAction(
        code="159928",
        trade_date="2021-06-25",
        event="share_split",
    )
    crossing = calculate_share_flow(
        _frame(
            [
                "2021-06-21",
                "2021-06-22",
                "2021-06-23",
                "2021-06-24",
                "2021-06-25",
                "2021-06-28",
            ],
            [100.0, 100.0, 100.0, 100.0, 400.0, 404.0],
            code="159928",
        ),
        code="159928",
        decision_date="2021-06-29",
        signal_date="2021-06-28",
        corporate_actions=(action,),
    )
    resumed = calculate_share_flow(
        _frame(
            [
                "2021-06-25",
                "2021-06-28",
                "2021-06-29",
                "2021-06-30",
                "2021-07-01",
                "2021-07-02",
            ],
            [400.0, 404.0, 406.0, 408.0, 410.0, 412.0],
            code="159928",
        ),
        code="159928",
        decision_date="2021-07-05",
        signal_date="2021-07-02",
        corporate_actions=(action,),
    )

    assert crossing.value is None
    assert crossing.raw_state == "corporate_action"
    assert crossing.comparison_group == "excluded"
    assert resumed.value == pytest.approx(math.log(412.0 / 400.0))
    assert resumed.raw_state == "net_creation"
    assert resumed.baseline_date == "2021-06-25"


def test_share_flow_loader_reads_corporate_actions_once(tmp_path, monkeypatch):
    import cross_signal_strategy.research.share_flow_diagnostics as module

    meta = tmp_path / "meta" / "corporate_actions.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "code": ["159928"],
        "trade_date": ["2021-06-25"],
        "event": ["share_split"],
        "evidence": ["official ex-right date"],
        "required_handling": ["neutralize"],
    }).to_csv(meta, index=False)
    monkeypatch.setattr(module, "APPROVED_SHARE_FLOW_ROOT", tmp_path)
    loader = module.ShareFlowDataLoader(tmp_path)

    first = loader.load_corporate_actions()
    second = loader.load_corporate_actions()

    assert first == second == (
        module.CorporateAction("159928", "2021-06-25", "share_split"),
    )


def test_share_flow_signal_adapter_adds_only_shadow_fields_and_caches_result():
    from cross_signal_strategy.research.share_flow_diagnostics import ShareFlowSignalAdapter

    dates = [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
        "2020-01-09",
    ]
    base_score = _candidate_score()
    source = _FakeSignalSource(base_score)
    flow_loader = _FakeFlowLoader(_frame(dates, [100, 101, 102, 103, 104, 110]))
    adapter = ShareFlowSignalAdapter(source=source, flow_loader=flow_loader)

    first = adapter.score("159915", "2020-01-10")
    first["buy_score"] = -999
    second = adapter.score("159915", "2020-01-10")

    for key, value in base_score.items():
        assert second[key] == value
    assert second["share_flow_value_5"] == pytest.approx(math.log(1.1))
    assert second["share_flow_raw_state"] == "net_creation"
    assert second["share_flow_comparison_group"] == "positive"
    assert second["share_flow_signal_date"] == "2020-01-09"
    assert second["share_flow_baseline_date"] == "2020-01-02"
    assert second["share_flow_blocked"] is False
    assert base_score == _candidate_score()
    assert source.score_calls == [("159915", "2020-01-10", True)]
    assert source.frame_calls == [("159915", "2020-01-10")]
    assert flow_loader.history_calls == [("159915", "2020-01-09")]
    assert flow_loader.action_calls == 1


def test_share_flow_signal_adapter_rejects_price_signal_date_mismatch():
    from cross_signal_strategy.research.share_flow_diagnostics import ShareFlowSignalAdapter

    source = _FakeSignalSource(
        _candidate_score(signal_date="2020-01-09"),
        frame_signal_date="2020-01-08",
    )
    adapter = ShareFlowSignalAdapter(
        source=source,
        flow_loader=_FakeFlowLoader(pd.DataFrame()),
    )

    with pytest.raises(ValueError, match="signal_date does not match"):
        adapter.score("159915", "2020-01-10")


def test_share_flow_signal_adapter_blocks_qdii_without_loading_share_history():
    from cross_signal_strategy.research.share_flow_diagnostics import ShareFlowSignalAdapter

    source = _FakeSignalSource(_candidate_score(code="513100"))
    flow_loader = _FakeFlowLoader(pd.DataFrame())
    adapter = ShareFlowSignalAdapter(source=source, flow_loader=flow_loader)

    score = adapter.score("513100", "2020-01-10")

    assert score["share_flow_raw_state"] == "blocked_qdii"
    assert score["share_flow_comparison_group"] == "excluded"
    assert score["share_flow_blocked"] is True
    assert flow_loader.history_calls == []


def test_share_flow_shadow_adapter_preserves_planned_orders():
    from cross_signal_strategy.local.local_backtester import LocalBroker
    from cross_signal_strategy.local.local_order_planner import LocalCrossSignalOrderPlanner
    from cross_signal_strategy.research.share_flow_diagnostics import ShareFlowSignalAdapter

    dates = [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
        "2020-01-09",
    ]
    base_source = _FakeSignalSource(_candidate_score())
    shadow_source = _FakeSignalSource(_candidate_score())
    shadow = ShareFlowSignalAdapter(
        source=shadow_source,
        flow_loader=_FakeFlowLoader(_frame(dates, [100, 101, 102, 103, 104, 110])),
    )
    base_planner = LocalCrossSignalOrderPlanner(base_source, etf_pool=["159915"])
    shadow_planner = LocalCrossSignalOrderPlanner(shadow, etf_pool=["159915"])

    base_orders = base_planner.plan_orders(
        "2020-01-10",
        "2020-01-09",
        LocalBroker(initial_cash=20000.0),
    )
    shadow_orders = shadow_planner.plan_orders(
        "2020-01-10",
        "2020-01-09",
        LocalBroker(initial_cash=20000.0),
    )

    assert shadow_orders == base_orders
    assert all(key.startswith("share_flow_") for key in (
        set(shadow_planner.last_scores["159915"])
        - set(base_planner.last_scores["159915"])
    ))


def _closed_trade(
    buy_date,
    code,
    raw_state,
    comparison_group,
    pnl,
    return_pct,
):
    from cross_signal_strategy.research.trade_diagnostics import ClosedTradeDiagnostic

    return ClosedTradeDiagnostic(
        code=code,
        buy_date=buy_date,
        sell_date=buy_date[:4] + "-12-20",
        sell_reason="signal_sell",
        amount=100,
        buy_price=1.0,
        sell_price=1.0 + return_pct / 100.0,
        pnl=pnl,
        return_pct=return_pct,
        entry_score={
            "share_flow_raw_state": raw_state,
            "share_flow_comparison_group": comparison_group,
        },
        exit_score={},
    )


def test_share_flow_report_tracks_coverage_raw_states_and_group_statistics():
    from cross_signal_strategy.research.share_flow_diagnostics import build_share_flow_report

    trades = [
        _closed_trade(
            "2019-02-01", "159915", "net_creation", "positive", 100.0, 10.0
        ),
        _closed_trade(
            "2019-03-01", "512100", "net_redemption", "non_positive", -50.0, -5.0
        ),
        _closed_trade(
            "2019-04-01", "159928", "corporate_action", "excluded", 20.0, 2.0
        ),
        _closed_trade(
            "2019-05-01", "513100", "blocked_qdii", "excluded", 30.0, 3.0
        ),
    ]

    report = build_share_flow_report(trades)

    assert report.coverage.total_closed_buys == 4
    assert report.coverage.eligible_domestic_closed_buys == 3
    assert report.coverage.comparable_closed_buys == 2
    assert report.coverage.coverage_rate_all == pytest.approx(0.5)
    assert report.coverage.coverage_rate_eligible == pytest.approx(2 / 3)
    assert report.raw_state_counts == {
        "blocked_qdii": 1,
        "corporate_action": 1,
        "net_creation": 1,
        "net_redemption": 1,
    }
    assert report.by_group["positive"].closed_trades == 1
    assert report.by_group["positive"].wins == 1
    assert report.by_group["positive"].average_return == pytest.approx(0.10)
    assert report.by_group["non_positive"].losses == 1
    assert report.by_group["non_positive"].realized_pnl == pytest.approx(-50.0)
    assert report.by_year_group["2019:positive"].closed_trades == 1
    assert report.gate.passed is False


def _gate_stats(closed_trades, wins, average_return):
    from cross_signal_strategy.research.share_flow_diagnostics import ShareFlowStats

    return ShareFlowStats(
        closed_trades=closed_trades,
        wins=wins,
        losses=closed_trades - wins,
        average_return=average_return,
    )


def test_share_flow_gate_passes_only_for_same_direction_annual_dominance():
    from cross_signal_strategy.research.share_flow_diagnostics import evaluate_share_flow_gate

    positive = {
        2019: _gate_stats(2, 2, 0.10),
        2020: _gate_stats(2, 2, 0.08),
        2021: _gate_stats(2, 1, 0.04),
    }
    non_positive = {
        2019: _gate_stats(2, 1, 0.01),
        2020: _gate_stats(2, 1, -0.02),
        2021: _gate_stats(2, 0, -0.03),
    }

    decision = evaluate_share_flow_gate(positive, non_positive)

    assert decision.passed is True
    assert decision.dominant_group == "positive"
    assert decision.reasons == ()


@pytest.mark.parametrize("failure_mode", ["sparse", "reversed_year", "tie"])
def test_share_flow_gate_rejects_sparse_inconsistent_or_tied_evidence(failure_mode):
    from cross_signal_strategy.research.share_flow_diagnostics import evaluate_share_flow_gate

    positive = {
        2019: _gate_stats(2, 2, 0.10),
        2020: _gate_stats(2, 2, 0.08),
        2021: _gate_stats(2, 2, 0.06),
    }
    non_positive = {
        2019: _gate_stats(2, 1, 0.01),
        2020: _gate_stats(2, 1, 0.00),
        2021: _gate_stats(2, 1, -0.01),
    }
    if failure_mode == "sparse":
        positive[2021] = _gate_stats(1, 1, 0.06)
    elif failure_mode == "reversed_year":
        non_positive[2020] = _gate_stats(2, 2, 0.12)
    else:
        non_positive[2020] = _gate_stats(2, 2, 0.08)

    decision = evaluate_share_flow_gate(positive, non_positive)

    assert decision.passed is False
    assert decision.reasons


def test_share_flow_report_rejects_trades_outside_training_window():
    from cross_signal_strategy.research.share_flow_diagnostics import build_share_flow_report

    trade = _closed_trade(
        "2022-01-04", "159915", "net_creation", "positive", 100.0, 10.0
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_share_flow_report([trade])


def test_training_share_flow_runner_reuses_official_diagnostic_replay(monkeypatch):
    import cross_signal_strategy.research.share_flow_diagnostics as module

    trade = _closed_trade(
        "2019-02-01", "159915", "net_creation", "positive", 100.0, 10.0
    )
    price_loader = object()
    flow_loader = object()
    source = object()
    calls = []

    class FakePlanner:
        def __init__(self, adapter, trade_dates):
            calls.append(("planner", adapter.source, adapter.flow_loader, trade_dates))
            self.entry_score_snapshots = {("2019-02-01", "159915"): {}}
            self.exit_score_snapshots = {}

        def plan_orders(self, *args, **kwargs):
            return []

    class FakeEngine:
        def __init__(self, loader, initial_cash):
            calls.append(("engine", loader, initial_cash))

        def run(self, trade_dates, planner):
            calls.append(("run", trade_dates, planner.__self__.__class__.__name__))
            return ["day-result"]

    monkeypatch.setattr(
        module,
        "get_training_trade_dates",
        lambda loader: ["2019-02-01", "2019-02-20"],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "build_training_signal_adapter",
        lambda loader: source,
        raising=False,
    )
    monkeypatch.setattr(module, "DiagnosticOrderPlanner", FakePlanner, raising=False)
    monkeypatch.setattr(module, "LocalBacktestEngine", FakeEngine, raising=False)
    monkeypatch.setattr(
        module,
        "build_closed_trade_diagnostics",
        lambda results, entries, exits: (
            calls.append(("closed", results, entries, exits)) or [trade]
        ),
        raising=False,
    )

    report = module.run_training_share_flow_observation(
        loader=price_loader,
        flow_loader=flow_loader,
        initial_cash=20000.0,
    )

    assert report.coverage.total_closed_buys == 1
    assert calls[0] == (
        "planner",
        source,
        flow_loader,
        ["2019-02-01", "2019-02-20"],
    )
    assert ("engine", price_loader, 20000.0) in calls
    assert ("run", ["2019-02-01", "2019-02-20"], "FakePlanner") in calls


def test_share_flow_report_formatter_states_locked_scope_and_gate_reasons():
    from cross_signal_strategy.research.share_flow_diagnostics import (
        build_share_flow_report,
        format_share_flow_report,
    )

    report = build_share_flow_report([
        _closed_trade(
            "2019-02-01", "159915", "net_creation", "positive", 100.0, 10.0
        )
    ])

    text = format_share_flow_report(report)

    assert "observation-only" in text
    assert "2019-2021" in text
    assert "log(shares[T-1]/shares[T-6])" in text
    assert "159915,512100,159928,518880,159985" in text
    assert "513100,513500,513880,513050" in text
    assert "COVERAGE" in text
    assert "GROUP positive" in text
    assert "OBSERVATION_GATE passed=False" in text
    assert "GATE_REASON" in text
