# -*- coding: utf-8 -*-
"""Tests for the pre-registered extreme-lag attribution."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _event(date, side, code, amount):
    from cross_signal_strategy.research.order_path_diagnostics import OrderPathEvent

    return OrderPathEvent(date=date, side=side, code=code, amount=amount)


def _closed_trade(**overrides):
    from cross_signal_strategy.research.trade_diagnostics import ClosedTradeDiagnostic

    values = {
        "code": "510300",
        "buy_date": "2020-01-07",
        "sell_date": "2020-01-20",
        "sell_reason": "signal_sell",
        "amount": 1000,
        "buy_price": 10.0,
        "sell_price": 10.5,
        "pnl": 490.0,
        "return_pct": 5.0,
        "entry_score": {
            "signal_date": "2020-01-06",
            "max_data_date": "2020-01-06",
            "atr": 0.5,
            "reversal_score": 28,
            "rsi6_cross_rsi12_up": True,
            "rsi6_cross_rsi12_up_age": 2,
            "rsi6_cross_rsi24_up": False,
            "rsi6_cross_rsi24_down": False,
            "macd_cross_up": True,
            "macd_cross_up_age": 0,
            "kdj_k_cross_up": True,
            "kdj_k_cross_up_age": 1,
            "kdj_j_cross_up": False,
        },
        "exit_score": {},
    }
    values.update(overrides)
    return ClosedTradeDiagnostic(**values)


def _signal_frame(last_date="2020-01-06"):
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", last_date])
    return pd.DataFrame({"date": dates, "close": [9.0, 9.2, 9.6, 9.8]})


def test_official_fill_path_requires_exact_date_side_code_and_amount():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        assert_official_fill_path,
    )

    expected = [
        _event("2019-01-04", "BUY", "510300", 1200),
        _event("2019-01-11", "SELL", "510300", 1200),
    ]

    evidence = assert_official_fill_path(expected, list(expected))

    assert evidence.status == "aligned"
    assert evidence.expected_count == 2
    assert evidence.actual_count == 2

    with pytest.raises(ValueError, match="amount mismatch"):
        assert_official_fill_path(
            expected,
            [expected[0], _event("2019-01-11", "SELL", "510300", 1100)],
        )


def test_official_fill_path_rejects_missing_expected_evidence_and_key_mismatch():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        assert_official_fill_path,
    )

    with pytest.raises(ValueError, match="official JoinQuant filled path is required"):
        assert_official_fill_path([], [_event("2019-01-04", "BUY", "510300", 1200)])

    with pytest.raises(ValueError, match="order path mismatch"):
        assert_official_fill_path(
            [_event("2019-01-04", "BUY", "510300", 1200)],
            [_event("2019-01-04", "BUY", "159915", 1200)],
        )

    with pytest.raises(ValueError, match="2019-2021 training window"):
        assert_official_fill_path(
            [_event("2022-01-04", "BUY", "510300", 1200)],
            [_event("2022-01-04", "BUY", "510300", 1200)],
        )


def test_training_episode_dates_reject_non_training_or_reversed_dates():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        assert_training_episode_dates,
    )

    assert_training_episode_dates("2019-01-02", "2021-12-31")

    with pytest.raises(ValueError, match="2019-2021 training window"):
        assert_training_episode_dates("2018-12-31", "2019-01-02")
    with pytest.raises(ValueError, match="sell date precedes buy date"):
        assert_training_episode_dates("2020-01-03", "2020-01-02")


def test_report_path_rejects_training_and_warmup_source_roots():
    from cross_signal_strategy.local.local_data_loader import (
        APPROVED_TRAINING_ROOT,
        APPROVED_WARMUP_ROOT,
    )
    from cross_signal_strategy.research.extreme_lag_attribution import assert_report_path

    safe = Path("cross_signal_strategy/reports/extreme_lag_attribution_2019_2021.md")
    assert assert_report_path(safe).name == safe.name

    for protected in (APPROVED_TRAINING_ROOT, APPROVED_WARMUP_ROOT):
        with pytest.raises(ValueError, match="read-only"):
            assert_report_path(Path(protected) / "derived.md")


def test_entry_lag_accounts_for_each_contributing_cross_age_and_weight():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        build_entry_lag_observation,
    )

    item = build_entry_lag_observation(
        _closed_trade(),
        _signal_frame(),
        forward_closes=[10.2, 9.0, 11.0, 10.5, 10.8],
    )

    assert [(cross.name, cross.age, cross.weight) for cross in item.contributing_crosses] == [
        ("rsi6_cross_rsi12_up", 2, 12.0),
        ("macd_cross_up", 0, 10.0),
        ("kdj_k_cross_up", 1, 6.0),
    ]
    assert item.reversal_contribution_by_age == (10.0, 6.0, 12.0)
    assert item.age_two_reversal_share == pytest.approx(12.0 / 28.0)
    assert item.earliest_cross_date == "2020-01-02"
    assert item.earliest_cross_to_fill_sessions == 3
    assert item.extension_from_earliest_cross_atr == pytest.approx((10.0 - 9.2) / 0.5)
    assert item.execution_gap_atr == pytest.approx((10.0 - 9.8) / 0.5)
    assert item.evaluation_mae_5 == pytest.approx(-0.10)
    assert item.evaluation_mfe_5 == pytest.approx(0.10)


def test_entry_lag_rejects_non_t_minus_one_signal_frame_and_missing_active_age():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        build_entry_lag_observation,
    )

    with pytest.raises(ValueError, match="ends after signal_date"):
        build_entry_lag_observation(
            _closed_trade(),
            _signal_frame(last_date="2020-01-07"),
            forward_closes=[10.0] * 5,
        )

    broken = dict(_closed_trade().entry_score)
    broken.pop("macd_cross_up_age")
    with pytest.raises(ValueError, match="active cross age"):
        build_entry_lag_observation(
            _closed_trade(entry_score=broken),
            _signal_frame(),
            forward_closes=[10.0] * 5,
        )


def test_entry_lag_retains_missing_atr_and_short_forward_path():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        build_entry_lag_observation,
    )

    score = dict(_closed_trade().entry_score)
    score["atr"] = None
    item = build_entry_lag_observation(
        _closed_trade(entry_score=score),
        _signal_frame(),
        forward_closes=[10.2, 9.8],
    )

    assert item.extension_from_earliest_cross_atr is None
    assert item.execution_gap_atr is None
    assert item.evaluation_mae_5 is None
    assert item.evaluation_mfe_5 is None
    assert "entry_atr" in item.missing_fields
    assert "evaluation_path_5" in item.missing_fields


def test_forward_entry_labels_cannot_change_signal_derived_fields():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        build_entry_lag_observation,
    )

    adverse = build_entry_lag_observation(
        _closed_trade(), _signal_frame(), forward_closes=[8.0] * 5
    )
    favorable = build_entry_lag_observation(
        _closed_trade(), _signal_frame(), forward_closes=[12.0] * 5
    )

    assert adverse.contributing_crosses == favorable.contributing_crosses
    assert adverse.reversal_contribution_by_age == favorable.reversal_contribution_by_age
    assert adverse.age_two_reversal_share == favorable.age_two_reversal_share
    assert adverse.earliest_cross_date == favorable.earliest_cross_date
    assert adverse.extension_from_earliest_cross_atr == favorable.extension_from_earliest_cross_atr
    assert adverse.evaluation_mae_5 != favorable.evaluation_mae_5


def test_warmup_cross_can_define_age_but_not_price_performance_metrics():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        build_entry_lag_observation,
    )

    score = dict(_closed_trade().entry_score)
    score["signal_date"] = "2018-12-31"
    score["max_data_date"] = "2018-12-31"
    trade = _closed_trade(
        buy_date="2019-01-02",
        sell_date="2019-01-10",
        entry_score=score,
    )
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2018-12-26", "2018-12-27", "2018-12-28", "2018-12-31"]),
        "close": [9.0, 9.2, 9.6, 9.8],
    })

    item = build_entry_lag_observation(trade, frame, [10.0] * 5)

    assert item.contributing_crosses
    assert item.extension_from_earliest_cross_atr is None
    assert item.execution_gap_atr is None
    assert "warmup_price_metrics_excluded" in item.missing_fields


def test_exit_lag_uses_first_eligible_high_score_and_trading_session_delay():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        ExitSignalDay,
        build_exit_lag_observation,
    )

    dates = pd.bdate_range("2020-01-02", periods=9).strftime("%Y-%m-%d").tolist()
    trade = _closed_trade(
        buy_date=dates[0], sell_date=dates[8], buy_price=10.0, sell_price=10.5
    )
    days = [
        ExitSignalDay(dates[4], dates[3], 35.0, False, False, 10.7),
        ExitSignalDay(dates[5], dates[4], 32.0, False, False, 10.6),
        ExitSignalDay(dates[6], dates[5], 40.0, True, True, 10.8),
        ExitSignalDay(dates[7], dates[6], 38.0, True, False, 10.4),
    ]

    item = build_exit_lag_observation(
        trade,
        trade_dates=dates,
        signal_days=days,
        peak_close=11.0,
        post_exit_closes=[10.4, 10.3, 10.2, 10.1, 10.0],
        min_hold_days=5,
    )

    assert item.first_high_score_date == dates[5]
    assert item.first_high_score_state == "confirmation_absent"
    assert item.first_high_score_to_exit_sessions == 3
    assert item.profit_at_first_high_score == pytest.approx(0.06)
    assert item.peak_close_profit == pytest.approx(0.10)
    assert item.exit_profit == pytest.approx(0.05)
    assert item.giveback_from_peak == pytest.approx(0.05)
    assert item.incremental_giveback_after_first_high_score == pytest.approx(0.01)
    assert item.evaluation_post_exit_return_3 == pytest.approx(10.2 / 10.5 - 1.0)
    assert item.evaluation_post_exit_return_5 == pytest.approx(10.0 / 10.5 - 1.0)


def test_exit_lag_classifies_confirmation_and_protection_states():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        ExitSignalDay,
        build_exit_lag_observation,
    )

    dates = pd.bdate_range("2020-01-02", periods=7).strftime("%Y-%m-%d").tolist()
    trade = _closed_trade(buy_date=dates[0], sell_date=dates[6])

    protected = build_exit_lag_observation(
        trade,
        dates,
        [ExitSignalDay(dates[5], dates[4], 31.0, True, True, 10.2)],
        peak_close=10.5,
        post_exit_closes=[],
    )
    confirmed = build_exit_lag_observation(
        trade,
        dates,
        [ExitSignalDay(dates[5], dates[4], 31.0, True, False, 10.2)],
        peak_close=10.5,
        post_exit_closes=[],
    )

    assert protected.first_high_score_state == "protected"
    assert confirmed.first_high_score_state == "confirmation_present"


def test_exit_lag_retains_missing_high_score_and_separates_atr_exit():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        build_exit_lag_observation,
    )

    dates = pd.bdate_range("2020-01-02", periods=7).strftime("%Y-%m-%d").tolist()
    trade = _closed_trade(
        buy_date=dates[0], sell_date=dates[6], sell_reason="atr_stop"
    )
    item = build_exit_lag_observation(
        trade,
        trade_dates=dates,
        signal_days=[],
        peak_close=10.6,
        post_exit_closes=[10.1, 10.0],
    )

    assert item.exit_type == "atr_stop"
    assert item.first_high_score_date is None
    assert item.first_high_score_to_exit_sessions is None
    assert item.incremental_giveback_after_first_high_score is None
    assert item.evaluation_post_exit_return_3 is None
    assert item.evaluation_post_exit_return_5 is None
    assert "first_high_score" in item.missing_fields
    assert "evaluation_post_exit_3" in item.missing_fields
    assert "evaluation_post_exit_5" in item.missing_fields


def test_exit_signal_days_must_be_t_minus_one_and_forward_labels_are_isolated():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        ExitSignalDay,
        build_exit_lag_observation,
    )

    dates = pd.bdate_range("2020-01-02", periods=7).strftime("%Y-%m-%d").tolist()
    trade = _closed_trade(buy_date=dates[0], sell_date=dates[6])
    invalid = ExitSignalDay(dates[5], dates[5], 35.0, True, False, 10.2)
    with pytest.raises(ValueError, match="must precede execution date"):
        build_exit_lag_observation(trade, dates, [invalid], 10.5, [10.0] * 5)

    day = ExitSignalDay(dates[5], dates[4], 35.0, True, False, 10.2)
    adverse = build_exit_lag_observation(trade, dates, [day], 10.5, [8.0] * 5)
    favorable = build_exit_lag_observation(trade, dates, [day], 10.5, [12.0] * 5)

    assert adverse.first_high_score_date == favorable.first_high_score_date
    assert adverse.first_high_score_state == favorable.first_high_score_state
    assert adverse.first_high_score_to_exit_sessions == favorable.first_high_score_to_exit_sessions
    assert adverse.giveback_from_peak == favorable.giveback_from_peak
    assert adverse.evaluation_post_exit_return_5 != favorable.evaluation_post_exit_return_5


def _entry_observation(year, code, extension, mfe):
    from cross_signal_strategy.research.extreme_lag_attribution import EntryLagObservation

    return EntryLagObservation(
        code=code,
        buy_date=f"{year}-02-01",
        signal_date=f"{year}-01-31",
        buy_price=10.0,
        entry_atr=0.5,
        contributing_crosses=(),
        reversal_contribution_by_age=(10.0, 0.0, 0.0),
        age_two_reversal_share=0.0,
        earliest_cross_date=f"{year}-01-31",
        earliest_cross_to_fill_sessions=1,
        extension_from_earliest_cross_atr=extension,
        execution_gap_atr=0.2,
        evaluation_mae_5=-0.02,
        evaluation_mfe_5=mfe,
        missing_fields=(),
    )


def _exit_observation(year, code, exit_type, delay, giveback):
    from cross_signal_strategy.research.extreme_lag_attribution import ExitLagObservation

    return ExitLagObservation(
        code=code,
        buy_date=f"{year}-02-01",
        sell_date=f"{year}-02-20",
        exit_type=exit_type,
        first_high_score_date=f"{year}-02-14",
        first_high_score_signal_date=f"{year}-02-13",
        first_high_score_state="confirmation_absent",
        first_high_score_to_exit_sessions=delay,
        profit_at_first_high_score=0.05,
        peak_close_profit=0.08,
        exit_profit=0.03,
        giveback_from_peak=0.05,
        incremental_giveback_after_first_high_score=giveback,
        evaluation_post_exit_return_3=0.01,
        evaluation_post_exit_return_5=0.02,
        missing_fields=(),
    )


def test_distribution_summary_keeps_missing_count_and_literal_quartiles():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        summarize_distribution,
    )

    stats = summarize_distribution([1.0, 2.0, None, 4.0])

    assert stats.count == 4
    assert stats.usable_count == 3
    assert stats.missing_count == 1
    assert stats.median == pytest.approx(2.0)
    assert stats.q1 == pytest.approx(1.5)
    assert stats.q3 == pytest.approx(3.0)
    assert stats.minimum == pytest.approx(1.0)
    assert stats.maximum == pytest.approx(4.0)


def test_extreme_lag_summary_groups_full_year_etf_and_exit_type():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        OfficialPathEvidence,
        summarize_extreme_lag,
    )

    entries = [
        _entry_observation(2019, "510300", 1.0, 0.06),
        _entry_observation(2020, "510300", 2.0, 0.04),
        _entry_observation(2021, "513100", 3.0, 0.02),
    ]
    exits = [
        _exit_observation(2019, "510300", "signal_sell", 2, 0.01),
        _exit_observation(2020, "513100", "atr_stop", 3, 0.02),
    ]
    report = summarize_extreme_lag(
        entries, exits, OfficialPathEvidence("aligned", 5, 5)
    )

    assert report.entry_distributions["full"]["extension_atr"].count == 3
    assert report.entry_distributions["year:2019"]["extension_atr"].count == 1
    assert report.entry_distributions["etf:510300"]["extension_atr"].count == 2
    assert report.exit_distributions["exit_type:signal_sell"]["delay_sessions"].count == 1
    assert report.exit_distributions["exit_type:atr_stop"]["delay_sessions"].count == 1


def test_step0_stops_for_inconsistent_annual_direction_or_etf_concentration():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        OfficialPathEvidence,
        summarize_extreme_lag,
    )

    inconsistent = []
    for year, code in ((2019, "510300"), (2020, "513100"), (2021, "518880")):
        inconsistent.extend([
            _entry_observation(year, code, 1.0, 0.06),
            _entry_observation(year, code, 2.0, 0.04 if year != 2021 else 0.08),
        ])
    inconsistent_report = summarize_extreme_lag(
        inconsistent, [], OfficialPathEvidence("aligned", 6, 6)
    )
    assert inconsistent_report.decision.status == "stop"
    assert any("direction" in reason for reason in inconsistent_report.decision.reasons)

    concentrated = []
    for year in (2019, 2020, 2021):
        concentrated.extend([
            _entry_observation(year, "510300", 1.0, 0.06),
            _entry_observation(year, "510300", 2.0, 0.04),
        ])
    concentrated_report = summarize_extreme_lag(
        concentrated, [], OfficialPathEvidence("aligned", 6, 6)
    )
    assert concentrated_report.decision.status == "stop"
    assert any("ETF concentration" in reason for reason in concentrated_report.decision.reasons)


def test_step0_stops_when_path_is_not_aligned_and_report_marks_forward_labels():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        OfficialPathEvidence,
        format_extreme_lag_report,
        summarize_extreme_lag,
    )

    report = summarize_extreme_lag(
        [], [], OfficialPathEvidence("blocked_missing_official_path", 0, 0)
    )
    rendered = format_extreme_lag_report(report)

    assert report.decision.status == "stop"
    assert report.decision.reasons == ("official filled path is not aligned",)
    assert "Step 0 status: STOP" in rendered
    assert "Entry distributions" in rendered
    assert "Exit distributions" in rendered
    assert "Tail observations (examples only)" in rendered
    assert "forward labels only" in rendered


def test_entry_attribution_retains_open_filled_buy_at_training_end():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        build_entry_lag_observation,
    )

    trade = SimpleNamespace(
        code="510300",
        buy_date="2020-01-07",
        buy_price=10.0,
        entry_score=_closed_trade().entry_score,
    )

    item = build_entry_lag_observation(
        trade, _signal_frame(), forward_closes=[10.0] * 5
    )

    assert item.buy_date == "2020-01-07"
    assert item.code == "510300"


def test_training_runner_requires_official_path_before_touching_market_loader():
    from cross_signal_strategy.research.extreme_lag_attribution import (
        run_training_extreme_lag_attribution,
    )

    class ExplodingLoader:
        def load_daily_frame(self, code, trade_date):
            raise AssertionError("loader must not be touched without official path evidence")

    with pytest.raises(ValueError, match="official JoinQuant filled path is required"):
        run_training_extreme_lag_attribution([], loader=ExplodingLoader())


def test_artifact_writer_emits_explicit_blocked_markdown_and_json(tmp_path):
    import json

    from cross_signal_strategy.research.extreme_lag_attribution import (
        OfficialPathEvidence,
        summarize_extreme_lag,
        write_extreme_lag_artifacts,
    )

    report = summarize_extreme_lag(
        [], [], OfficialPathEvidence("blocked_missing_official_path", 0, 0)
    )
    markdown_path, json_path = write_extreme_lag_artifacts(report, tmp_path)

    assert markdown_path.parent == tmp_path.resolve()
    assert "Step 0 status: STOP" in markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["path_evidence"]["status"] == "blocked_missing_official_path"
    assert payload["decision"]["status"] == "stop"


def test_cli_report_directory_must_be_cross_signal_reports_directory(tmp_path):
    from cross_signal_strategy.research.extreme_lag_attribution import (
        assert_repository_report_dir,
    )

    approved = Path("cross_signal_strategy/reports")
    assert assert_repository_report_dir(approved).name == "reports"

    with pytest.raises(ValueError, match="cross_signal_strategy/reports"):
        assert_repository_report_dir(tmp_path)


def test_aligned_capture_binds_filled_episode_and_builds_entry_exit_rows():
    from cross_signal_strategy.local.local_backtester import DayResult, OrderResult
    from cross_signal_strategy.research.extreme_lag_attribution import (
        build_extreme_lag_from_capture,
    )

    dates = pd.bdate_range("2020-01-07", periods=12).strftime("%Y-%m-%d").tolist()
    score = dict(_closed_trade().entry_score)
    buy = OrderResult(
        code="510300", amount_delta=1000, exec_price=10.0, commission=5.0,
        side_time=f"{dates[0]} 09:35", filled=True, reason="buy_signal",
    )
    sell = OrderResult(
        code="510300", amount_delta=-1000, exec_price=10.5, commission=5.0,
        side_time=f"{dates[6]} 09:35", filled=True, reason="signal_sell",
    )
    results = [
        DayResult(dates[0], "2020-01-06", [buy], 10000.0, {}, {}, 20000.0),
        DayResult(dates[6], dates[5], [sell], 20500.0, {}, {}, 20500.0),
    ]
    exit_score = {
        "signal_date": dates[4],
        "sell_score": 35,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "close_below_falling_ma10": False,
        "downside_continuation": False,
        "far_above_ma20_and_rsi6_down": False,
        "adx": 10.0,
        "plus_di": 10.0,
        "minus_di": 12.0,
        "ma20_slope_non_negative": False,
    }
    planner = SimpleNamespace(
        entry_score_snapshots={(dates[0], "510300"): score},
        exit_score_snapshots={(dates[6], "510300"): exit_score},
        daily_score_snapshots={(dates[5], "510300"): exit_score},
        daily_execution_prices={(dates[5], "510300"): 10.6},
        params={"min_signal_hold_days": 5},
    )

    class FakeAdapter:
        def load_signal_frame(self, code, current_date):
            return _signal_frame(), "2020-01-06"

    history = pd.DataFrame({
        "date": dates,
        "close": [10.1, 10.2, 10.3, 10.4, 10.6, 10.8, 10.4, 10.3, 10.2, 10.1, 10.0, 9.9],
    })

    class FakeLoader:
        def load_daily_frame(self, code, trade_date):
            return history.copy()

    expected = [
        _event(dates[0], "BUY", "510300", 1000),
        _event(dates[6], "SELL", "510300", 1000),
    ]
    report = build_extreme_lag_from_capture(
        expected,
        results,
        planner,
        FakeAdapter(),
        FakeLoader(),
        dates,
    )

    assert report.path_evidence.status == "aligned"
    assert len(report.entries) == 1
    assert len(report.exits) == 1
    assert report.entries[0].buy_date == dates[0]
    assert report.exits[0].first_high_score_date == dates[5]
    assert report.exits[0].first_high_score_state == "confirmation_absent"
