# -*- coding: utf-8 -*-
"""Tests for US-QDII previous-NAV premium attribution."""

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def trade(code="513100", year=2019, pnl=100.0, buy_price=10.0):
    from cross_signal_strategy.research.trade_diagnostics import ClosedTradeDiagnostic

    sell_price = buy_price + float(pnl) / 100.0
    return ClosedTradeDiagnostic(
        code=code,
        buy_date="%d-01-02" % year,
        sell_date="%d-01-10" % year,
        sell_reason="signal_sell",
        amount=100,
        buy_price=buy_price,
        sell_price=sell_price,
        pnl=float(pnl),
        return_pct=(sell_price / buy_price - 1.0) * 100.0,
    )


def stats(trades, average_return, win_rate):
    from cross_signal_strategy.research.us_qdii_premium_diagnostics import PremiumTradeStats

    wins = int(round(trades * win_rate))
    losses = max(0, trades - wins)
    return PremiumTradeStats(
        closed_trades=trades,
        wins=wins,
        losses=losses,
        realized_pnl=0.0,
        gross_profit=100.0,
        gross_loss=100.0 if losses else 0.0,
        average_return=average_return,
        average_premium=0.08,
    )


def test_premium_bucket_uses_pre_registered_economic_boundaries():
    from cross_signal_strategy.research.us_qdii_premium_diagnostics import premium_bucket

    assert premium_bucket(-0.01) == "at_most_2"
    assert premium_bucket(0.02) == "at_most_2"
    assert premium_bucket(0.020001) == "2_to_5"
    assert premium_bucket(0.05) == "2_to_5"
    assert premium_bucket(0.050001) == "5_to_10"
    assert premium_bucket(0.10) == "5_to_10"
    assert premium_bucket(0.100001) == "above_10"
    assert premium_bucket(float("nan")) == "missing"


def test_report_uses_raw_0935_price_and_ignores_non_us_qdii_trades():
    from cross_signal_strategy.research.us_qdii_premium_diagnostics import (
        build_us_qdii_premium_report,
    )

    references = {
        ("513100", "2019-01-02"): (10.8, 10.0),
        ("513500", "2020-01-02"): (10.2, 10.0),
    }
    report = build_us_qdii_premium_report(
        trades=[
            trade("513100", 2019, pnl=100.0, buy_price=11.5),
            trade("513500", 2020, pnl=-40.0),
            trade("159920", 2021, pnl=500.0),
        ],
        reference_lookup=lambda code, date: references[(code, date)],
    )

    assert report.targeted_trades == 2
    assert report.covered_trades == 2
    assert report.missing_trades == 0
    assert report.by_bucket["5_to_10"].closed_trades == 1
    assert report.by_bucket["5_to_10"].average_premium == pytest.approx(0.08)
    assert report.by_bucket["at_most_2"].closed_trades == 1
    assert report.by_year_bucket["2019:5_to_10"].realized_pnl == pytest.approx(100.0)
    assert report.by_code_bucket["513500:at_most_2"].realized_pnl == pytest.approx(-40.0)


def test_report_records_missing_reference_without_dropping_target_trade():
    from cross_signal_strategy.research.us_qdii_premium_diagnostics import (
        build_us_qdii_premium_report,
    )

    report = build_us_qdii_premium_report(
        trades=[trade("513100", 2019), trade("513500", 2020)],
        reference_lookup=lambda code, date: (10.0, 10.0)
        if code == "513100"
        else (_ for _ in ()).throw(KeyError("missing")),
    )

    assert report.targeted_trades == 2
    assert report.covered_trades == 1
    assert report.missing_trades == 1
    assert report.coverage_rate == pytest.approx(0.5)
    assert report.by_bucket["missing"].closed_trades == 1


def test_candidate_gate_requires_coverage_sample_year_and_code_consistency():
    from cross_signal_strategy.research.us_qdii_premium_diagnostics import (
        evaluate_premium_candidate_gate,
    )

    elevated_by_year = {
        2019: stats(4, average_return=-0.03, win_rate=0.25),
        2020: stats(4, average_return=-0.02, win_rate=0.25),
        2021: stats(2, average_return=-0.01, win_rate=0.50),
    }
    normal_by_year = {
        year: stats(10, average_return=0.03, win_rate=0.60)
        for year in (2019, 2020, 2021)
    }
    elevated_by_code = {
        "513100": stats(6, average_return=-0.03, win_rate=0.20),
        "513500": stats(4, average_return=-0.01, win_rate=0.25),
    }

    passed = evaluate_premium_candidate_gate(
        targeted_trades=30,
        covered_trades=28,
        elevated_by_year=elevated_by_year,
        normal_by_year=normal_by_year,
        elevated_by_code=elevated_by_code,
    )
    failed = evaluate_premium_candidate_gate(
        targeted_trades=30,
        covered_trades=20,
        elevated_by_year=elevated_by_year,
        normal_by_year=normal_by_year,
        elevated_by_code=elevated_by_code,
    )

    assert passed.passed is True
    assert passed.reasons == ()
    assert failed.passed is False
    assert any("coverage" in reason for reason in failed.reasons)


def test_report_rejects_dates_outside_training_window():
    from cross_signal_strategy.research.us_qdii_premium_diagnostics import (
        build_us_qdii_premium_report,
    )

    with pytest.raises(ValueError, match="outside 2019-2021 training window"):
        build_us_qdii_premium_report(
            trades=[trade("513100", 2022)],
            reference_lookup=lambda code, date: (10.0, 10.0),
        )


def test_training_runner_reads_0935_close_and_iopv_from_approved_loader():
    from cross_signal_strategy.research.us_qdii_premium_diagnostics import (
        run_training_us_qdii_premium,
    )

    calls = []

    class FakeLoader:
        def get_minute_bar(self, code, date, time):
            calls.append((code, date, time))
            return {"close": 10.8, "iopv": 10.0}

    report = run_training_us_qdii_premium(
        loader=FakeLoader(),
        trades=[trade("513100", 2019, pnl=100.0)],
    )

    assert calls == [("513100", "2019-01-02", "09:35")]
    assert report.by_bucket["5_to_10"].closed_trades == 1


def test_format_report_includes_coverage_buckets_and_gate():
    from cross_signal_strategy.research.us_qdii_premium_diagnostics import (
        build_us_qdii_premium_report,
        format_us_qdii_premium_report,
    )

    report = build_us_qdii_premium_report(
        trades=[trade("513100", 2019)],
        reference_lookup=lambda code, date: (10.8, 10.0),
    )
    text = format_us_qdii_premium_report(report)

    assert "targeted=1 covered=1 missing=0 coverage=100.00%" in text
    assert "BUCKET 5_to_10" in text
    assert "GATE passed=False" in text
