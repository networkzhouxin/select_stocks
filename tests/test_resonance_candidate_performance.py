import importlib.util
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    ROOT / "resonance_reversal_strategy" / "research"
    / "analyze_candidate_performance.py"
)


def load_analyzer():
    if not ANALYZER_PATH.is_file():
        pytest.fail("candidate performance analyzer module is missing")
    spec = importlib.util.spec_from_file_location(
        "candidate_performance_analyzer", ANALYZER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_structured_log(tmp_path, payloads, name="structured.log"):
    if isinstance(payloads, dict):
        payloads = [payloads]
    lines = []
    for index, payload in enumerate(payloads):
        lines.append(
            "2021-01-%02d 15:30:00 - INFO - %s\n" % (
                index + 4,
                json.dumps(payload, allow_nan=True, sort_keys=True),
            )
        )
    path = tmp_path / name
    path.write_text("".join(lines), encoding="utf-8")
    return path


def portfolio_payload(closing_date, **overrides):
    payload = {
        "event": "portfolio_summary",
        "closing_date": closing_date,
        "total_value": 20000.0,
        "available_cash": 1000.0,
        "positions": {"510300.XSHG": 1000},
        "highest_close_anchors": {"510300.XSHG": 4.0},
    }
    payload.update(overrides)
    return payload


def test_parse_fill_decodes_html_and_preserves_independent_amounts(tmp_path):
    analyzer = load_analyzer()
    log = tmp_path / "candidate.log"
    log.write_text(
        "2021-06-18 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=&#x6f;pen) trade price: 5.065, "
        "amount:1600, commission: 5.0\n"
        "2021-06-25 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=close) trade price: 1.232, "
        "amount: 6400, commission: 5.0\n",
        encoding="utf-8",
    )

    parsed = analyzer.parse_joinquant_log([log])

    assert [item.amount for item in parsed.fills] == [1600, 6400]
    assert [item.side for item in parsed.fills] == ["BUY", "SELL"]
    assert [item.code for item in parsed.fills] == [
        "159928.XSHE", "159928.XSHE",
    ]


def test_parse_portfolio_summary_preserves_frozen_fields(tmp_path):
    analyzer = load_analyzer()
    log = write_structured_log(
        tmp_path, portfolio_payload("2021-01-04"),
    )

    parsed = analyzer.parse_joinquant_log([log])

    assert len(parsed.portfolio_points) == 1
    point = parsed.portfolio_points[0]
    assert point.closing_date.isoformat() == "2021-01-04"
    assert point.total_value == pytest.approx(20000.0)
    assert point.available_cash == pytest.approx(1000.0)
    assert point.positions == (("510300.XSHG", 1000),)


def test_parse_portfolio_summary_rejects_nonfinite_total_value(tmp_path):
    analyzer = load_analyzer()
    log = write_structured_log(
        tmp_path,
        portfolio_payload("2021-01-04", total_value=float("nan")),
    )

    with pytest.raises(ValueError, match="total_value.*finite"):
        analyzer.parse_joinquant_log([log])


@pytest.mark.parametrize(
    "positions",
    [
        {"510300.XSHG": -1},
        {"510300.XSHG": 1.5},
        {"": 100},
        [],
    ],
)
def test_parse_portfolio_summary_rejects_malformed_positions(
        tmp_path, positions):
    analyzer = load_analyzer()
    log = write_structured_log(
        tmp_path,
        portfolio_payload("2021-01-04", positions=positions),
    )

    with pytest.raises(ValueError, match="positions"):
        analyzer.parse_joinquant_log([log])


def test_parse_portfolio_summary_rejects_duplicate_date(tmp_path):
    analyzer = load_analyzer()
    log = write_structured_log(
        tmp_path,
        [
            portfolio_payload("2021-01-04"),
            portfolio_payload("2021-01-04"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate portfolio date"):
        analyzer.parse_joinquant_log([log])


def test_parse_portfolio_summary_rejects_decreasing_date(tmp_path):
    analyzer = load_analyzer()
    log = write_structured_log(
        tmp_path,
        [
            portfolio_payload("2021-01-05"),
            portfolio_payload("2021-01-04"),
        ],
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        analyzer.parse_joinquant_log([log])


def test_pair_completed_trade_uses_cash_flows_across_split(tmp_path):
    analyzer = load_analyzer()
    log = tmp_path / "split.log"
    log.write_text(
        "2021-06-18 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=open) trade price: 5.065, "
        "amount:1600, commission: 5.0\n"
        "2021-06-25 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=close) trade price: 1.232, "
        "amount:6400, commission: 5.0\n",
        encoding="utf-8",
    )

    ledger = analyzer.pair_completed_trades(
        analyzer.parse_joinquant_log([log]).fills,
    )

    assert ledger.open_positions == ()
    assert len(ledger.completed_trades) == 1
    trade = ledger.completed_trades[0]
    assert trade.pnl == pytest.approx(-229.2)
    assert trade.return_rate == pytest.approx(-229.2 / 8109.0)
    assert trade.amount_ratio == pytest.approx(4.0)
    assert trade.buy_amount == 1600
    assert trade.sell_amount == 6400


def test_pair_completed_trades_preserves_unclosed_buy(tmp_path):
    analyzer = load_analyzer()
    log = tmp_path / "open.log"
    log.write_text(
        "2021-06-18 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=open) trade price: 5.065, "
        "amount:1600, commission: 5.0\n",
        encoding="utf-8",
    )

    ledger = analyzer.pair_completed_trades(
        analyzer.parse_joinquant_log([log]).fills,
    )

    assert ledger.completed_trades == ()
    assert len(ledger.open_positions) == 1
    assert ledger.open_positions[0].code == "159928.XSHE"


def test_pair_completed_trades_rejects_sell_without_open(tmp_path):
    analyzer = load_analyzer()
    log = tmp_path / "orphan-sell.log"
    log.write_text(
        "2021-06-25 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=close) trade price: 1.232, "
        "amount:6400, commission: 5.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sell without open"):
        analyzer.pair_completed_trades(
            analyzer.parse_joinquant_log([log]).fills,
        )


def test_pair_completed_trades_rejects_duplicate_open(tmp_path):
    analyzer = load_analyzer()
    log = tmp_path / "duplicate-open.log"
    log.write_text(
        "2021-06-18 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=open) trade price: 5.065, "
        "amount:1600, commission: 5.0\n"
        "2021-06-21 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=open) trade price: 4.900, "
        "amount:1600, commission: 5.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate open"):
        analyzer.pair_completed_trades(
            analyzer.parse_joinquant_log([log]).fills,
        )


def test_pair_completed_trades_rejects_decreasing_fill_time(tmp_path):
    analyzer = load_analyzer()
    first = tmp_path / "01-later.log"
    second = tmp_path / "02-earlier.log"
    first.write_text(
        "2021-06-25 09:35:00 - INFO - order StockOrder("
        "security=159928.XSHE action=open) trade price: 1.232, "
        "amount:6400, commission: 5.0\n",
        encoding="utf-8",
    )
    second.write_text(
        "2021-06-18 09:35:00 - INFO - order StockOrder("
        "security=510300.XSHG action=open) trade price: 5.065, "
        "amount:1600, commission: 5.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fill timestamps"):
        analyzer.pair_completed_trades(
            analyzer.parse_joinquant_log([first, second]).fills,
        )


def make_completed_trade(analyzer, pnl, return_rate, index=0):
    return analyzer.CompletedTrade(
        code="TEST%02d.XSHG" % index,
        entry_date=analyzer.date(2021, 1, 4),
        exit_date=analyzer.date(2021, 1, 5),
        buy_price=10.0,
        buy_amount=100,
        buy_commission=0.0,
        sell_price=10.0,
        sell_amount=100,
        sell_commission=0.0,
        pnl=float(pnl),
        return_rate=float(return_rate),
        amount_ratio=1.0,
    )


def make_portfolio_points(analyzer, values):
    return tuple(
        analyzer.PortfolioPoint(
            closing_date=analyzer.date(2021, 1, 4 + index),
            total_value=float(value),
            available_cash=float(value),
            positions=(),
        )
        for index, value in enumerate(values)
    )


def passing_candidate_metrics():
    return {
        "total_return": 1.30,
        "win_rate": 0.60,
        "wilson_lower_95": 0.51,
        "max_drawdown": 0.06,
        "closed_trade_count": 80,
        "median_trade_return": 0.01,
        "top_10pct_gross_profit_share": 0.49,
    }


def test_metrics_use_daily_equity_for_drawdown_and_wilson_lower_bound():
    analyzer = load_analyzer()
    points = make_portfolio_points(analyzer, [20000, 22000, 19800, 23000])
    returns = [0.10, -0.05, 0.03, 0.02]
    trades = tuple(
        make_completed_trade(analyzer, value * 1000, value, index)
        for index, value in enumerate(returns)
    )

    report = analyzer.summarize_performance_from_records(
        points, trades, 20000.0,
    )

    assert report["total_return"] == pytest.approx(0.15)
    assert report["max_drawdown"] == pytest.approx(0.10)
    assert report["closed_trade_count"] == 4
    assert report["win_rate"] == pytest.approx(0.75)
    assert 0 < report["wilson_lower_95"] < report["win_rate"]


def test_top_ten_percent_uses_all_completed_trades_as_denominator():
    analyzer = load_analyzer()
    pnls = [100, 90, 80, 70, 60, 50, 40, 30, 20, -10]
    trades = tuple(
        make_completed_trade(analyzer, pnl, pnl / 1000.0, index)
        for index, pnl in enumerate(pnls)
    )

    summary = analyzer.summarize_trades(trades)

    assert summary["top_10pct_trade_count"] == 1
    assert summary["top_10pct_gross_profit_share"] == pytest.approx(100 / 540)
    assert summary["gross_profit"] == pytest.approx(540)
    assert summary["gross_loss"] == pytest.approx(-10)
    assert summary["profit_factor"] == pytest.approx(54)


def test_final_gates_fail_closed_without_double_friction_report():
    analyzer = load_analyzer()

    gates = analyzer.evaluate_final_gates(
        passing_candidate_metrics(), None,
    )

    assert gates["double_friction_beats_benchmark"] is False
    assert gates["all_passed"] is False


def test_final_gates_apply_frozen_strict_and_inclusive_boundaries():
    analyzer = load_analyzer()
    candidate = {
        "total_return": 1.2925,
        "win_rate": 0.558,
        "wilson_lower_95": 0.50,
        "max_drawdown": 0.0628,
        "closed_trade_count": 80,
        "median_trade_return": 0.0,
        "top_10pct_gross_profit_share": 0.50,
    }

    gates = analyzer.evaluate_final_gates(
        candidate, {"total_return": 0.6410},
    )

    assert gates == {
        "total_return_above_cross": False,
        "win_rate_above_cross": False,
        "wilson_lower_above_half": False,
        "max_drawdown_below_cross": False,
        "closed_trades_at_least_80": True,
        "median_trade_return_positive": False,
        "top_profit_share_at_most_half": True,
        "double_friction_beats_benchmark": False,
        "all_passed": False,
    }


def write_initialization_log(path, builds):
    lines = []
    for build in builds:
        lines.append(
            "2019-01-01 00:00:00 - INFO - %s\n" % json.dumps({
                "event": "strategy_initialized",
                "build": build,
            }, sort_keys=True)
        )
    path.write_text("".join(lines), encoding="utf-8")


def cli_args(baseline, candidate, double_friction, calendar, output):
    return [
        "--baseline-log", str(baseline),
        "--candidate-log", str(candidate),
        "--double-friction-log", str(double_friction),
        "--expected-candidate-build", "20260828.TEST",
        "--session-calendar", str(calendar),
        "--session-calendar-sha256", "0" * 64,
        "--output", str(output),
    ]


def test_cli_requires_distinct_double_friction_log_and_output(tmp_path):
    analyzer = load_analyzer()
    baseline = tmp_path / "baseline.log"
    candidate = tmp_path / "candidate.log"
    calendar = tmp_path / "calendar.json"
    output = tmp_path / "report.json"
    for path in (baseline, candidate, calendar):
        path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be distinct"):
        analyzer.main(
            cli_args(baseline, candidate, candidate, calendar, output),
        )

    double_friction = tmp_path / "double.log"
    double_friction.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be distinct"):
        analyzer.main(
            cli_args(
                baseline, candidate, double_friction, calendar, candidate,
            ),
        )


def test_cli_report_contains_manifest_and_all_frozen_gates(
        tmp_path, monkeypatch):
    analyzer = load_analyzer()
    baseline = tmp_path / "baseline.log"
    candidate = tmp_path / "candidate.log"
    double_friction = tmp_path / "double.log"
    calendar = tmp_path / "calendar.json"
    output = tmp_path / "report.json"
    for path in (baseline, candidate, double_friction, calendar):
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        analyzer,
        "read_session_calendar_manifest_bytes",
        lambda path: b"{}",
    )
    monkeypatch.setattr(
        analyzer,
        "validate_session_calendar_manifest",
        lambda raw, digest: {"schema_version": 2},
    )
    expected_report = {
        "session_calendar": {"schema_version": 2},
        "run_identity": {
            "baseline_build": "20260827.4",
            "candidate_build": "20260828.TEST",
            "double_friction_build": "20260828.TEST",
        },
        "gates": {
            "total_return_above_cross": True,
            "win_rate_above_cross": True,
            "wilson_lower_above_half": True,
            "max_drawdown_below_cross": True,
            "closed_trades_at_least_80": True,
            "median_trade_return_positive": True,
            "top_profit_share_at_most_half": True,
            "double_friction_beats_benchmark": True,
            "all_passed": True,
        },
    }
    monkeypatch.setattr(
        analyzer, "analyze_paths",
        lambda args, manifest: expected_report,
    )

    exit_code = analyzer.main(
        cli_args(
            baseline, candidate, double_friction, calendar, output,
        ),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report == expected_report
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


@pytest.mark.parametrize(
    "builds,expected,match",
    [
        ([], "20260827.4", "missing.*initialization"),
        (["20260827.4", "20260827.4"], "20260827.4", "duplicate"),
        (["WRONG"], "20260827.4", "build mismatch"),
    ],
)
def test_build_identity_fails_closed(builds, expected, match, tmp_path):
    analyzer = load_analyzer()
    log = tmp_path / "role.log"
    write_initialization_log(log, builds)
    parsed = analyzer.parse_joinquant_log([log])

    with pytest.raises(ValueError, match=match):
        analyzer.require_expected_build(parsed, expected, "baseline")


class FakeManifest:
    def __init__(self, sessions):
        self.sessions = tuple(sessions)


def test_training_boundary_requires_exact_manifest_portfolio_sessions():
    analyzer = load_analyzer()
    sessions = (
        analyzer.date(2018, 12, 28),
        analyzer.date(2021, 1, 4),
        analyzer.date(2021, 1, 5),
    )
    points = make_portfolio_points(analyzer, [20000, 20100])
    parsed = analyzer.ParsedLog((), points, ("20260827.4",))

    analyzer.validate_training_boundary(
        parsed, FakeManifest(sessions), "baseline",
    )

    missing = analyzer.ParsedLog(
        (), points[:1], ("20260827.4",),
    )
    with pytest.raises(ValueError, match="portfolio sessions.*manifest"):
        analyzer.validate_training_boundary(
            missing, FakeManifest(sessions), "baseline",
        )


def test_training_boundary_rejects_2022_portfolio_and_fill_records():
    analyzer = load_analyzer()
    sessions = (
        analyzer.date(2019, 1, 2),
        analyzer.date(2021, 12, 31),
    )
    point_2022 = analyzer.PortfolioPoint(
        closing_date=analyzer.date(2022, 1, 4),
        total_value=20000.0,
        available_cash=20000.0,
        positions=(),
    )
    parsed_point = analyzer.ParsedLog(
        (), (point_2022,), ("20260827.4",),
    )
    with pytest.raises(ValueError, match="portfolio.*outside 2019-2021"):
        analyzer.validate_training_boundary(
            parsed_point, FakeManifest(sessions), "baseline",
        )

    fill_2022 = analyzer.Fill(
        timestamp=analyzer.datetime(2022, 1, 4, 9, 35),
        trade_date=analyzer.date(2022, 1, 4),
        code="510300.XSHG",
        side="BUY",
        price=4.0,
        amount=100,
        commission=5.0,
    )
    valid_points = (
        analyzer.PortfolioPoint(
            closing_date=session,
            total_value=20000.0,
            available_cash=20000.0,
            positions=(),
        )
        for session in sessions
    )
    parsed_fill = analyzer.ParsedLog(
        (fill_2022,), tuple(valid_points), ("20260827.4",),
    )
    with pytest.raises(ValueError, match="fill.*outside 2019-2021"):
        analyzer.validate_training_boundary(
            parsed_fill, FakeManifest(sessions), "baseline",
        )
