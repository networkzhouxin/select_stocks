from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_FILE = (
    ROOT
    / "cross_signal_strategy"
    / "archive"
    / "probes"
    / "smart_trade_joinquant_underlying_availability_probe.py"
)
README_FILE = ROOT / "cross_signal_strategy" / "README.md"
DECISIONS_FILE = ROOT / "cross_signal_strategy" / "docs" / "decisions.md"


def probe_text():
    return PROBE_FILE.read_text(encoding="utf-8")


def test_probe_is_isolated_training_only_and_places_no_orders():
    text = probe_text()

    assert "Temporary JoinQuant underlying-index availability probe" in text
    assert '"513500.XSHG"' in text
    assert '"513050.XSHG"' in text
    assert "datetime.date(2019, 1, 2)" in text
    assert "datetime.date(2020, 2, 7)" in text
    assert "datetime.date(2020, 9, 21)" in text
    assert "datetime.date(2021, 12, 27)" in text
    assert "order_target" not in text
    assert "order_value" not in text
    assert "order(" not in text


def test_probe_enables_future_data_guard_and_runs_only_at_0935():
    text = probe_text()

    assert 'set_option("avoid_future_data", True)' in text
    assert 'run_daily(probe_underlying_availability, time="09:35")' in text
    assert "run_daily(" in text
    assert text.count("run_daily(") == 1


def test_probe_discovers_tracking_indices_from_official_fund_metadata():
    text = probe_text()

    assert "finance.FUND_INVEST_TARGET" in text
    assert "finance.FUND_INVEST_TARGET.pub_date <= context.current_dt.date()" in text
    assert "traced_index_name" in text
    assert "traced_index_code" in text
    assert "[underlying-availability-metadata]" in text


def test_probe_reads_only_t_minus_one_bars_and_runs_same_day_negative_control():
    text = probe_text()

    assert "end_date=context.previous_date" in text
    assert 'frequency="daily"' in text
    assert 'fields=["close"]' in text
    assert "count=2" in text
    assert "end_date=context.current_dt.date()" in text
    assert "[underlying-availability-t1]" in text
    assert "[underlying-availability-same-day]" in text


def test_probe_states_evidence_limit_in_code_and_runtime_log():
    text = probe_text()

    assert "cannot prove the index publisher's original release timestamp" in text
    assert "platform_readability_only=True" in text
    assert "publisher_timestamp_proved=False" in text


def test_probe_scope_and_evidence_limit_are_documented():
    readme = README_FILE.read_text(encoding="utf-8")
    decisions = DECISIONS_FILE.read_text(encoding="utf-8")

    assert "smart_trade_joinquant_underlying_availability_probe.py" in readme
    assert "cannot establish the publisher's original release timestamp" in readme
    assert "Probe JoinQuant Underlying-Index Readability Without Publishing Availability" in decisions
    assert "513050 and 513500 remain blocked" in decisions
