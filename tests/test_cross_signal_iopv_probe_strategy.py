from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_FILE = (
    ROOT
    / "cross_signal_strategy"
    / "smart_trade_joinquant_cross_signal_iopv_probe.py"
)


def probe_text():
    return PROBE_FILE.read_text(encoding="utf-8")


def test_iopv_probe_is_isolated_and_places_no_orders():
    text = probe_text()

    assert "Temporary IOPV capability probe" in text
    assert 'PROBE_CODE = "513100.XSHG"' in text
    assert "order_target" not in text
    assert "order_value" not in text
    assert 'set_option("avoid_future_data", True)' in text


def test_iopv_probe_checks_only_two_training_dates():
    text = probe_text()

    assert "datetime.date(2020, 2, 7)" in text
    assert "datetime.date(2020, 9, 21)" in text
    assert "if context.current_dt.date() not in TARGET_DATES:" in text


def test_iopv_probe_checks_current_data_iopv_and_supported_price_fields():
    text = probe_text()

    assert "get_current_data()[PROBE_CODE]" in text
    assert 'hasattr(current, "iopv")' in text
    assert 'getattr(current, "iopv", None)' in text
    assert 'fields=["iopv"]' in text
    assert "[iopv-probe-current]" in text
    assert "[iopv-probe-field]" in text


def test_iopv_probe_compares_same_day_and_previous_day_nav():
    text = probe_text()

    assert "get_extras(" in text
    assert '"unit_net_value"' in text
    assert "get_trade_days(end_date=context.current_dt.date(), count=2)" in text
    assert "[iopv-probe-nav]" in text


def test_iopv_probe_prints_minute_window_at_three_checkpoints():
    text = probe_text()

    assert 'frequency="1m"' in text
    assert 'fields=["close", "volume", "money"]' in text
    assert "[iopv-probe-minute]" in text
    assert 'run_daily(probe_iopv_capability, time="09:34")' in text
    assert 'run_daily(probe_iopv_capability, time="09:35")' in text
    assert 'run_daily(probe_iopv_capability, time="09:36")' in text
