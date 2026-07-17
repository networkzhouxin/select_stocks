from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_FILE = (
    ROOT / "cross_signal_strategy" / "archive" / "probes"
    / "smart_trade_joinquant_cross_signal_etf_probe_513880.py"
)


def test_probe_strategy_file_contains_513880_status_diagnostics():
    text = PROBE_FILE.read_text(encoding="utf-8")

    assert "def probe_513880_status(context):" in text
    assert "513880.XSHG" in text
    assert "get_current_data()[code]" in text
    assert "paused=" in text
    assert "last_price=" in text
    assert "day_open=" in text
    assert "frequency=\"1m\"" in text
    assert "fields=[\"open\", \"close\", \"high\", \"low\", \"volume\", \"money\"]" in text
    assert "skip_paused=False" in text


def test_probe_strategy_registers_three_intraday_checkpoints():
    text = PROBE_FILE.read_text(encoding="utf-8")

    assert 'run_daily(probe_513880_status, time="09:35")' in text
    assert 'run_daily(probe_513880_status, time="10:35")' in text
    assert 'run_daily(probe_513880_status, time="14:50")' in text
    assert 'run_daily(probe_513880_day_volume, time="15:30")' in text
    assert 'run_daily(do_trading, time="09:35")' in text
    assert 'run_daily(after_close, time="15:30")' in text


def test_probe_strategy_prints_full_day_minute_volume_summary():
    text = PROBE_FILE.read_text(encoding="utf-8")

    assert "def probe_513880_day_volume(context):" in text
    assert 'start_date="2019-12-12 09:30:00"' in text
    assert 'end_date="2019-12-12 15:00:00"' in text
    assert "[probe-513880-day-volume]" in text
    assert "total_minutes=%s" in text
    assert "nonzero_minutes=%s" in text
    assert "total_volume=%s" in text
    assert "total_money=%s" in text
    assert "first_nonzero=%s" in text
    assert "last_nonzero=%s" in text
    assert 'if context.current_dt.date() != datetime.date(2019, 12, 12):' in text
    assert 'datetime.time(14, 50)' not in text
