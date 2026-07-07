from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_FILE = ROOT / "cross_signal_strategy" / "smart_trade_joinquant_cross_signal_etf_probe_513880.py"


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
    assert 'run_daily(do_trading, time="09:35")' in text
    assert 'run_daily(after_close, time="15:30")' in text
