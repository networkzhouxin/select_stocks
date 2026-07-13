# -*- coding: utf-8 -*-
"""Guards for the isolated, no-order PTrade IOPV capability probe."""

import ast
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = (
    ROOT
    / "cross_signal_strategy"
    / "smart_trade_ptrade_cross_signal_iopv_probe.py"
)
DOC = ROOT / "cross_signal_strategy" / "docs" / "ptrade_iopv_probe.md"
README = ROOT / "cross_signal_strategy" / "README.md"
DECISIONS = ROOT / "cross_signal_strategy" / "docs" / "decisions.md"
IOPV_QUALITY = ROOT / "cross_signal_strategy" / "docs" / "iopv_data_quality.md"


def probe_text():
    return PROBE.read_text(encoding="utf-8")


def test_probe_is_isolated_and_contains_no_order_calls():
    text = probe_text()
    tree = ast.parse(text)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "Temporary PTrade IOPV capability probe" in text
    assert not {
        "order",
        "order_value",
        "order_target",
        "order_target_value",
        "buy",
        "sell",
    }.intersection(called_names)


def test_probe_covers_only_qdii_codes_in_the_frozen_ptrade_pool():
    text = probe_text()

    for code in ("513100.SS", "513500.SS", "513880.SS", "513050.SS"):
        assert code in text
    for code in ("159915.SZ", "512100.SS", "159928.SZ", "518880.SS", "159985.SZ"):
        assert code not in text


def test_probe_checks_realtime_iopv_timestamp_and_etf_metadata():
    text = probe_text()

    assert "get_snapshot(QDII_CODES)" in text
    assert "get_etf_info(QDII_CODES)" in text
    assert 'snapshot.get("iopv")' in text
    assert 'snapshot.get("hsTimeStamp")' in text
    assert 'etf_info.get("publish")' in text
    assert 'etf_info.get("nav_pre")' in text
    assert 'etf_info.get("nav_percu")' in text
    assert "[ptrade-iopv-snapshot]" in text
    assert "[ptrade-iopv-etf-info]" in text


def test_probe_uses_three_intraday_callbacks_within_ptrade_schedule_limit():
    text = probe_text()

    assert text.count("run_daily(context, probe_iopv_capability") == 3
    assert 'time="09:34"' in text
    assert 'time="09:35"' in text
    assert 'time="09:36"' in text


def test_probe_documentation_forbids_strategy_or_threshold_inference():
    text = DOC.read_text(encoding="utf-8")

    assert "capability-only" in text
    assert "places no orders" in text
    assert "must not define or tune a premium threshold" in text
    assert "must not be used as validation-period performance evidence" in text
    assert "get_snapshot()" in text
    assert "get_etf_info()" in text


def test_repository_records_platform_capability_without_reopening_research():
    readme = README.read_text(encoding="utf-8")
    decisions = DECISIONS.read_text(encoding="utf-8")
    quality = IOPV_QUALITY.read_text(encoding="utf-8")

    assert "smart_trade_ptrade_cross_signal_iopv_probe.py" in readme
    assert "docs/ptrade_iopv_probe.md" in readme
    assert "Add An Isolated PTrade IOPV Capability Probe" in decisions
    assert "PTrade Live Capability Follow-Up" in quality
    assert "does not reopen" in quality
