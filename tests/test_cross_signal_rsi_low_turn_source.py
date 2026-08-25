import importlib
import json
import sys
import types
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from cross_signal_strategy.research.rsi_low_turn_shadow import (
    calculate_rsi6,
    detect_rsi_low_turn,
)
from cross_signal_strategy.research.rsi_low_turn_source import (
    SourceContractError,
    load_arrival_input,
    load_manifest,
)
import cross_signal_strategy.research.rsi_low_turn_source as source_module


VALID_MANIFEST = {
    "purpose": "rsi_low_turn_prospective_shadow",
    "version": "rsi-low-turn-shadow-v0.1",
    "collection_start": "2026-08-26",
    "timezone": "Asia/Shanghai",
    "append_only": True,
    "daily_subdir": "daily",
    "minute_subdir": "minute_0935",
}
ARRIVAL_2026_08_26 = datetime(2026, 8, 26, 9, 35, tzinfo=ZoneInfo("Asia/Shanghai"))


def write_manifest(root: Path, payload: dict[str, object]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def build_valid_source(tmp_path: Path, minute_overrides=None) -> Path:
    root = write_manifest(tmp_path / "source", VALID_MANIFEST)
    (root / "daily").mkdir()
    (root / "minute_0935").mkdir()
    dates = pd.bdate_range(end="2026-08-25", periods=30)
    daily = pd.DataFrame({
        "code": "513100", "date": dates.date,
        "open": np.linspace(2.30, 2.01, 30),
        "high": np.linspace(2.32, 2.03, 30),
        "low": np.linspace(2.28, 1.99, 30),
        "close": np.linspace(2.30, 2.01, 30),
        "volume": 100000,
        "available_at": [f"{day}T15:01:00+08:00" for day in dates.date],
        "source": "pytest_fixture",
    })
    daily.to_csv(root / "daily" / "513100.csv", index=False)
    minute = {
        "code": "513100", "timestamp": "2026-08-26T09:35:00+08:00",
        "open": 2.035, "close": 2.035, "volume": 1000, "num_trades": 10,
        "available_at": "2026-08-26T09:35:00+08:00", "source": "pytest_fixture",
    }
    minute.update(minute_overrides or {})
    pd.DataFrame([minute]).to_csv(root / "minute_0935" / "513100.csv", index=False)
    return root


def build_source_with_future_rows(tmp_path: Path) -> Path:
    root = build_valid_source(tmp_path)
    path = root / "daily" / "513100.csv"
    frame = pd.read_csv(path)
    frame.loc[len(frame)] = [
        "513100", "2026-08-26", 9, 9, 9, 9, 1,
        "2026-08-26T15:01:00+08:00", "pytest_future_row",
    ]
    frame.to_csv(path, index=False)
    return root


def build_zero_trade_args(tmp_path: Path):
    root = build_valid_source(tmp_path, {"volume": 0, "num_trades": 0})
    daily_path = root / "daily" / "513100.csv"
    daily = pd.read_csv(daily_path)
    daily.loc[len(daily) - 1, "close"] = daily.loc[len(daily) - 2, "close"] + 0.005
    daily.to_csv(daily_path, index=False)
    return root, root, "513100", ARRIVAL_2026_08_26


def rsi_series_ending(r2, r1, r0):
    def fake_rsi(close):
        result = pd.Series(np.nan, index=close.index, dtype=float)
        result.iloc[-3:] = [r2, r1, r0]
        return result

    return fake_rsi


def calculate_formal_background_from_same_frame(root: Path) -> dict[str, float]:
    # The formal JoinQuant module is allowed only here, after a test-local stub.
    sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))
    formal = importlib.import_module("cross_signal_strategy.smart_trade_joinquant_cross_signal_etf")
    frame = pd.read_csv(root / "daily" / "513100.csv")
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    rsi12 = formal.calc_rsi(close, 12)
    rsi24 = formal.calc_rsi(close, 24)
    k, d, j = formal.calc_kdj(high, low, close, 9, 3, 3)
    dif, dea, hist = formal.calc_macd(close, 12, 26, 9)
    upper, mid, lower = formal.calc_bollinger(close, 20, 2)
    atr = formal.calc_atr(high, low, close, 14)
    return {
        "rsi12": rsi12.iloc[-1], "rsi24": rsi24.iloc[-1],
        "kdj_k": k.iloc[-1], "kdj_d": d.iloc[-1], "kdj_j": j.iloc[-1],
        "macd_dif": dif.iloc[-1], "macd_dea": dea.iloc[-1], "macd_hist": hist.iloc[-1],
        "boll_upper": upper.iloc[-1], "boll_mid": mid.iloc[-1], "boll_lower": lower.iloc[-1],
        "atr14": atr.iloc[-1],
    }


def test_root_must_equal_separately_approved_root(tmp_path):
    source = write_manifest(tmp_path / "source", VALID_MANIFEST)
    with pytest.raises(SourceContractError, match="approved root"):
        load_manifest(source, tmp_path / "other")


@pytest.mark.parametrize("name", [
    "cross_signal_train_2019_2021", "cross_signal_warmup_2018", "按年份合并", "validation_2022_2023",
])
def test_forbidden_roots_are_refused(tmp_path, name):
    root = write_manifest(tmp_path / name, VALID_MANIFEST)
    with pytest.raises(SourceContractError, match="forbidden"):
        load_manifest(root, root)


def test_collection_start_cannot_precede_freeze(tmp_path):
    root = write_manifest(tmp_path / "source", dict(VALID_MANIFEST, collection_start="2026-08-25"))
    with pytest.raises(SourceContractError, match="2026-08-26"):
        load_manifest(root, root)


def test_manifest_must_exactly_describe_append_only_contract(tmp_path):
    root = write_manifest(tmp_path / "source", dict(VALID_MANIFEST, append_only=False))
    with pytest.raises(SourceContractError, match="append_only"):
        load_manifest(root, root)


def test_loader_uses_t_minus_one_and_exact_0935_open(tmp_path):
    root = build_valid_source(tmp_path)
    item = load_arrival_input(root, root, "513100", ARRIVAL_2026_08_26)
    assert item.signal_date == date(2026, 8, 25)
    assert item.entry_open == pytest.approx(2.035)
    assert item.price_proved is True


def test_t_day_daily_and_late_available_rows_are_invisible(tmp_path):
    root = build_source_with_future_rows(tmp_path)
    item = load_arrival_input(root, root, "513100", ARRIVAL_2026_08_26)
    assert item.signal_date == date(2026, 8, 25)
    assert len(item.source_hashes) == 3


def test_loader_rejects_non_0935_or_non_shanghai_arrival(tmp_path):
    root = build_valid_source(tmp_path)
    with pytest.raises(SourceContractError, match="09:35"):
        load_arrival_input(root, root, "513100", ARRIVAL_2026_08_26.replace(minute=34))
    with pytest.raises(SourceContractError, match="Asia/Shanghai"):
        load_arrival_input(root, root, "513100", datetime(2026, 8, 26, 9, 35))


def test_zero_trade_price_is_audit_only(tmp_path, monkeypatch):
    monkeypatch.setattr(source_module, "calculate_rsi6", rsi_series_ending(24, 18, 21))
    item = load_arrival_input(*build_zero_trade_args(tmp_path))
    decision = detect_rsi_low_turn(item)
    assert decision.signal_detected is True
    assert decision.valid_event is False
    assert "price_unproved" in decision.reasons


def test_background_indicators_match_formal_pure_helpers(tmp_path):
    root = build_valid_source(tmp_path)
    item = load_arrival_input(root, root, "513100", ARRIVAL_2026_08_26)
    expected = calculate_formal_background_from_same_frame(root)
    assert item.background == pytest.approx(expected)
