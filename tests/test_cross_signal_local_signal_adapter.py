# -*- coding: utf-8 -*-
"""Tests for local T-1 daily signal adaptation."""

import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))
TRAIN_ROOT = pathlib.Path(r"G:\financial\history_data\cross_signal_train_2019_2021")


def test_signal_frame_uses_previous_trading_day_only():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    adapter = LocalSignalAdapter(CrossSignalTrainingDataLoader(TRAIN_ROOT))

    frame, signal_date = adapter.load_signal_frame("510300", "2019-01-03")

    assert signal_date == "2019-01-02"
    assert frame["date"].max() == "2019-01-02"
    assert "2019-01-03" not in set(frame["date"])


def test_signal_score_reports_short_data_without_using_future_rows():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    adapter = LocalSignalAdapter(CrossSignalTrainingDataLoader(TRAIN_ROOT))

    score, reason = adapter.score("510300", "2019-01-03", return_reason=True)

    assert score is None
    assert reason.startswith("short_data:")


def test_signal_score_matches_strategy_snapshot_scoring_after_lookback():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    adapter = LocalSignalAdapter(CrossSignalTrainingDataLoader(TRAIN_ROOT))

    score, reason = adapter.score("510300", "2019-07-01", return_reason=True)

    assert reason is None
    assert score["code"] == "510300"
    assert score["current_date"] == "2019-07-01"
    assert score["signal_date"] == "2019-06-28"
    assert score["max_data_date"] == "2019-06-28"
    assert isinstance(score["buy_score"], (int, float))
    assert isinstance(score["sell_score"], (int, float))
    assert score["close"] == pytest.approx(3.859)
    assert score["atr"] > 0
