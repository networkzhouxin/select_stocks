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
WARMUP_ROOT = pathlib.Path(r"G:\financial\history_data\cross_signal_warmup_2018")


def test_signal_frame_uses_previous_trading_day_only():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    adapter = LocalSignalAdapter(CrossSignalTrainingDataLoader(TRAIN_ROOT))

    frame, signal_date = adapter.load_signal_frame("510300", "2019-01-03")

    assert signal_date == "2019-01-02"
    assert frame["date"].max() == "2019-01-02"
    assert "2019-01-03" not in set(frame["date"])


def test_signal_frame_uses_2018_warmup_without_leaking_current_day():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    adapter = LocalSignalAdapter(CrossSignalTrainingDataLoader(TRAIN_ROOT), warmup_root=WARMUP_ROOT)

    frame, signal_date = adapter.load_signal_frame("510300", "2019-01-03")

    assert signal_date == "2019-01-02"
    assert frame["date"].min().startswith("2018-")
    assert frame["date"].max() == "2019-01-02"
    assert "2019-01-03" not in set(frame["date"])


def test_missing_warmup_file_does_not_block_listed_2019_data():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    adapter = LocalSignalAdapter(CrossSignalTrainingDataLoader(TRAIN_ROOT), warmup_root=WARMUP_ROOT)

    frame, signal_date = adapter.load_signal_frame("159985", "2019-12-10")

    assert signal_date == "2019-12-09"
    assert frame["date"].min().startswith("2019-")


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


def test_signal_score_allows_new_listing_when_required_indicators_are_valid():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    adapter = LocalSignalAdapter(CrossSignalTrainingDataLoader(TRAIN_ROOT), warmup_root=WARMUP_ROOT)

    score, reason = adapter.score("513880", "2019-10-18", return_reason=True)

    assert reason is None
    assert score["code"] == "513880"
    assert score["signal_date"] == "2019-10-17"
    assert score["buy_score"] == 80
    assert score["sell_score"] == 6


def test_signal_score_allows_listing_before_ma60_when_core_indicators_are_valid():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    adapter = LocalSignalAdapter(CrossSignalTrainingDataLoader(TRAIN_ROOT), warmup_root=WARMUP_ROOT)

    score, reason = adapter.score("159985", "2020-03-03", return_reason=True)

    assert reason is None
    assert score["code"] == "159985"
    assert score["signal_date"] == "2020-03-02"
    assert score["buy_score"] == 70
    assert score["trend_score"] == 0
    assert score["sell_score"] == 0


def test_signal_score_suppresses_sub_float_falling_ma10_artifact():
    from cross_signal_strategy.local_adjustment import default_training_adjustment_factors
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    adapter = LocalSignalAdapter(
        CrossSignalTrainingDataLoader(TRAIN_ROOT),
        warmup_root=WARMUP_ROOT,
        adjustment_factors=default_training_adjustment_factors(),
    )

    score, reason = adapter.score("159928", "2019-11-13", return_reason=True)

    assert reason is None
    assert score["signal_date"] == "2019-11-12"
    assert not score["close_below_falling_ma10"]
    assert score["sell_score"] == 24


def test_signal_frame_applies_current_day_adjustment_without_future_events():
    import pandas as pd

    from cross_signal_strategy.local_adjustment import LocalAdjustmentFactors

    frame = pd.DataFrame(
        {
            "date": ["2020-01-15", "2020-01-16", "2020-01-17"],
            "open": [2.90, 2.95, 2.80],
            "high": [2.92, 2.96, 2.82],
            "low": [2.88, 2.94, 2.78],
            "close": [2.91, 2.947, 2.784],
            "volume": [1000.0, 1200.0, 1300.0],
        }
    )
    factors = LocalAdjustmentFactors.from_records(
        [
            {"code": "510880", "ex_date": "2020-01-17", "ex_factor": 1.0513740030198886},
            {"code": "510880", "ex_date": "2021-01-18", "ex_factor": 1.0543561221399267},
        ]
    )

    adjusted = factors.adjust_daily_frame(frame, "510880", "2020-01-17")

    assert adjusted.loc[0, "close"] == pytest.approx(2.91 / 1.0513740030198886)
    assert adjusted.loc[1, "close"] == pytest.approx(2.947 / 1.0513740030198886)
    assert adjusted.loc[2, "close"] == pytest.approx(2.784)
    assert adjusted.loc[1, "volume"] == pytest.approx(1200.0)


def test_local_signal_adapter_can_align_ex_dividend_signal_close():
    from cross_signal_strategy.local_adjustment import LocalAdjustmentFactors
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_signal_adapter import LocalSignalAdapter

    factors = LocalAdjustmentFactors.from_records(
        [{"code": "510880", "ex_date": "2020-01-17", "ex_factor": 1.0513740030198886}]
    )
    adapter = LocalSignalAdapter(
        CrossSignalTrainingDataLoader(TRAIN_ROOT),
        warmup_root=WARMUP_ROOT,
        adjustment_factors=factors,
    )

    score, reason = adapter.score("510880", "2020-01-17", return_reason=True)

    assert reason is None
    assert score["signal_date"] == "2020-01-16"
    assert score["close"] == pytest.approx(2.803, abs=0.001)
