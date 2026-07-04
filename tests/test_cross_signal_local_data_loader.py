# -*- coding: utf-8 -*-
"""Tests for the cross-signal local training data loader."""

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_ROOT = pathlib.Path(r"G:\financial\history_data\cross_signal_train_2019_2021")


def test_loader_rejects_non_training_root():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    with pytest.raises(ValueError, match="approved training data root"):
        CrossSignalTrainingDataLoader(pathlib.Path(r"G:\financial\history_data\按年份合并"))


def test_loader_rejects_paths_inside_training_root_for_writes():
    from cross_signal_strategy.local_data_loader import assert_not_training_write_path

    with pytest.raises(ValueError, match="read-only"):
        assert_not_training_write_path(TRAIN_ROOT / "minute_1m" / "2019" / "510300.csv")


def test_loader_rejects_paths_inside_warmup_root_for_writes():
    from cross_signal_strategy.local_data_loader import assert_not_training_write_path

    warmup_root = pathlib.Path(r"G:\financial\history_data\cross_signal_warmup_2018")

    with pytest.raises(ValueError, match="read-only"):
        assert_not_training_write_path(warmup_root / "daily" / "2018" / "510300.csv")


def test_loader_returns_0935_minute_bar_from_training_data():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)
    row = loader.get_minute_bar("510300", "2019-01-02", "09:35")

    assert row["code"] == "510300"
    assert row["date"] == "2019-01-02"
    assert row["time"] == "09:35"
    assert row["close"] == pytest.approx(3.060)
    assert row["amount"] == pytest.approx(8222915.0)


def test_loader_rejects_dates_outside_training_window():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)

    with pytest.raises(ValueError, match="outside training window"):
        loader.get_minute_bar("510300", "2022-01-04", "09:35")


def test_loader_detects_out_of_window_dates_in_loaded_frame(tmp_path):
    from cross_signal_strategy.local_data_loader import assert_dates_in_training_window

    frame = pd.DataFrame({"date": ["2019-01-02", "2022-01-04"]})

    with pytest.raises(ValueError, match="outside training window"):
        assert_dates_in_training_window(frame)
