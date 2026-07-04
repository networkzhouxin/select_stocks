# -*- coding: utf-8 -*-
"""Tests for full training-window local replay helpers."""

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_ROOT = pathlib.Path(r"G:\financial\history_data\cross_signal_train_2019_2021")


def test_training_trade_dates_are_read_from_training_daily_data_only():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_training_run import get_training_trade_dates

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)

    dates = get_training_trade_dates(loader, reference_code="510300")

    assert dates[0] == "2019-01-02"
    assert dates[-1] == "2021-12-31"
    assert len(dates) == 730
    assert all("2019-01-01" <= d <= "2021-12-31" for d in dates)
    assert dates == sorted(dates)


def test_full_training_replay_completes_without_date_or_position_violations():
    from cross_signal_strategy.local_data_loader import CrossSignalTrainingDataLoader
    from cross_signal_strategy.local_training_run import run_training_replay

    loader = CrossSignalTrainingDataLoader(TRAIN_ROOT)

    summary = run_training_replay(loader, initial_cash=20000.0)

    assert summary.start_date == "2019-01-02"
    assert summary.end_date == "2021-12-31"
    assert summary.trading_days == 730
    assert summary.start_value == 20000.0
    assert summary.end_value > 0
    assert summary.max_holdings <= 3
    assert summary.max_drawdown >= 0
    assert summary.buy_count >= 0
    assert summary.sell_count >= 0
    assert all("2019-01-01" <= d <= "2021-12-31" for d in summary.order_dates)
