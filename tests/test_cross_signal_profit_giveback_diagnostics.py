# -*- coding: utf-8 -*-
"""Tests for the observation-only profit-giveback diagnostic."""

import pytest


def _row(**overrides):
    from cross_signal_strategy.research.trade_quality_ledger import TradeQualityRow

    values = {
        "code": "513050",
        "buy_date": "2020-01-02",
        "sell_date": "2020-01-10",
        "sell_reason": "signal_sell",
        "market_group": "qdii",
        "realized_return_pct": -0.5,
        "holding_trade_days": 7,
        "holding_mfe": 0.06,
        "holding_mae": -0.02,
        "entry_mfe": {5: 0.04, 10: 0.06},
        "entry_mae": {5: -0.01, 10: -0.02},
        "first_profitable_close_offset": 1,
        "first_atr_barrier": "up_first",
        "post_sell_returns": {5: 0.01, 10: 0.02},
        "entry_atr_pct": 0.03,
    }
    values.update(overrides)
    return TradeQualityRow(**values)


def test_profit_giveback_observation_uses_one_entry_atr_and_fractional_units():
    from cross_signal_strategy.research.profit_giveback_diagnostics import (
        build_profit_giveback_observations,
    )

    observation = build_profit_giveback_observations([_row()])[0]

    assert observation.reached_one_entry_atr
    assert observation.peak_return == pytest.approx(0.06)
    assert observation.realized_return == pytest.approx(-0.005)
    assert observation.giveback_from_peak == pytest.approx(0.065)
    assert observation.round_trip_to_non_profit


def test_profit_giveback_summary_reports_fixed_2_3_4_percent_bands_by_year():
    from cross_signal_strategy.research.profit_giveback_diagnostics import (
        build_profit_giveback_report,
    )

    report = build_profit_giveback_report([
        _row(buy_date="2019-01-02", sell_date="2019-01-10", holding_mfe=0.025,
             realized_return_pct=-1.0, entry_atr_pct=0.02),
        _row(buy_date="2020-01-02", sell_date="2020-01-10", holding_mfe=0.035,
             realized_return_pct=1.0, entry_atr_pct=0.04),
        _row(buy_date="2021-01-04", sell_date="2021-01-12", holding_mfe=0.045,
             realized_return_pct=-0.2, entry_atr_pct=0.03),
    ])

    assert report.all_trades.count == 3
    assert report.fixed_peak_bands[0.02].reached_count == 3
    assert report.fixed_peak_bands[0.03].reached_count == 2
    assert report.fixed_peak_bands[0.04].reached_count == 1
    assert report.fixed_peak_bands[0.02].round_trip_count == 2
    assert report.by_year[2019].round_trip_count == 1
    assert report.by_year[2020].round_trip_count == 0
    assert report.by_year[2021].round_trip_count == 1


def test_profit_giveback_diagnostic_rejects_rows_outside_training_window():
    from cross_signal_strategy.research.profit_giveback_diagnostics import (
        build_profit_giveback_observations,
    )

    with pytest.raises(ValueError, match="training window"):
        build_profit_giveback_observations([
            _row(buy_date="2022-01-04", sell_date="2022-01-10"),
        ])
