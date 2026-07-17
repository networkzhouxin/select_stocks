# -*- coding: utf-8 -*-
"""Tests for the isolated point-in-time underlying-index data contract."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _frame(
    code="513100",
    source_id="NDX",
    session_dates=("2019-01-02", "2019-01-03"),
    available_at=(
        "2019-01-03T06:15:00+08:00",
        "2019-01-04T06:15:00+08:00",
    ),
    closes=(100.0, 101.0),
    is_final=(True, True),
):
    return pd.DataFrame({
        "etf_code": [code] * len(session_dates),
        "source_id": [source_id] * len(session_dates),
        "session_date": list(session_dates),
        "available_at": list(available_at),
        "close": list(closes),
        "is_final": list(is_final),
    })


def test_formal_qdii_underlying_mapping_is_fixed():
    from cross_signal_strategy.research.underlying_market_data import UNDERLYING_SPECS

    assert set(UNDERLYING_SPECS) == {"513100", "513500", "513050", "513880"}
    assert UNDERLYING_SPECS["513100"].source_id == "NDX"
    assert UNDERLYING_SPECS["513500"].source_id == "SPX"
    assert UNDERLYING_SPECS["513050"].source_id == "H30533"
    assert UNDERLYING_SPECS["513880"].source_id == "N225"


def test_underlying_loader_rejects_non_approved_root(tmp_path):
    from cross_signal_strategy.research.underlying_market_data import (
        UnderlyingMarketDataLoader,
    )

    with pytest.raises(ValueError, match="approved underlying-index data root"):
        UnderlyingMarketDataLoader(tmp_path)


def test_loader_combines_warmup_and_training_but_returns_only_0935_visible_rows(
    tmp_path,
    monkeypatch,
):
    import cross_signal_strategy.research.underlying_market_data as module

    warmup_path = tmp_path / "warmup" / "2018" / "513100.csv"
    training_path = tmp_path / "training" / "2019" / "513100.csv"
    warmup_path.parent.mkdir(parents=True)
    training_path.parent.mkdir(parents=True)
    _frame(
        session_dates=("2018-12-27", "2018-12-28"),
        available_at=(
            "2018-12-28T06:15:00+08:00",
            "2018-12-29T06:15:00+08:00",
        ),
        closes=(98.0, 99.0),
    ).to_csv(warmup_path, index=False)
    _frame(
        session_dates=("2019-01-02", "2019-01-03", "2019-01-04"),
        available_at=(
            "2019-01-03T06:15:00+08:00",
            "2019-01-04T06:15:00+08:00",
            "2019-01-07T10:00:00+08:00",
        ),
        closes=(100.0, 101.0, 999.0),
        is_final=(True, True, True),
    ).to_csv(training_path, index=False)
    monkeypatch.setattr(module, "APPROVED_UNDERLYING_ROOT", tmp_path)

    loader = module.UnderlyingMarketDataLoader(tmp_path)
    visible = loader.load_history("513100", "2019-01-07T09:35:00+08:00")

    assert list(visible["session_date"].dt.strftime("%Y-%m-%d")) == [
        "2018-12-27",
        "2018-12-28",
        "2019-01-02",
        "2019-01-03",
    ]
    assert visible["close"].max() == pytest.approx(101.0)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"etf_code": ["513100"]}), "missing columns"),
        (_frame(source_id="SPX"), "source mismatch"),
        (_frame(session_dates=("2022-01-03", "2022-01-04")), "approved source dates"),
        (
            _frame(available_at=("2019-01-03 06:15:00", "2019-01-04 06:15:00")),
            "timezone-aware available_at",
        ),
        (_frame(closes=(100.0, 0.0)), "positive close"),
        (_frame(is_final=(True, False)), "final observations"),
        (
            _frame(session_dates=("2019-01-02", "2019-01-02")),
            "duplicate session_date",
        ),
    ],
)
def test_validate_underlying_frame_rejects_bad_source_rows(frame, message):
    from cross_signal_strategy.research.underlying_market_data import (
        validate_underlying_frame,
    )

    with pytest.raises(ValueError, match=message):
        validate_underlying_frame(frame, expected_code="513100")


def test_direction_uses_only_final_values_available_by_china_0935():
    from cross_signal_strategy.research.underlying_market_data import (
        select_underlying_direction,
    )

    frame = _frame(
        session_dates=("2019-01-02", "2019-01-03", "2019-01-04"),
        available_at=(
            "2019-01-03T06:15:00+08:00",
            "2019-01-04T06:15:00+08:00",
            "2019-01-07T10:00:00+08:00",
        ),
        closes=(100.0, 102.0, 50.0),
        is_final=(True, True, True),
    )

    observation = select_underlying_direction(
        frame,
        code="513100",
        decision_at="2019-01-07T09:35:00+08:00",
    )

    assert observation is not None
    assert observation.previous_session_date == "2019-01-02"
    assert observation.latest_session_date == "2019-01-03"
    assert observation.one_session_return == pytest.approx(0.02)
    assert observation.confirmed is True


def test_japan_same_day_close_published_after_0935_is_future_and_excluded():
    from cross_signal_strategy.research.underlying_market_data import (
        select_underlying_direction,
    )

    frame = _frame(
        code="513880",
        source_id="N225",
        session_dates=("2021-06-01", "2021-06-02", "2021-06-03"),
        available_at=(
            "2021-06-01T15:30:00+09:00",
            "2021-06-02T15:30:00+09:00",
            "2021-06-03T15:30:00+09:00",
        ),
        closes=(100.0, 98.0, 110.0),
        is_final=(True, True, True),
    )

    observation = select_underlying_direction(
        frame,
        code="513880",
        decision_at="2021-06-03T09:35:00+08:00",
    )

    assert observation is not None
    assert observation.latest_session_date == "2021-06-02"
    assert observation.one_session_return == pytest.approx(-0.02)
    assert observation.confirmed is False


def test_future_session_date_is_excluded_even_if_source_timestamp_is_badly_early():
    from cross_signal_strategy.research.underlying_market_data import (
        select_underlying_direction,
    )

    frame = _frame(
        session_dates=("2019-01-02", "2019-01-03", "2019-01-08"),
        available_at=(
            "2019-01-03T06:15:00+08:00",
            "2019-01-04T06:15:00+08:00",
            "2019-01-07T06:15:00+08:00",
        ),
        closes=(100.0, 101.0, 999.0),
        is_final=(True, True, True),
    )

    observation = select_underlying_direction(
        frame,
        code="513100",
        decision_at="2019-01-07T09:35:00+08:00",
    )

    assert observation is not None
    assert observation.latest_session_date == "2019-01-03"
    assert observation.one_session_return == pytest.approx(0.01)


def test_2018_warmup_can_supply_previous_close_but_decision_must_be_training():
    from cross_signal_strategy.research.underlying_market_data import (
        select_underlying_direction,
    )

    frame = _frame(
        session_dates=("2018-12-27", "2018-12-28"),
        available_at=(
            "2018-12-28T06:15:00+08:00",
            "2018-12-29T06:15:00+08:00",
        ),
        closes=(100.0, 101.0),
        is_final=(True, True),
    )

    observation = select_underlying_direction(
        frame,
        code="513100",
        decision_at="2019-01-02T09:35:00+08:00",
    )
    assert observation is not None
    assert observation.confirmed is True

    with pytest.raises(ValueError, match="2019-2021 training window"):
        select_underlying_direction(
            frame,
            code="513100",
            decision_at="2018-12-28T09:35:00+08:00",
        )


def test_decision_timestamp_must_be_exact_china_0935():
    from cross_signal_strategy.research.underlying_market_data import (
        select_underlying_direction,
    )

    with pytest.raises(ValueError, match="09:35 Asia/Shanghai"):
        select_underlying_direction(
            _frame(),
            code="513100",
            decision_at="2019-01-07T10:00:00+08:00",
        )
