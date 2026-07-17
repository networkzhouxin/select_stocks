# -*- coding: utf-8 -*-
"""Tests for audited acquisition of the four pre-registered source indices."""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _raw_frame(dates=("2019-01-02", "2019-01-03"), closes=(100.0, 101.0)):
    return pd.DataFrame({"session_date": list(dates), "close": list(closes)})


def test_source_registry_locks_four_indices_and_training_only_queries():
    from cross_signal_strategy.research.underlying_source_acquisition import (
        SOURCE_PLANS,
    )

    assert set(SOURCE_PLANS) == {"513100", "513500", "513050", "513880"}
    assert SOURCE_PLANS["513100"].provider == "FRED"
    assert SOURCE_PLANS["513100"].locator == "NASDAQ100"
    assert SOURCE_PLANS["513500"].locator == "SP500"
    assert SOURCE_PLANS["513050"].provider == "CSI"
    assert SOURCE_PLANS["513050"].locator == "H30533"
    assert SOURCE_PLANS["513880"].locator == "NIKKEI225"
    assert {
        plan.request_start for plan in SOURCE_PLANS.values()
    } == {"2018-01-01"}
    assert {
        plan.request_end for plan in SOURCE_PLANS.values()
    } == {"2021-12-31"}


def test_fred_parser_drops_source_holidays_without_forward_filling():
    from cross_signal_strategy.research.underlying_source_acquisition import (
        parse_fred_csv,
    )

    text = "observation_date,NASDAQ100\n2019-01-02,6329.96\n2019-01-03,.\n2019-01-04,6422.67\n"

    frame = parse_fred_csv(text, series_id="NASDAQ100")

    assert list(frame["session_date"].dt.strftime("%Y-%m-%d")) == [
        "2019-01-02",
        "2019-01-04",
    ]
    assert list(frame["close"]) == pytest.approx([6329.96, 6422.67])


def test_source_normalizer_rejects_validation_or_pre_warmup_dates():
    from cross_signal_strategy.research.underlying_source_acquisition import (
        normalize_raw_history,
    )

    with pytest.raises(ValueError, match="2018-2021 source boundary"):
        normalize_raw_history(_raw_frame(("2022-01-03",), (100.0,)))
    with pytest.raises(ValueError, match="2018-2021 source boundary"):
        normalize_raw_history(_raw_frame(("2017-12-29",), (100.0,)))


def test_csindex_normalizer_uses_only_official_date_and_close_columns():
    from cross_signal_strategy.research.underlying_source_acquisition import (
        normalize_csindex_history,
    )

    source = pd.DataFrame({
        "日期": [pd.Timestamp("2019-01-02").date(), pd.Timestamp("2019-01-03").date()],
        "指数代码": ["H30533", "H30533"],
        "收盘": [6803.01, 6606.44],
        "涨跌幅": [0.0, -2.89],
    })

    frame = normalize_csindex_history(source)

    assert list(frame.columns) == ["session_date", "close"]
    assert list(frame["session_date"].dt.strftime("%Y-%m-%d")) == [
        "2019-01-02",
        "2019-01-03",
    ]
    assert list(frame["close"]) == pytest.approx([6803.01, 6606.44])


def test_fetchers_lock_symbol_and_date_range_without_reading_validation_data():
    from cross_signal_strategy.research.underlying_source_acquisition import (
        fetch_csindex_history,
        fetch_fred_history,
    )

    fred_calls = []

    class Response:
        text = "observation_date,NASDAQ100\n2019-01-02,6329.96\n"

        def raise_for_status(self):
            return None

    def fake_get(url, params, timeout):
        fred_calls.append((url, params, timeout))
        return Response()

    csi_calls = []

    def fake_csi(**kwargs):
        csi_calls.append(kwargs)
        return pd.DataFrame({"日期": ["2019-01-02"], "收盘": [6803.01]})

    fred = fetch_fred_history("513100", http_get=fake_get)
    csi = fetch_csindex_history(fetcher=fake_csi)

    assert len(fred) == 1
    assert len(csi) == 1
    assert fred_calls[0][1] == {
        "id": "NASDAQ100",
        "cosd": "2018-01-01",
        "coed": "2021-12-31",
    }
    assert csi_calls == [{
        "symbol": "H30533",
        "start_date": "20180101",
        "end_date": "20211231",
    }]


def test_approved_publication_policies_are_timezone_aware_and_conservative():
    from cross_signal_strategy.research.underlying_source_acquisition import (
        apply_approved_availability_policy,
    )

    ndx = apply_approved_availability_policy("513100", _raw_frame())
    n225 = apply_approved_availability_policy("513880", _raw_frame())

    assert ndx.loc[0, "available_at"].isoformat() == "2019-01-02T22:15:00+00:00"
    assert n225.loc[0, "available_at"].isoformat() == "2019-01-02T07:00:00+00:00"
    assert ndx["is_final"].eq(True).all()
    assert n225["is_final"].eq(True).all()


@pytest.mark.parametrize("code", ["513500", "513050"])
def test_unproven_publication_time_blocks_formal_contract_rows(code):
    from cross_signal_strategy.research.underlying_source_acquisition import (
        AvailabilityEvidenceMissing,
        apply_approved_availability_policy,
    )

    with pytest.raises(AvailabilityEvidenceMissing, match=code):
        apply_approved_availability_policy(code, _raw_frame())


def test_contract_bundle_is_all_or_nothing_when_any_policy_is_blocked():
    from cross_signal_strategy.research.underlying_source_acquisition import (
        AvailabilityEvidenceMissing,
        build_contract_bundle,
    )

    frames = {code: _raw_frame() for code in ("513100", "513500", "513050", "513880")}

    with pytest.raises(AvailabilityEvidenceMissing, match="513050,513500"):
        build_contract_bundle(frames)


def test_raw_staging_manifest_has_hashes_and_cannot_target_approved_root(
    tmp_path,
    monkeypatch,
):
    import cross_signal_strategy.research.underlying_source_acquisition as module

    approved = tmp_path / "approved"
    staging = tmp_path / "staging"
    monkeypatch.setattr(module, "APPROVED_UNDERLYING_ROOT", approved)
    frames = {code: _raw_frame() for code in module.SOURCE_PLANS}

    manifest_path = module.write_raw_staging_bundle(
        frames,
        staging,
        acquired_at="2026-07-18T10:00:00+08:00",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert manifest["formal_publishable"] is False
    assert manifest["blocked_codes"] == ["513050", "513500"]
    assert set(manifest["files"]) == set(module.SOURCE_PLANS)
    assert all(len(item["sha256"]) == 64 for item in manifest["files"].values())
    assert all((staging / item["path"]).exists() for item in manifest["files"].values())

    with pytest.raises(ValueError, match="approved immutable root"):
        module.write_raw_staging_bundle(
            frames,
            approved,
            acquired_at="2026-07-18T10:00:00+08:00",
        )


def test_acquisition_runner_collects_then_stages_without_publishing(tmp_path):
    from cross_signal_strategy.research.underlying_source_acquisition import (
        run_source_acquisition,
    )

    class Response:
        def __init__(self, series_id):
            self.text = (
                "observation_date,%s\n2019-01-02,100.0\n2019-01-03,101.0\n"
                % series_id
            )

        def raise_for_status(self):
            return None

    def fake_get(url, params, timeout):
        return Response(params["id"])

    def fake_csi(**kwargs):
        return pd.DataFrame({
            "日期": ["2019-01-02", "2019-01-03"],
            "收盘": [6803.01, 6606.44],
        })

    manifest_path = run_source_acquisition(
        staging_root=tmp_path / "staging",
        acquired_at="2026-07-18T10:00:00+08:00",
        http_get=fake_get,
        csindex_fetcher=fake_csi,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["formal_publishable"] is False
    assert manifest["blocked_codes"] == ["513050", "513500"]
    assert not (tmp_path / "staging" / "warmup").exists()
    assert not (tmp_path / "staging" / "training").exists()
