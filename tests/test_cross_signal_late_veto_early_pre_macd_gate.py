# -*- coding: utf-8 -*-
"""Tests for the frozen candidate's paired JoinQuant result gate."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _run_config(*, candidate: bool) -> dict:
    return {
        "strategy_version": (
            "cross-v0.3.3-late-veto-early-pre-macd-candidate"
            if candidate
            else "cross-v0.3.3"
        ),
        "build_id": "20260822.3-candidate" if candidate else "20260822.2",
        "fingerprint": "f6b08195dd3d" if candidate else "77e44d93d255",
        "start_date": "2019-01-01",
        "end_date": "2021-12-31",
        "initial_cash": 20000,
        "frequency": "daily",
        "execution_time": "09:35",
        "commission_rate": 0.0003,
        "minimum_commission": 5,
        "slippage_rate": 0.001,
        "friction_profile": "nominal",
        "log_sha256": ("c" if candidate else "a") * 64,
        "trade_export_sha256": ("d" if candidate else "b") * 64,
    }


def _metrics(*, candidate: bool) -> dict:
    return {
        "total_return": 1.31 if candidate else 1.2925001,
        "max_drawdown": 0.06 if candidate else 0.0628,
        "win_rate": 0.59 if candidate else 0.5578947368421053,
        "closed_trade_count": 100 if candidate else 95,
        "profit_loss_ratio": 5.4 if candidate else 5.297,
        "annual_returns": {
            "2019": 0.30,
            "2020": 0.40,
            "2021": 0.20,
        },
        "positive_to_negative_round_trips": 30 if candidate else 31,
        "early_fills": 3 if candidate else 0,
        "early_fill_years": [2019, 2020] if candidate else [],
    }


def valid_training_payload() -> dict:
    return {
        "kind": "training_nominal",
        "baseline": {
            "config": _run_config(candidate=False),
            "metrics": _metrics(candidate=False),
        },
        "candidate": {
            "config": _run_config(candidate=True),
            "metrics": _metrics(candidate=True),
        },
    }


def write_pair(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def payload_for_kind(kind: str) -> dict:
    payload = valid_training_payload()
    payload["kind"] = kind
    windows = {
        "training_nominal": ("2019-01-01", "2021-12-31", "nominal"),
        "training_double_friction": (
            "2019-01-01", "2021-12-31", "double"
        ),
        "validation_2022_2023": ("2022-01-01", "2023-12-31", "nominal"),
        "validation_2024_latest": ("2024-01-01", "2026-09-01", "nominal"),
        "validation_2015_2018": ("2015-01-01", "2018-12-31", "nominal"),
        "validation_2010_2014": ("2010-01-01", "2014-12-31", "nominal"),
        "full_period": ("2010-01-01", "2026-09-01", "nominal"),
    }
    start, end, profile = windows[kind]
    for label in ("baseline", "candidate"):
        config = payload[label]["config"]
        config["start_date"] = start
        config["end_date"] = end
        config["friction_profile"] = profile
        if profile == "double":
            config["commission_rate"] = 0.0006
            config["minimum_commission"] = 10
            config["slippage_rate"] = 0.002
    if kind not in {"training_nominal", "full_period"}:
        for field in ("total_return", "max_drawdown", "win_rate"):
            payload["candidate"]["metrics"][field] = payload["baseline"][
                "metrics"
            ][field]
    return payload


def test_load_pair_preserves_decimal_precision_and_frozen_identities(tmp_path):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        load_paired_run,
    )

    pair = load_paired_run(write_pair(tmp_path, valid_training_payload()))

    assert pair.baseline.metrics.total_return == Decimal("1.2925001")
    assert pair.baseline.config.fingerprint == "77e44d93d255"
    assert pair.candidate.config.fingerprint == "f6b08195dd3d"


@pytest.mark.parametrize(
    "field,value",
    [
        ("fingerprint", "wrong"),
        ("start_date", "2019-01-02"),
        ("initial_cash", 10000),
        ("execution_time", "15:00"),
        ("commission_rate", 0.0004),
    ],
)
def test_invalid_or_unpaired_identity_is_rejected(tmp_path, field, value):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        load_paired_run,
    )

    payload = valid_training_payload()
    payload["candidate"]["config"][field] = value

    with pytest.raises(ValueError):
        load_paired_run(write_pair(tmp_path, payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("log_sha256", "abc"),
        ("trade_export_sha256", "z" * 64),
    ],
)
def test_malformed_evidence_hash_is_rejected(tmp_path, field, value):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        load_paired_run,
    )

    payload = valid_training_payload()
    payload["baseline"]["config"][field] = value

    with pytest.raises(ValueError, match="SHA-256"):
        load_paired_run(write_pair(tmp_path, payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("closed_trade_count", 95.5),
        ("closed_trade_count", -1),
        ("early_fills", -1),
        ("positive_to_negative_round_trips", -1),
    ],
)
def test_trade_counts_must_be_nonnegative_integers(tmp_path, field, value):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        load_paired_run,
    )

    payload = valid_training_payload()
    payload["candidate"]["metrics"][field] = value

    with pytest.raises(ValueError, match="nonnegative integer"):
        load_paired_run(write_pair(tmp_path, payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("win_rate", 1.0001),
        ("win_rate", -0.0001),
        ("max_drawdown", 1.0001),
        ("max_drawdown", -0.0001),
    ],
)
def test_rates_and_drawdown_must_stay_in_unit_interval(tmp_path, field, value):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        load_paired_run,
    )

    payload = valid_training_payload()
    payload["candidate"]["metrics"][field] = value

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_paired_run(write_pair(tmp_path, payload))


@pytest.mark.parametrize(
    "field",
    ["total_return", "profit_loss_ratio"],
)
def test_gate_metrics_must_be_finite(tmp_path, field):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        load_paired_run,
    )

    payload = valid_training_payload()
    payload["candidate"]["metrics"][field] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        load_paired_run(write_pair(tmp_path, payload))


def test_annual_returns_must_be_finite(tmp_path):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        load_paired_run,
    )

    payload = valid_training_payload()
    payload["candidate"]["metrics"]["annual_returns"]["2020"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        load_paired_run(write_pair(tmp_path, payload))


def test_training_gate_accepts_complete_nondegrading_improvement(tmp_path):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        evaluate_pair,
        load_paired_run,
    )

    decision = evaluate_pair(
        load_paired_run(write_pair(tmp_path, valid_training_payload()))
    )

    assert decision.passed is True
    assert decision.reasons == ()


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("total_return", 1.2925000, "return"),
        ("max_drawdown", 0.0628001, "drawdown"),
        ("win_rate", 0.5878947, "win-rate"),
        ("early_fills", 2, "early fills"),
        ("early_fill_years", [2020], "two years"),
        ("profit_loss_ratio", 2.9999, "profit/loss"),
        ("positive_to_negative_round_trips", 32, "positive-to-negative"),
    ],
)
def test_training_gate_rejects_each_independent_failure(
    tmp_path, field, value, reason
):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        evaluate_pair,
        load_paired_run,
    )

    payload = valid_training_payload()
    payload["candidate"]["metrics"][field] = value
    decision = evaluate_pair(load_paired_run(write_pair(tmp_path, payload)))

    assert decision.passed is False
    assert any(reason in item for item in decision.reasons)


def test_training_gate_requires_positive_return_in_each_training_year(tmp_path):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        evaluate_pair,
        load_paired_run,
    )

    payload = valid_training_payload()
    payload["candidate"]["metrics"]["annual_returns"]["2020"] = 0
    decision = evaluate_pair(load_paired_run(write_pair(tmp_path, payload)))

    assert decision.passed is False
    assert any("annual return" in item for item in decision.reasons)


def test_training_gate_must_beat_failed_standalone_late_veto(tmp_path):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        evaluate_pair,
        load_paired_run,
    )

    payload = valid_training_payload()
    payload["baseline"]["metrics"]["win_rate"] = 0.50
    payload["candidate"]["metrics"]["win_rate"] = 0.53
    decision = evaluate_pair(load_paired_run(write_pair(tmp_path, payload)))

    assert decision.passed is False
    assert any("standalone late-veto" in item for item in decision.reasons)


@pytest.mark.parametrize(
    "kind",
    [
        "training_nominal",
        "training_double_friction",
        "validation_2022_2023",
        "validation_2024_latest",
        "validation_2015_2018",
        "validation_2010_2014",
        "full_period",
    ],
)
def test_each_frozen_gate_kind_accepts_its_exact_passing_boundary(tmp_path, kind):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        evaluate_pair,
        load_paired_run,
    )

    decision = evaluate_pair(
        load_paired_run(write_pair(tmp_path, payload_for_kind(kind)))
    )

    assert decision.passed is True


def test_full_period_requires_three_percentage_point_win_rate_gain(tmp_path):
    from cross_signal_strategy.research.late_veto_early_pre_macd_gate import (
        evaluate_pair,
        load_paired_run,
    )

    payload = payload_for_kind("full_period")
    payload["candidate"]["metrics"]["win_rate"] = 0.5878947
    decision = evaluate_pair(load_paired_run(write_pair(tmp_path, payload)))

    assert decision.passed is False
    assert any("win-rate" in item for item in decision.reasons)


@pytest.mark.parametrize(
    "mutation,expected_code,expected_text",
    [
        (None, 0, "PASS"),
        (("metrics", "total_return", 1.0), 1, "FAIL"),
        (("config", "fingerprint", "wrong"), 2, "INVALID"),
    ],
)
def test_cli_distinguishes_pass_failure_and_invalid_evidence(
    tmp_path, mutation, expected_code, expected_text
):
    payload = valid_training_payload()
    if mutation is not None:
        section, field, value = mutation
        payload["candidate"][section][field] = value
    path = write_pair(tmp_path, payload)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cross_signal_strategy.research.late_veto_early_pre_macd_gate",
            str(path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_code
    assert expected_text in completed.stdout
