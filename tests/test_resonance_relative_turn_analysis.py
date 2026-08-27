import importlib.util
import json
import os
import pathlib
import sys
import types
from datetime import datetime
from unittest import mock

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    ROOT / "resonance_reversal_strategy" / "research"
    / "analyze_relative_turn_observations.py"
)
spec = importlib.util.spec_from_file_location("relative_analyzer", ANALYZER_PATH)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)

STRATEGY_PATH = ROOT / "resonance_reversal_strategy" / "smart_trade_joinquant_resonance_reversal_etf.py"
sys.modules.setdefault("jqdata", types.ModuleType("jqdata"))
strategy_spec = importlib.util.spec_from_file_location("relative_builder_strategy", STRATEGY_PATH)
strategy = importlib.util.module_from_spec(strategy_spec)
strategy_spec.loader.exec_module(strategy)


BUILD = "20260827.4"
FINGERPRINT = "f47d32b87be6d926"


def make_order_path(log_date="2021-01-05"):
    return [
        {
            "event": "order_transition", "_log_date": log_date,
            "_log_timestamp": log_date + "T09:35:00",
            "_ordinal": index, "side": "BUY" if index % 2 else "SELL",
            "code": "510300.XSHG", "outcome": "FILLED",
            "before_amount": 0 if index % 2 else 100,
            "after_amount": 100 if index % 2 else 0,
        }
        for index in range(1, 139)
    ]


def make_initialized_record():
    return {
        "event": "strategy_initialized", "build": BUILD,
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "relative_observation_fingerprint": FINGERPRINT,
    }


def make_relative_record(index, direction="BUY_TURN", branch=None, code=None):
    year = 2019 + index // 10
    branch = branch or (
        "HARD_BOLL_SOFT_OSC" if index % 2 else "SOFT_ALL_THREE"
    )
    code = code or ("510300.XSHG", "159915.XSHE", "518880.XSHG")[index % 3]
    signal_date = "%04d-01-%02d" % (year, index % 10 + 2)
    if branch == "HARD_BOLL_SOFT_OSC":
        supporters = ["BOLL", "RSI"]
        source_map = {"BOLL": "HARD", "RSI": "RELATIVE"}
    else:
        supporters = ["BOLL", "KDJ", "RSI"]
        source_map = {indicator: "RELATIVE" for indicator in supporters}
    return {
        "event": "relative_resonance_observation",
        "relative_observation_id": "RELATIVE:%02d" % index,
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": branch,
        "code": code,
        "direction": direction,
        "signal_date": signal_date,
        "expires_date": signal_date,
        "supporters": supporters,
        "supporter_event_dates": {
            indicator: signal_date for indicator in supporters
        },
        "hard_or_relative_source_by_indicator": source_map,
        "build": BUILD,
        "relative_observation_fingerprint": FINGERPRINT,
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "event_close": 10.0,
    }


def make_outcome(record, horizon, value):
    adjusted = value if record["direction"] == "BUY_TURN" else -value
    return {
        "event": "observation_outcome",
        "resonance_id": record["relative_observation_id"],
        "relative_observation_id": record["relative_observation_id"],
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": record["branch"], "direction": record["direction"],
        "code": record["code"], "event_date": record["signal_date"],
        "horizon": horizon, "build": BUILD,
        "relative_observation_fingerprint": FINGERPRINT,
        "supporters": record["supporters"],
        "supporter_event_dates": record["supporter_event_dates"],
        "hard_or_relative_source_by_indicator": (
            record["hard_or_relative_source_by_indicator"]
        ),
        "outcome": {
            "status": "RECORDED", "closing_date": record["signal_date"],
            "closing_price": record["event_close"] * (1.0 + value),
            "return": value, "direction_adjusted_return": adjusted,
        },
    }


def make_candidate_records():
    records = [make_initialized_record()]
    for index in range(30):
        record = make_relative_record(index)
        records.append(record)
        records.extend(make_outcome(record, horizon, value) for horizon, value in (
            (1, 0.005), (3, 0.01), (5, 0.02),
        ))
    records.extend(make_order_path())
    records.append({
        "event": "portfolio_summary", "closing_date": "2021-12-31",
        "total_value": 23856.40, "_log_timestamp": "2021-12-31T15:30:00",
    })
    return records


def make_baseline_records():
    records = [{
        "event": "strategy_initialized", "build": "20260827.3",
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
    }]
    records.extend(make_order_path())
    records.append({
        "event": "portfolio_summary", "closing_date": "2021-12-31",
        "total_value": 23856.40, "_log_timestamp": "2021-12-31T15:30:00",
    })
    for index in range(30):
        resonance_id = "FORMAL:%02d" % index
        records.append({
            "event": "resonance_decision", "accepted": True,
            "reason": "COMPLETE_RESONANCE", "resonance_id": resonance_id,
            "code": "510300.XSHG", "direction": "BUY_TURN",
            "signal_date": "2021-01-05",
        })
        records.append({
            "event": "observation_outcome", "resonance_id": resonance_id,
            "code": "510300.XSHG", "event_date": "2021-01-05",
            "horizon": 5,
            "outcome": {
                "status": "RECORDED", "closing_date": "2021-01-05",
                "return": 0.01,
            },
        })
    return records


def test_parser_accepts_plain_and_html_escaped_json_and_skips_noise():
    plain = (
        '2021-01-05 09:35:00 - INFO - '
        '{"event":"relative_resonance_observation","signal_date":"2021-01-04"}'
    )
    escaped = plain.replace('"', "&quot;")
    first = analyzer.parse_joinquant_log_line(plain, 1)
    second = analyzer.parse_joinquant_log_line(escaped, 2)
    assert first["event"] == second["event"]
    assert first["_log_date"] == "2021-01-05"
    assert second["_ordinal"] == 2
    assert analyzer.parse_joinquant_log_line("", 3) is None
    assert analyzer.parse_joinquant_log_line("ordinary output", 3) is None
    assert analyzer.parse_joinquant_log_line("2021-01-05 - {bad json", 4) is None
    assert analyzer.parse_joinquant_log_line("2021-01-05 - []", 5) is None


def test_load_records_skips_bad_lines_and_preserves_source_file(tmp_path):
    log_path = tmp_path / "candidate.log"
    content = "\n".join((
        "noise",
        '2021-01-05 - INFO - {"event":"one"}',
        '2021-01-05 - INFO - {"event":',
        '2021-01-06 - INFO - {"event":"two"}',
    ))
    log_path.write_text(content, encoding="utf-8")

    records = analyzer.load_log_records([log_path])

    assert [record["event"] for record in records] == ["one", "two"]
    assert [record["_ordinal"] for record in records] == [2, 4]
    assert log_path.read_text(encoding="utf-8") == content


def test_analyzer_reports_validation_period_qualified_observation_as_error():
    candidate = [make_initialized_record(), {
        "event": "relative_resonance_observation",
        "relative_observation_id": "RELATIVE:2022",
        "signal_date": "2022-01-04", "expires_date": "2022-01-04",
        "build": BUILD, "relative_observation_fingerprint": FINGERPRINT,
    }]

    report = analyzer.analyze_records(candidate, [])

    assert any("outside 2019-2021: 2022-01-04" in error
               for error in report["data_quality"]["errors"])
    assert report["continue_candidate"] is False


def test_empty_input_is_reported_as_incomplete_without_writing_state():
    report = analyzer.analyze_records([], [])

    assert report["metrics"]["candidate_count"] == 0
    assert report["data_quality"]["errors"] == [
        "missing baseline strategy_initialized record",
        "missing candidate strategy_initialized record",
    ]
    assert report["continue_candidate"] is False


def test_filled_order_path_requires_exact_date_side_code_and_amounts():
    baseline = [{
        "event": "order_transition", "_log_date": "2021-01-05",
        "_ordinal": 1, "side": "BUY", "code": "510300.XSHG",
        "outcome": "FILLED", "before_amount": 0, "after_amount": 100,
    }]
    changed = [dict(baseline[0], after_amount=200)]

    assert analyzer.extract_filled_order_path(baseline) != (
        analyzer.extract_filled_order_path(changed)
    )


def test_frozen_report_passes_only_when_every_gate_and_path_match():
    report = analyzer.analyze_records(
        make_candidate_records(), make_baseline_records(),
    )

    assert report["metrics"]["candidate_count"] == 30
    assert report["metrics"]["year_counts"] == {
        "2019": 10, "2020": 10, "2021": 10,
    }
    assert report["metrics"]["direction_counts"] == {
        "BUY_TURN": 30, "SELL_TURN": 0,
    }
    assert report["metrics"]["etf_counts"] == {
        "159915.XSHE": 10, "510300.XSHG": 10, "518880.XSHG": 10,
    }
    assert report["metrics"]["horizon_5"]["median"] == pytest.approx(0.02)
    assert report["metrics"]["horizon_5"]["hit_rate"] == pytest.approx(1.0)
    assert all(report["gates"].values())
    assert report["continue_candidate"] is True


def test_analysis_keeps_branches_directions_and_horizons_separate():
    first = make_relative_record(
        0, direction="BUY_TURN", branch="SOFT_ALL_THREE", code="518880.XSHG",
    )
    second = make_relative_record(
        1, direction="SELL_TURN", branch="HARD_BOLL_SOFT_OSC", code="510300.XSHG",
    )
    candidate = [make_initialized_record(), first, second]
    candidate.extend((
        make_outcome(first, 1, -0.02), make_outcome(first, 3, 0.01),
        make_outcome(first, 5, 0.03), make_outcome(second, 1, 0.02),
        make_outcome(second, 3, -0.01), make_outcome(second, 5, -0.04),
    ))

    report = analyzer.analyze_records(candidate, [])

    assert report["metrics"]["direction_counts"] == {
        "BUY_TURN": 1, "SELL_TURN": 1,
    }
    assert report["metrics"]["by_branch"]["SOFT_ALL_THREE"]["horizon_5"]["median"] == pytest.approx(0.03)
    assert report["metrics"]["by_branch"]["HARD_BOLL_SOFT_OSC"]["horizon_5"]["median"] == pytest.approx(0.04)
    assert report["metrics"]["horizon_1"]["median"] == pytest.approx(-0.02)
    assert report["metrics"]["horizon_3"]["median"] == pytest.approx(0.01)


def test_duplicate_registration_and_outcome_are_reported_once():
    candidate = make_candidate_records()
    observation = next(
        record for record in candidate
        if record.get("relative_observation_id") == "RELATIVE:00"
        and record["event"] == "relative_resonance_observation"
    )
    outcome = next(
        record for record in candidate
        if record.get("relative_observation_id") == "RELATIVE:00"
        and record.get("horizon") == 5
    )
    candidate.extend((dict(observation), dict(outcome)))

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["metrics"]["candidate_count"] == 30
    assert report["data_quality"]["errors"] == [
        "duplicate relative candidate: RELATIVE:00",
        "duplicate relative outcome: RELATIVE:00/5",
    ]
    assert report["gates"]["data_quality_complete"] is False


def test_foreign_relative_and_formal_observation_logs_are_ignored_and_reported():
    candidate = make_candidate_records()
    foreign = make_relative_record(99)
    foreign["build"] = "20260827.5"
    foreign_outcome = make_outcome(foreign, 1, 0.8)
    foreign_outcome["build"] = "20260827.5"
    foreign_fingerprint = make_relative_record(8)
    foreign_fingerprint["relative_observation_fingerprint"] = "other-contract"
    foreign_fingerprint_outcome = make_outcome(foreign_fingerprint, 1, 0.7)
    foreign_fingerprint_outcome["relative_observation_fingerprint"] = "other-contract"
    candidate.extend((foreign, foreign_outcome, foreign_fingerprint,
                      foreign_fingerprint_outcome, {
        "event": "observation_outcome", "resonance_id": "FORMAL:foreign",
        "horizon": 5, "outcome": {"status": "RECORDED", "return": 0.9},
    }))

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["metrics"]["candidate_count"] == 30
    assert report["data_quality"]["ignored_record_counts"] == {
        "formal_observation_outcome": 1,
        "relative_build_mismatch": 2,
        "relative_fingerprint_mismatch": 2,
    }
    assert any("build mismatch" in error for error in report["data_quality"]["errors"])
    assert any("fingerprint mismatch" in error for error in report["data_quality"]["errors"])


def test_report_is_deterministic_with_unordered_relative_input():
    candidate = make_candidate_records()
    initialized = candidate[:1]
    trailing = candidate[-139:]
    relative = candidate[1:-139]

    first = analyzer.analyze_records(
        initialized + relative + trailing, make_baseline_records(),
    )
    second = analyzer.analyze_records(
        initialized + list(reversed(relative)) + trailing,
        make_baseline_records(),
    )

    assert first == second
    assert list(first["metrics"]["etf_counts"]) == [
        "159915.XSHE", "510300.XSHG", "518880.XSHG",
    ]


def test_cli_writes_only_explicit_output_file_and_keeps_inputs_unchanged(tmp_path):
    candidate_path = tmp_path / "candidate.log"
    baseline_path = tmp_path / "baseline.log"
    output_path = tmp_path / "report" / "relative.json"
    candidate_lines = [
        "2021-01-05 09:35:00 - INFO - " + json.dumps(record, sort_keys=True)
        for record in make_candidate_records()
    ]
    baseline_lines = [
        "2021-01-05 09:35:00 - INFO - " + json.dumps(record, sort_keys=True)
        for record in make_baseline_records()
    ]
    candidate_path.write_text("\n".join(candidate_lines), encoding="utf-8")
    baseline_path.write_text("\n".join(baseline_lines), encoding="utf-8")
    candidate_before = candidate_path.read_bytes()
    baseline_before = baseline_path.read_bytes()

    status = analyzer.main([
        "--candidate-log", str(candidate_path),
        "--baseline-log", str(baseline_path),
        "--output", str(output_path),
    ])

    assert status == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["continue_candidate"] is True
    assert candidate_path.read_bytes() == candidate_before
    assert baseline_path.read_bytes() == baseline_before


def test_foreign_relative_contract_is_an_error_before_build_filtering():
    foreign = make_relative_record(99)
    foreign["build"] = "20260827.5"
    foreign["relative_observation_fingerprint"] = "foreign-contract"
    foreign["signal_date"] = "2022-01-04"
    foreign["expires_date"] = "2022-01-04"

    report = analyzer.analyze_records(
        make_candidate_records() + [foreign], make_baseline_records(),
    )

    errors = report["data_quality"]["errors"]
    assert any("outside 2019-2021: 2022-01-04" in error for error in errors)
    assert any("build mismatch" in error for error in errors)
    assert any("fingerprint mismatch" in error for error in errors)
    assert report["continue_candidate"] is False


def test_relative_outcomes_require_exact_registration_identity_and_horizons():
    registration = make_relative_record(0)
    orphan = make_outcome(make_relative_record(1), 1, 0.01)
    mismatch = make_outcome(registration, 3, 0.01)
    mismatch["code"] = "518880.XSHG"
    mismatch["supporter_event_dates"] = {"BOLL": "2021-01-03"}
    invalid_horizon = make_outcome(registration, 1, 0.01)
    invalid_horizon["horizon"] = 1.9
    boolean_horizon = make_outcome(registration, 1, 0.01)
    boolean_horizon["horizon"] = True
    duplicate_conflict = make_outcome(registration, 5, 0.01)
    duplicate_conflict["outcome"]["direction_adjusted_return"] = 0.02
    candidate = [make_initialized_record(), registration, orphan, mismatch,
                 invalid_horizon, boolean_horizon, duplicate_conflict,
                 dict(duplicate_conflict)]

    report = analyzer.analyze_records(candidate, [])

    errors = report["data_quality"]["errors"]
    assert any("orphan relative outcome" in error for error in errors)
    assert any("relative outcome code mismatch" in error for error in errors)
    assert any("relative outcome supporter_event_dates mismatch" in error for error in errors)
    assert any("invalid relative horizon: 1.9" in error for error in errors)
    assert any("invalid relative horizon: True" in error for error in errors)
    assert any("duplicate relative outcome" in error for error in errors)


def test_baseline_requires_frozen_initialization_and_linked_formal_records():
    baseline = make_baseline_records()
    baseline[0]["build"] = "20260827.4"
    formal = next(record for record in baseline if record.get("event") == "resonance_decision")
    formal["direction"] = "NEUTRAL"
    baseline.append(dict(formal, direction="BUY_TURN"))
    mismatched = next(
        record for record in baseline
        if record.get("resonance_id") == "FORMAL:01"
        and record.get("event") == "observation_outcome"
    )
    mismatched["event_date"] = "2021-01-06"
    mismatched["direction"] = "SELL_TURN"
    orphan = {
        "event": "observation_outcome", "resonance_id": "FORMAL:orphan",
        "code": "510300.XSHG", "event_date": "2022-01-04", "horizon": 5,
        "outcome": {"status": "RECORDED", "return": 0.01},
    }

    report = analyzer.analyze_records(make_candidate_records(), baseline + [orphan])

    errors = report["data_quality"]["errors"]
    assert any("baseline initialization build mismatch" in error for error in errors)
    assert any("invalid formal direction" in error for error in errors)
    assert any("duplicate formal registration" in error for error in errors)
    assert any("formal outcome event_date mismatch" in error for error in errors)
    assert any("formal outcome direction mismatch" in error for error in errors)
    assert any("orphan formal outcome" in error for error in errors)
    assert any("outside 2019-2021: 2022-01-04" in error for error in errors)


def test_parser_rejects_nonfinite_json_constants_and_records_reject_bad_fields():
    assert analyzer.parse_joinquant_log_line(
        '2021-01-05 - {"event":"x","value":NaN}', 1,
    )["_parse_error"] == "non-finite JSON constant: NaN"
    broken = make_relative_record(0)
    broken["direction"] = "NEUTRAL"
    broken["event_close"] = float("inf")
    broken.pop("supporters")
    invalid_order = {
        "event": "order_transition", "outcome": "FILLED", "side": "BUY",
        "code": "510300.XSHG", "before_amount": -1, "after_amount": float("nan"),
    }

    report = analyzer.analyze_records(
        [make_initialized_record(), broken, invalid_order], [],
    )

    errors = report["data_quality"]["errors"]
    assert any("invalid candidate direction" in error for error in errors)
    assert any("event_close" in error for error in errors)
    assert any("supporters" in error for error in errors)
    assert any("invalid filled order" in error for error in errors)


def test_cli_rejects_output_alias_of_an_input_without_overwriting(tmp_path, capsys):
    candidate_path = tmp_path / "candidate.log"
    baseline_path = tmp_path / "baseline.log"
    candidate_path.write_text("noise", encoding="utf-8")
    baseline_path.write_text("noise", encoding="utf-8")
    before = candidate_path.read_bytes()
    alias_path = tmp_path / "candidate-alias.log"
    os.link(candidate_path, alias_path)

    status = analyzer.main([
        "--candidate-log", str(candidate_path),
        "--baseline-log", str(baseline_path),
        "--output", str(alias_path),
    ])

    assert status == 2
    assert capsys.readouterr().err == "input error: output path must not match an input log\n"
    assert candidate_path.read_bytes() == before
    assert alias_path.read_bytes() == before


def test_grouped_return_schema_is_complete_and_deterministic():
    report = analyzer.analyze_records(
        make_candidate_records(), make_baseline_records(),
    )

    assert set(report["metrics"]["by_branch"]) == {
        "HARD_BOLL_SOFT_OSC", "SOFT_ALL_THREE",
    }
    assert set(report["metrics"]["by_direction"]) == {
        "BUY_TURN", "SELL_TURN",
    }
    for group in (
            *report["metrics"]["by_branch"].values(),
            *report["metrics"]["by_direction"].values()):
        assert set(group) == {"horizon_1", "horizon_3", "horizon_5"}
        assert group["horizon_5"]["count"] >= 0


def test_relative_contract_rejects_impossible_branch_fingerprints_and_outcome_identity():
    registration = make_relative_record(0, branch="HARD_BOLL_SOFT_OSC")
    registration["supporters"] = ["BOLL", "KDJ", "RSI"]
    registration["hard_or_relative_source_by_indicator"] = {
        "BOLL": "HARD", "KDJ": "HARD", "RSI": "RELATIVE",
    }
    registration["supporter_event_dates"] = {
        indicator: registration["signal_date"] for indicator in registration["supporters"]
    }
    registration["parameter_fingerprint"] = "wrong"
    outcome = make_outcome(registration, 1, 0.01)
    outcome["resonance_id"] = "RELATIVE:other"
    outcome["outcome"]["direction_adjusted_return"] = 0.02

    report = analyzer.analyze_records([make_initialized_record(), registration, outcome], [])

    errors = report["data_quality"]["errors"]
    assert any("impossible candidate branch/supporters/source contract" in error for error in errors)
    assert any("registration parameter_fingerprint mismatch" in error for error in errors)
    assert any("relative outcome resonance_id mismatch" in error for error in errors)
    assert any("direction_adjusted_return mismatch" in error for error in errors)


def test_relative_sell_outcome_requires_negated_direction_adjusted_return():
    registration = make_relative_record(0, direction="SELL_TURN")
    outcome = make_outcome(registration, 1, 0.01)
    outcome["outcome"]["direction_adjusted_return"] = 0.01

    report = analyzer.analyze_records([make_initialized_record(), registration, outcome], [])

    assert any("direction_adjusted_return mismatch" in error
               for error in report["data_quality"]["errors"])


def test_baseline_requires_every_formal_registration_to_have_horizon_five_and_merges_identical_diagnostics():
    baseline = make_baseline_records()
    baseline[:] = [record for record in baseline if not (
        record.get("event") == "observation_outcome"
        and record.get("resonance_id") == "FORMAL:00"
    )]
    formal = next(record for record in baseline if record.get("resonance_id") == "FORMAL:01")
    baseline.append(dict(formal))

    report = analyzer.analyze_records(make_candidate_records(), baseline)

    errors = report["data_quality"]["errors"]
    assert any("missing formal horizon 5: FORMAL:00" in error for error in errors)
    assert not any("duplicate formal registration: FORMAL:01" in error for error in errors)


def test_strict_record_types_dates_and_overflow_become_quality_errors_not_exceptions():
    candidate = make_candidate_records()
    candidate[1]["signal_date"] += " trailing"
    order = next(record for record in candidate if record.get("event") == "order_transition")
    order["_log_timestamp"] = None
    order["before_amount"] = "0"
    for record in candidate:
        if record.get("event") == "observation_outcome" and record.get("horizon") == 5:
            record["outcome"]["return"] = 1e308
            record["outcome"]["direction_adjusted_return"] = 1e308

    report = analyzer.analyze_records(candidate, make_baseline_records())

    errors = report["data_quality"]["errors"]
    assert any("invalid signal date" in error for error in errors)
    assert any("invalid filled order log timestamp" in error for error in errors)
    assert any("invalid filled order before_amount" in error for error in errors)
    assert any("non-finite aggregate" in error for error in errors)
    assert report["continue_candidate"] is False


def _write_log(path, records):
    lines = []
    for index, record in enumerate(records):
        timestamp = record.get("_log_timestamp", "2021-01-05T09:35:00").replace("T", " ")
        payload = {key: value for key, value in record.items() if not key.startswith("_")}
        lines.append(timestamp + " - INFO - " + json.dumps(payload, sort_keys=True))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_cli_atomic_write_keeps_existing_output_when_replace_fails(tmp_path, capsys):
    candidate_path = tmp_path / "candidate.log"
    baseline_path = tmp_path / "baseline.log"
    output_path = tmp_path / "report.json"
    _write_log(candidate_path, make_candidate_records())
    _write_log(baseline_path, make_baseline_records())
    output_path.write_bytes(b"old output")

    with mock.patch.object(analyzer.os, "replace", side_effect=OSError("disk fault")):
        status = analyzer.main([
            "--candidate-log", str(candidate_path),
            "--baseline-log", str(baseline_path), "--output", str(output_path),
        ])

    assert status == 2
    assert capsys.readouterr().err == "output error: disk fault\n"
    assert output_path.read_bytes() == b"old output"
    assert not list(tmp_path.glob(".report.json.*"))


def test_cli_normalizes_multi_file_order_and_uses_frozen_closing_summary(tmp_path):
    candidate = make_candidate_records()
    candidate.append({"event": "portfolio_summary", "closing_date": "2021-01-01",
                      "total_value": 1.0, "_log_timestamp": "2021-01-01T15:30:00"})
    early = tmp_path / "candidate-early.log"
    late = tmp_path / "candidate-late.log"
    baseline = tmp_path / "baseline.log"
    _write_log(early, candidate[::2])
    _write_log(late, candidate[1::2])
    _write_log(baseline, make_baseline_records())
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    analyzer.main(["--candidate-log", str(early), "--candidate-log", str(late),
                   "--baseline-log", str(baseline), "--output", str(first_output)])
    analyzer.main(["--candidate-log", str(late), "--candidate-log", str(early),
                   "--baseline-log", str(baseline), "--output", str(second_output)])

    first = json.loads(first_output.read_text(encoding="utf-8"))
    second = json.loads(second_output.read_text(encoding="utf-8"))
    assert first == second
    assert first["metrics"]["final_asset"] == pytest.approx(23856.40)


def _builder_three_supporter_registration():
    signal_date = "2021-01-08"
    hard = strategy.empty_event_book()
    hard["active"]["BOLL"] = strategy.make_turn_event(
        "BOLL", strategy.TurnDirection.BUY_TURN, signal_date, signal_date, {"fixture": "hard"},
    )
    relative = strategy.empty_event_book()
    for indicator in ("RSI", "KDJ"):
        relative["active"][indicator] = strategy.make_relative_turn_event(
            indicator, strategy.TurnDirection.BUY_TURN, signal_date, signal_date,
            {"fixture": indicator},
        )
    observation = strategy.build_relative_resonance_observation(
        "510300.XSHG", strategy.TurnDirection.BUY_TURN,
        hard, relative, signal_date, 10.0,
    )
    assert observation is not None
    registration = make_relative_record(0)
    registration.update({
        "relative_observation_id": observation["relative_observation_id"],
        "observation_kind": observation["observation_kind"],
        "branch": observation["branch"], "code": observation["code"],
        "direction": observation["direction"].value,
        "signal_date": observation["signal_date"].isoformat(),
        "expires_date": observation["expires_date"].isoformat(),
        "supporters": list(observation["supporters"]),
        "supporter_event_dates": {
            key: value.isoformat() for key, value in observation["supporter_event_dates"].items()
        },
        "hard_or_relative_source_by_indicator": observation["hard_or_relative_source_by_indicator"],
        "event_close": observation["event_close"],
    })
    return registration


def test_real_builder_hard_boll_branch_with_both_relative_oscillators_is_accepted():
    registration = _builder_three_supporter_registration()
    outcome = make_outcome(registration, 5, 0.02)

    report = analyzer.analyze_records([make_initialized_record(), registration, outcome], [])

    assert registration["supporters"] == ["BOLL", "KDJ", "RSI"]
    assert not any("impossible candidate branch" in error
                   for error in report["data_quality"]["errors"])


def test_outcome_return_must_be_recomputed_from_closing_price_and_event_close():
    registration = make_relative_record(0)
    outcome = make_outcome(registration, 5, 0.02)
    outcome["outcome"]["closing_price"] = 30.0

    report = analyzer.analyze_records([make_initialized_record(), registration, outcome], [])

    assert any("relative outcome return mismatch" in error
               for error in report["data_quality"]["errors"])


def test_baseline_unavailable_horizon_five_is_comparison_incomplete():
    baseline = make_baseline_records()
    outcome = next(record for record in baseline if record.get("resonance_id") == "FORMAL:00"
                   and record.get("event") == "observation_outcome")
    outcome["outcome"] = {"status": "PRICE_UNAVAILABLE", "closing_date": "2021-01-05"}

    report = analyzer.analyze_records(make_candidate_records(), baseline)

    assert any("formal comparison incomplete: FORMAL:00" in error
               for error in report["data_quality"]["errors"])
    assert report["data_quality"]["formal_missing_outcome_count"] == 1


def test_frozen_summary_and_filled_quantity_contracts_reject_conflicts_and_unsafe_types():
    candidate = make_candidate_records()
    candidate.extend((
        {"event": "portfolio_summary", "closing_date": "2021-12-31", "total_value": 1.0},
        {"event": "portfolio_summary", "closing_date": "2021-12-31", "total_value": "23856.40"},
        {"event": "portfolio_summary", "closing_date": "2021-12-31", "total_value": True},
        {"event": "portfolio_summary", "closing_date": "2021-12-31", "total_value": float("inf")},
    ))
    order = next(record for record in candidate if record.get("event") == "order_transition")
    order["code"] = "   "
    order["before_amount"] = 100.0
    order["after_amount"] = 10 ** 400

    report = analyzer.analyze_records(candidate, make_baseline_records())

    errors = report["data_quality"]["errors"]
    assert any("conflicting candidate frozen portfolio summary" in error for error in errors)
    assert any("invalid candidate frozen portfolio summary total_value" in error for error in errors)
    assert any("invalid filled order identity" in error for error in errors)
    assert any("invalid filled order before_amount" in error for error in errors)
    assert report["metrics"]["final_asset"] is None


def test_nonfinite_parse_datetime_sort_and_positive_contribution_overflow_are_quality_errors():
    parsed = analyzer.parse_joinquant_log_line(
        '2021-01-05 09:35:00 - {"event":"x","value":NaN}', 1,
    )
    assert parsed["_parse_error"] == "non-finite JSON constant: NaN"
    assert analyzer.parse_joinquant_log_line(
        '2021-01-05 09:35:00 - {"event":"x","value":Infinity}', 2,
    )["_parse_error"] == "non-finite JSON constant: Infinity"
    assert analyzer.parse_joinquant_log_line(
        '2021-01-05 09:35:00 - {"event":"x","value":-Infinity}', 3,
    )["_parse_error"] == "non-finite JSON constant: -Infinity"

    candidate = make_candidate_records()
    candidate[1]["signal_date"] = datetime(2021, 1, 5, 9, 35)
    for record in candidate:
        if record.get("event") == "observation_outcome" and record.get("horizon") == 5:
            record["outcome"].update({
                "closing_price": 1e308,
                "return": 1e308,
                "direction_adjusted_return": 1e308,
            })

    report = analyzer.analyze_records(candidate + [parsed], make_baseline_records())

    assert any("parse error" in error for error in report["data_quality"]["errors"])
    assert any("non-finite aggregate: positive contribution" in error
               for error in report["data_quality"]["errors"])
    assert report["metrics"]["max_positive_contribution_by_etf"] is None
    assert json.dumps(report, allow_nan=False)


def test_same_second_baseline_filled_records_preserve_line_order_and_detect_inverse_path():
    candidate = make_candidate_records()
    baseline = make_baseline_records()
    baseline_orders = [record for record in baseline if record.get("event") == "order_transition"]
    baseline[:] = ([record for record in baseline if record.get("event") != "order_transition"]
                   + list(reversed(baseline_orders)))

    report = analyzer.analyze_records(candidate, baseline)

    assert report["gates"]["formal_order_path_exact"] is False


def test_loader_uses_canonical_file_order_and_flags_same_second_cross_file_ambiguity(tmp_path):
    first_path = tmp_path / "a.log"
    second_path = tmp_path / "z.log"
    first_path.write_text('2021-01-05 09:35:00 - {"event":"order_transition","outcome":"FILLED"}', encoding="utf-8")
    second_path.write_text('2021-01-05 09:35:00 - {"event":"order_transition","outcome":"FILLED"}', encoding="utf-8")

    loaded = analyzer.load_log_records([second_path, first_path])
    report = analyzer.analyze_records(loaded, [])

    assert [pathlib.Path(record["_source_path"]).name for record in loaded] == ["a.log", "z.log"]
    assert any("ambiguous filled order timestamp across files" in error
               for error in report["data_quality"]["errors"])
    assert report["gates"]["formal_order_path_exact"] is False


def test_relative_namespace_without_relative_markers_is_a_quality_error():
    lost = {
        "event": "observation_outcome", "resonance_id": "RELATIVE:lost",
        "code": "510300.XSHG", "event_date": "2021-01-05", "horizon": 5,
        "outcome": {"status": "RECORDED", "return": 0.01},
    }

    report = analyzer.analyze_records([make_initialized_record(), lost], [])

    assert any("relative namespace record missing relative markers" in error
               for error in report["data_quality"]["errors"])


def test_support_source_json_container_is_a_quality_error_not_an_exception():
    candidate = make_relative_record(0)
    candidate["hard_or_relative_source_by_indicator"]["BOLL"] = ["HARD"]

    report = analyzer.analyze_records([make_initialized_record(), candidate], [])

    assert any("invalid candidate hard_or_relative_source_by_indicator" in error
               for error in report["data_quality"]["errors"])


def test_cli_io_failures_return_stable_nonzero_without_traceback(tmp_path, capsys):
    missing = tmp_path / "missing.log"
    output = tmp_path / "report.json"

    status = analyzer.main([
        "--candidate-log", str(missing), "--baseline-log", str(missing), "--output", str(output),
    ])

    captured = capsys.readouterr()
    assert status == 2
    assert captured.err == "input error: input log must be a file: %s\n" % missing.resolve()
    assert "Traceback" not in captured.err
    assert not output.exists()
