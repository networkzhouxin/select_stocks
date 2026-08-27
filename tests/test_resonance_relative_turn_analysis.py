import importlib.util
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import types
from datetime import date, datetime, timedelta
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
SESSION_DATES = {
    2019: ("2019-01-02", "2019-01-03", "2019-01-04", "2019-01-07", "2019-01-08",
           "2019-01-09", "2019-01-10", "2019-01-11", "2019-01-14", "2019-01-15",
           "2019-01-16", "2019-01-17", "2019-01-18", "2019-01-21", "2019-01-22",
           "2019-01-23", "2019-01-24"),
    2020: ("2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08",
           "2020-01-09", "2020-01-10", "2020-01-13", "2020-01-14", "2020-01-15",
           "2020-01-16", "2020-01-17", "2020-01-20", "2020-01-21", "2020-01-22",
           "2020-01-23", "2020-02-03"),
    2021: ("2020-12-31", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07",
           "2021-01-08", "2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14",
           "2021-01-15", "2021-01-18", "2021-01-19", "2021-01-20", "2021-01-21",
           "2021-01-22", "2021-01-25"),
}


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


def _relative_observation_id(code, direction, branch, supporters, source_map, date_map):
    parts = ["RELATIVE", branch, direction, code]
    for indicator in sorted(supporters):
        parts.append("%s:%s:%s" % (
            indicator, source_map[indicator], date_map[indicator],
        ))
    return "RELATIVE:" + hashlib.sha256(
        "|".join(parts).encode("utf-8")
    ).hexdigest()[:20]


def make_relative_record(index, direction="BUY_TURN", branch=None, code=None):
    year = 2019 + index // 10
    branch = branch or (
        "HARD_BOLL_SOFT_OSC" if index % 2 else "SOFT_ALL_THREE"
    )
    code = code or ("510300.XSHG", "159915.XSHE", "518880.XSHG")[index % 3]
    signal_date = (SESSION_DATES[year][index % 10 + 2]
                   if year in SESSION_DATES else "2028-01-02")
    if branch == "HARD_BOLL_SOFT_OSC":
        supporters = ["BOLL", "RSI"]
        source_map = {"BOLL": "HARD", "RSI": "RELATIVE"}
    else:
        supporters = ["BOLL", "KDJ", "RSI"]
        source_map = {indicator: "RELATIVE" for indicator in supporters}
    date_map = {indicator: signal_date for indicator in supporters}
    session_dates = SESSION_DATES.get(year)
    if session_dates is not None:
        registration_date = session_dates[session_dates.index(signal_date) + 1]
    else:
        registration_date = (date.fromisoformat(signal_date) + timedelta(days=1)).isoformat()
    registration_timestamp = registration_date + "T09:35:00"
    return {
        "event": "relative_resonance_observation",
        "relative_observation_id": _relative_observation_id(
            code, direction, branch, supporters, source_map, date_map,
        ),
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": branch,
        "code": code,
        "direction": direction,
        "signal_date": signal_date,
        "expires_date": signal_date,
        "supporters": supporters,
        "supporter_event_dates": date_map,
        "hard_or_relative_source_by_indicator": source_map,
        "build": BUILD,
        "relative_observation_fingerprint": FINGERPRINT,
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "event_close": 10.0,
        "_log_timestamp": registration_timestamp,
    }


def make_outcome(record, horizon, value):
    adjusted = value if record["direction"] == "BUY_TURN" else -value
    session_dates = next((dates for dates in SESSION_DATES.values()
                          if record["signal_date"] in dates), None)
    if session_dates is None:
        registration_date = date.fromisoformat(record["_log_timestamp"][:10])
        closing_date = (registration_date + timedelta(days=horizon)).isoformat()
    else:
        closing_date = session_dates[
            session_dates.index(record["signal_date"]) + horizon
        ]
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
            "status": "RECORDED", "closing_date": closing_date,
            "closing_price": record["event_close"] * (1.0 + value),
            "return": value, "direction_adjusted_return": adjusted,
        },
        "_log_timestamp": closing_date + "T15:30:00",
    }


def make_candidate_records():
    records = [make_initialized_record()]
    for year in sorted(SESSION_DATES):
        session_dates = SESSION_DATES[year]
        for index, session_date in enumerate(session_dates[1:], 1):
            records.append({
                "event": "signal_snapshot", "build": BUILD,
                "parameter_fingerprint": "e1227fbd8b4a884e",
                "pool_fingerprint": "9123995edeb1ed84",
                "event_logic_fingerprint": "1c0b8a22f48c97c3",
                "relative_observation_fingerprint": FINGERPRINT,
                "code": "510300.XSHG", "decision_date": session_date,
                "signal_date": session_dates[index - 1], "valid": True,
                "_log_timestamp": session_date + "T09:35:00",
            })
    for index in range(30):
        record = make_relative_record(index)
        records.append(record)
        records.extend(make_outcome(record, horizon, value) for horizon, value in (
            (1, 0.005), (3, 0.01), (5, 0.02),
        ))
    records.extend(make_session_summaries())
    records.extend(make_order_path())
    records.append({
        "event": "portfolio_summary", "closing_date": "2021-12-31",
        "total_value": 23856.40, "_log_timestamp": "2021-12-31T15:30:00",
    })
    return records


def make_session_summaries():
    return [{
        "event": "portfolio_summary", "closing_date": session_date,
        "total_value": 20000.0, "_log_timestamp": session_date + "T15:30:00",
    } for session_date in sorted({
        session_date for dates in SESSION_DATES.values() for session_date in dates
    })]


def _replace_relative_identity(registration, outcomes):
    observation_id = _relative_observation_id(
        registration["code"], registration["direction"], registration["branch"],
        registration["supporters"], registration["hard_or_relative_source_by_indicator"],
        registration["supporter_event_dates"],
    )
    registration["relative_observation_id"] = observation_id
    for outcome in outcomes:
        outcome["relative_observation_id"] = observation_id
        outcome["resonance_id"] = observation_id


def make_baseline_records():
    records = [{
        "event": "strategy_initialized", "build": "20260827.3",
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
    }]
    records.extend(make_session_summaries())
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
            "_log_timestamp": "2021-01-06T09:35:00",
        })
        records.append({
            "event": "observation_outcome", "resonance_id": resonance_id,
            "code": "510300.XSHG", "event_date": "2021-01-05",
            "horizon": 5,
            "outcome": {
                "status": "RECORDED", "closing_date": "2021-01-12",
                "return": 0.01,
            },
            "_log_timestamp": "2021-01-12T15:30:00",
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


def test_parser_keeps_truncated_known_structured_event_as_quality_sentinel():
    parsed = analyzer.parse_joinquant_log_line(
        '2021-01-05 09:35:00 - INFO - {"event":"signal_snapshot",', 9,
    )

    assert parsed is not None
    assert parsed["_parse_error"].startswith("invalid structured JSON:")
    assert parsed["_ordinal"] == 9


def test_parser_matches_known_event_field_not_diagnostic_substrings():
    truncated = '2021-01-05 09:35:00 - INFO - { "event" : "signal_snapshot",'
    escaped = truncated.replace('"', "&quot;")

    assert analyzer.parse_joinquant_log_line(
        '2021-01-05 09:35:00 - INFO - diagnostic mentions signal_snapshot {not-json', 10,
    ) is None
    assert analyzer.parse_joinquant_log_line(truncated, 11)["_parse_error"].startswith(
        "invalid structured JSON:",
    )
    assert analyzer.parse_joinquant_log_line(escaped, 12)["_parse_error"].startswith(
        "invalid structured JSON:",
    )


def test_cli_reports_truncated_known_structured_line_without_mutating_input(tmp_path):
    candidate_path = tmp_path / "candidate.log"
    baseline_path = tmp_path / "baseline.log"
    output_path = tmp_path / "report.json"
    _write_log(candidate_path, make_candidate_records())
    with candidate_path.open("a", encoding="utf-8") as stream:
        stream.write('\n2021-01-05 09:35:00 - INFO - {"event":"signal_snapshot",')
    original = candidate_path.read_bytes()
    _write_log(baseline_path, make_baseline_records())

    completed = subprocess.run([
        sys.executable, str(ANALYZER_PATH), "--candidate-log", str(candidate_path),
        "--baseline-log", str(baseline_path), "--output", str(output_path),
    ], capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr
    assert candidate_path.read_bytes() == original
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["continue_candidate"] is False
    assert any(str(candidate_path.resolve()) in error and ":" in error
               and "invalid structured JSON" in error
               for error in report["data_quality"]["errors"])


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
        "baseline session calendar unavailable: invalid initialization",
        "candidate session calendar unavailable: invalid initialization",
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


def test_scope_summaries_are_complete_and_deterministically_ordered():
    report = analyzer.analyze_records(make_candidate_records(), make_baseline_records())
    scopes = report["metrics"]["scope_summaries"]

    assert list(scopes) == [
        "formal_resonance", "relative_total",
        "HARD_BOLL_SOFT_OSC", "SOFT_ALL_THREE",
    ]
    for summary in scopes.values():
        assert list(summary) == [
            "candidate_count", "direction_counts", "year_counts", "etf_counts",
            "horizon_1", "horizon_3", "horizon_5",
        ]
        assert list(summary["direction_counts"]) == ["BUY_TURN", "SELL_TURN"]
        assert list(summary["year_counts"]) == ["2019", "2020", "2021"]
        assert list(summary["etf_counts"]) == sorted(summary["etf_counts"])
    assert scopes["formal_resonance"]["candidate_count"] == 30
    assert scopes["formal_resonance"]["horizon_1"]["count"] == 0
    assert scopes["formal_resonance"]["horizon_3"]["count"] == 0
    assert scopes["formal_resonance"]["horizon_5"]["count"] == 30
    assert scopes["relative_total"]["candidate_count"] == 30
    assert scopes["relative_total"]["horizon_1"]["count"] == 30
    assert scopes["HARD_BOLL_SOFT_OSC"]["candidate_count"] == 15
    assert scopes["SOFT_ALL_THREE"]["candidate_count"] == 15


def test_business_records_require_real_emitter_timestamp_chronology():
    candidate = make_candidate_records()
    baseline = make_baseline_records()
    for record in candidate + baseline:
        if record.get("event") in {
                "relative_resonance_observation", "observation_outcome",
                "resonance_decision"}:
            record["_log_timestamp"] = "2019-01-01T00:00:00"

    report = analyzer.analyze_records(candidate, baseline)

    assert report["continue_candidate"] is False
    assert any("log timestamp" in error for error in report["data_quality"]["errors"])


def test_relative_and_formal_registration_timestamps_must_follow_signal_date():
    candidate = make_candidate_records()
    baseline = make_baseline_records()
    relative = next(record for record in candidate
                    if record.get("event") == "relative_resonance_observation")
    formal = next(record for record in baseline
                  if record.get("event") == "resonance_decision")
    relative["_log_timestamp"] = relative["signal_date"] + "T09:35:00"
    formal["_log_timestamp"] = formal["signal_date"] + "T09:35:00"

    report = analyzer.analyze_records(candidate, baseline)

    assert report["continue_candidate"] is False
    assert any("relative registration log timestamp must follow signal date" in error
               for error in report["data_quality"]["errors"])
    assert any("formal registration log timestamp must follow signal date" in error
               for error in report["data_quality"]["errors"])


def test_friday_signal_to_monday_registration_timestamp_is_legal():
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    outcomes = [record for record in candidate
                if record.get("event") == "observation_outcome"
                and record.get("relative_observation_id") == registration["relative_observation_id"]]
    registration["signal_date"] = registration["expires_date"] = "2019-01-04"
    registration["supporter_event_dates"] = {
        indicator: "2019-01-04" for indicator in registration["supporters"]
    }
    registration["_log_timestamp"] = "2019-01-07T09:35:00"
    _replace_relative_identity(registration, outcomes)
    for outcome, closing_date in zip(sorted(outcomes, key=lambda item: item["horizon"]),
                                     ("2019-01-07", "2019-01-09", "2019-01-11")):
        outcome["event_date"] = "2019-01-04"
        outcome["supporter_event_dates"] = dict(registration["supporter_event_dates"])
        outcome["outcome"]["closing_date"] = closing_date
        outcome["_log_timestamp"] = closing_date + "T15:30:00"

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["continue_candidate"] is True


def test_relative_support_window_requires_proven_previous_trading_session():
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    outcomes = [record for record in candidate
                if record.get("event") == "observation_outcome"
                and record.get("relative_observation_id") == registration["relative_observation_id"]]
    registration["signal_date"] = registration["expires_date"] = "2019-01-02"
    registration["supporter_event_dates"] = {
        "BOLL": "2019-01-01", "KDJ": "2019-01-02", "RSI": "2019-01-02",
    }
    registration["_log_timestamp"] = "2019-01-03T09:35:00"
    _replace_relative_identity(registration, outcomes)
    for outcome, closing_date in zip(sorted(outcomes, key=lambda item: item["horizon"]),
                                     ("2019-01-03", "2019-01-07", "2019-01-09")):
        outcome["event_date"] = "2019-01-02"
        outcome["supporter_event_dates"] = dict(registration["supporter_event_dates"])
        outcome["outcome"]["closing_date"] = closing_date
        outcome["_log_timestamp"] = closing_date + "T15:30:00"

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["continue_candidate"] is False
    assert any("candidate supporter window unverifiable" in error
               for error in report["data_quality"]["errors"])


def test_weekend_previous_trading_session_is_legal_with_snapshot_evidence():
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    outcomes = [record for record in candidate
                if record.get("event") == "observation_outcome"
                and record.get("relative_observation_id") == registration["relative_observation_id"]]
    registration["signal_date"] = registration["expires_date"] = "2021-01-11"
    registration["supporter_event_dates"] = {
        "BOLL": "2021-01-08", "KDJ": "2021-01-11", "RSI": "2021-01-11",
    }
    registration["_log_timestamp"] = "2021-01-12T09:35:00"
    _replace_relative_identity(registration, outcomes)
    for outcome, closing_date in zip(sorted(outcomes, key=lambda item: item["horizon"]),
                                     ("2021-01-12", "2021-01-14", "2021-01-18")):
        outcome["event_date"] = "2021-01-11"
        outcome["supporter_event_dates"] = dict(registration["supporter_event_dates"])
        outcome["outcome"]["closing_date"] = closing_date
        outcome["_log_timestamp"] = closing_date + "T15:30:00"

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["continue_candidate"] is True


def _session_snapshot(decision_date, signal_date, timestamp=None):
    return {
        "event": "signal_snapshot", "build": BUILD,
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "relative_observation_fingerprint": FINGERPRINT,
        "code": "510300.XSHG", "decision_date": decision_date,
        "signal_date": signal_date, "valid": True,
        "_log_timestamp": timestamp or decision_date + "T09:35:00",
    }


def test_relative_support_window_uses_explicit_snapshot_decision_to_signal_mapping():
    registration = make_relative_record(0)
    registration["signal_date"] = registration["expires_date"] = "2019-01-08"
    registration["supporter_event_dates"] = {
        "BOLL": "2019-01-04", "KDJ": "2019-01-08", "RSI": "2019-01-08",
    }
    registration["_log_timestamp"] = "2019-01-11T09:35:00"
    _replace_relative_identity(registration, [])
    candidate = [
        make_initialized_record(),
        _session_snapshot("2019-01-04", "2019-01-03"),
        _session_snapshot("2019-01-11", "2019-01-08"),
        registration,
    ]

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert any("candidate supporter window unverifiable" in error
               for error in report["data_quality"]["errors"])

    registration["supporter_event_dates"] = {
        "BOLL": "2019-01-07", "KDJ": "2019-01-08", "RSI": "2019-01-08",
    }
    _replace_relative_identity(registration, [])
    candidate.append(_session_snapshot("2019-01-08", "2019-01-07"))

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert not any("candidate supporter window" in error
                   or "invalid candidate supporter trading-session window" in error
                   for error in report["data_quality"]["errors"])
    assert not any("relative registration session evidence mismatch" in error
                   for error in report["data_quality"]["errors"])


def test_candidate_session_snapshot_requires_frozen_identity_time_and_consistency():
    candidate = make_candidate_records()
    malformed = next(record for record in candidate
                     if record.get("event") == "signal_snapshot")
    malformed.pop("relative_observation_fingerprint")
    malformed["_log_timestamp"] = malformed["decision_date"] + "T15:30:00"
    conflicting = _session_snapshot("2019-01-08", "2019-01-04")
    candidate.append(conflicting)

    report = analyzer.analyze_records(candidate, make_baseline_records())

    errors = report["data_quality"]["errors"]
    assert any("candidate signal_snapshot relative_observation_fingerprint mismatch" in error
               for error in errors)
    assert any("candidate signal_snapshot log timestamp must equal 09:35" in error
               for error in errors)
    assert any("conflicting candidate signal_snapshot session evidence" in error
               for error in errors)


def test_all_signal_date_supporters_do_not_require_predecessor_snapshot():
    registration = make_relative_record(0)
    registration["signal_date"] = registration["expires_date"] = "2019-01-08"
    registration["supporter_event_dates"] = {
        indicator: "2019-01-08" for indicator in registration["supporters"]
    }
    registration["_log_timestamp"] = "2019-01-09T09:35:00"
    _replace_relative_identity(registration, [])
    candidate = [
        make_initialized_record(),
        _session_snapshot("2019-01-09", "2019-01-08"),
        registration,
    ]

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert not any("candidate supporter window" in error
                   for error in report["data_quality"]["errors"])

    registration["supporter_event_dates"][registration["supporters"][0]] = "2019-01-07"
    _replace_relative_identity(registration, [])
    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert any("candidate supporter window unverifiable" in error
               for error in report["data_quality"]["errors"])

    candidate.append(_session_snapshot("2019-01-08", "2019-01-07"))
    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert not any("candidate supporter window" in error
                   or "invalid candidate supporter trading-session window" in error
                   for error in report["data_quality"]["errors"])


def _candidate_overlapping_baseline_formal():
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation"
                        and record["signal_date"] == "2021-01-05")
    outcomes = [record for record in candidate
                if record.get("event") == "observation_outcome"
                and record.get("relative_observation_id") == registration["relative_observation_id"]]
    registration["code"] = "510300.XSHG"
    registration["supporter_event_dates"] = {
        indicator: "2021-01-05" for indicator in registration["supporters"]
    }
    registration["_log_timestamp"] = "2021-01-06T09:35:00"
    _replace_relative_identity(registration, outcomes)
    for outcome in outcomes:
        outcome["code"] = registration["code"]
        outcome["supporter_event_dates"] = dict(registration["supporter_event_dates"])
        outcome["event_date"] = "2021-01-05"
    return candidate


def _single_formal_baseline():
    baseline = make_baseline_records()
    return [record for record in baseline if (
        record.get("event") not in {"resonance_decision", "observation_outcome"}
        or record.get("resonance_id") == "FORMAL:00"
    )]


def test_overlap_uses_baseline_formal_registration_when_candidate_has_none():
    candidate = _candidate_overlapping_baseline_formal()

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["data_quality"]["formal_overlap_count"] == 1
    assert report["gates"]["data_quality_complete"] is False
    assert report["continue_candidate"] is False


def test_overlap_excludes_incomplete_or_invalid_baseline_canonical_registrations():
    valid = analyzer.analyze_records(
        _candidate_overlapping_baseline_formal(), _single_formal_baseline(),
    )
    assert valid["data_quality"]["formal_overlap_count"] == 1

    bad_initialization = _single_formal_baseline()
    next(record for record in bad_initialization
         if record.get("event") == "strategy_initialized")["parameter_fingerprint"] = "bad"
    bad_timestamp = _single_formal_baseline()
    next(record for record in bad_timestamp
         if record.get("event") == "resonance_decision")["_log_timestamp"] = "2021-01-05T09:35:00"
    missing_horizon_five = [record for record in _single_formal_baseline() if not (
        record.get("event") == "observation_outcome" and record.get("horizon") == 5
    )]

    for baseline in (bad_initialization, bad_timestamp, missing_horizon_five):
        report = analyzer.analyze_records(_candidate_overlapping_baseline_formal(), baseline)
        assert report["data_quality"]["errors"]
        assert report["data_quality"]["formal_overlap_count"] == 0


def test_overlap_invalidates_id_when_any_registration_replica_is_invalid():
    baseline = _single_formal_baseline()
    registration = next(record for record in baseline
                        if record.get("event") == "resonance_decision")
    legal_duplicate = dict(registration)
    baseline.append(legal_duplicate)

    report = analyzer.analyze_records(_candidate_overlapping_baseline_formal(), baseline)

    assert report["data_quality"]["formal_overlap_count"] == 1

    invalid_duplicate = dict(registration)
    invalid_duplicate["_log_timestamp"] = "2021-01-06T00:00:00"
    baseline.append(invalid_duplicate)

    direct_errors = []
    for record in baseline:
        record["_record_namespace"] = analyzer._classify_record_namespace(record, direct_errors)
    _, _, _, canonical = analyzer._validate_baseline(baseline, direct_errors)

    assert any("formal registration log timestamp before 09:35" in error
               for error in direct_errors)
    assert canonical == {}

    invalid_duplicate["_log_timestamp"] = "2021-01-06T99:00:00"
    report = analyzer.analyze_records(_candidate_overlapping_baseline_formal(), baseline)

    assert any("invalid formal registration log timestamp" in error
               for error in report["data_quality"]["errors"])
    assert report["data_quality"]["formal_overlap_count"] == 0


def test_formal_registration_replica_permutation_is_deterministic():
    baseline = _single_formal_baseline()
    valid = next(record for record in baseline if record.get("event") == "resonance_decision")
    identity_invalid = dict(valid)
    identity_invalid["code"] = "159915.XSHE"

    before_valid = [record for record in baseline if record is not valid]
    before_valid.insert(1, identity_invalid)
    before_valid.insert(2, valid)
    after_valid = [record for record in baseline if record is not valid]
    after_valid.insert(1, valid)
    after_valid.insert(2, identity_invalid)

    first = analyzer.analyze_records(_candidate_overlapping_baseline_formal(), before_valid)
    second = analyzer.analyze_records(_candidate_overlapping_baseline_formal(), after_valid)

    assert first == second
    assert first["data_quality"]["formal_overlap_count"] == 0
    assert any("duplicate formal registration" in error
               for error in first["data_quality"]["errors"])


def test_formal_follow_up_decisions_are_not_registration_replicas():
    baseline = _single_formal_baseline()
    registration = next(record for record in baseline
                        if record.get("event") == "resonance_decision")
    follow_ups = [
        dict(registration, accepted=True, reason="BUY_CANDIDATE_SORTED:1"),
        dict(registration, accepted=False, reason="HELD_NO_ADD"),
        dict(registration, accepted=False, reason="UNHELD_RECORD_ONLY"),
    ]
    expected = analyzer.analyze_records(
        _candidate_overlapping_baseline_formal(), baseline,
    )
    before = [record for record in baseline if record is not registration]
    before[1:1] = follow_ups + [registration]
    after = list(baseline) + follow_ups

    first = analyzer.analyze_records(
        _candidate_overlapping_baseline_formal(), before,
    )
    second = analyzer.analyze_records(
        _candidate_overlapping_baseline_formal(), after,
    )

    assert first == second == expected
    assert first["data_quality"]["formal_overlap_count"] == 1

    bad_complete = dict(registration, code="159915.XSHE")
    report = analyzer.analyze_records(
        _candidate_overlapping_baseline_formal(), baseline + [bad_complete],
    )

    assert report["data_quality"]["formal_overlap_count"] == 0
    assert any("duplicate formal registration" in error
               for error in report["data_quality"]["errors"])


def test_filled_orders_and_frozen_summaries_require_emitter_time_and_date():
    candidate = make_candidate_records()
    baseline = make_baseline_records()
    for record in candidate + baseline:
        if record.get("event") == "order_transition":
            record["_log_timestamp"] = record["_log_date"] + "T00:00:00"
    candidate_summary = next(record for record in candidate
                             if record.get("closing_date") == "2021-12-31")
    baseline_summary = next(record for record in baseline
                            if record.get("closing_date") == "2021-12-31")
    candidate_summary["_log_timestamp"] = "2019-01-05T15:30:00"
    baseline_summary["_log_timestamp"] = "2021-12-31T14:59:00"

    report = analyzer.analyze_records(candidate, baseline)

    assert report["continue_candidate"] is False
    assert any("filled order log timestamp outside 09:35" in error
               for error in report["data_quality"]["errors"])
    assert any("candidate frozen portfolio summary log date mismatch" in error
               for error in report["data_quality"]["errors"])
    assert any("baseline frozen portfolio summary log timestamp before 15:30" in error
               for error in report["data_quality"]["errors"])
    assert report["metrics"]["final_asset"] is None


def test_cli_rejects_midnight_business_log_chronology(tmp_path):
    candidate = make_candidate_records()
    baseline = make_baseline_records()
    for record in candidate + baseline:
        if record.get("event") in {
                "relative_resonance_observation", "observation_outcome",
                "resonance_decision"}:
            record["_log_timestamp"] = "2019-01-01T00:00:00"
    candidate_path = tmp_path / "candidate.log"
    baseline_path = tmp_path / "baseline.log"
    output_path = tmp_path / "report.json"
    _write_log(candidate_path, candidate)
    _write_log(baseline_path, baseline)

    completed = subprocess.run([
        sys.executable, str(ANALYZER_PATH), "--candidate-log", str(candidate_path),
        "--baseline-log", str(baseline_path), "--output", str(output_path),
    ], capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["continue_candidate"] is False
    assert any("log timestamp" in error for error in report["data_quality"]["errors"])


def test_relative_outcome_requires_closing_day_after_close_timestamp_and_strict_sequence():
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    outcomes = {record["horizon"]: record for record in candidate
                if record.get("event") == "observation_outcome"
                and record.get("relative_observation_id") == registration["relative_observation_id"]}
    outcomes[1]["_log_timestamp"] = outcomes[1]["outcome"]["closing_date"] + "T14:59:00"
    outcomes[3]["_log_timestamp"] = outcomes[1]["_log_timestamp"]

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["continue_candidate"] is False
    assert any("relative outcome log timestamp before 15:30" in error
               for error in report["data_quality"]["errors"])
    assert any("relative outcome log timestamps are not strictly increasing" in error
               for error in report["data_quality"]["errors"])


def test_baseline_formal_timestamp_must_match_closing_date_and_after_close():
    baseline = make_baseline_records()
    formal_outcome = next(record for record in baseline
                          if record.get("event") == "observation_outcome")
    formal_outcome["_log_timestamp"] = "2021-01-05T14:59:00"

    report = analyzer.analyze_records(make_candidate_records(), baseline)

    assert report["continue_candidate"] is False
    assert any("formal outcome log date mismatch" in error
               for error in report["data_quality"]["errors"])
    assert any("formal outcome log timestamp before 15:30" in error
               for error in report["data_quality"]["errors"])


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
        if record["event"] == "relative_resonance_observation"
    )
    observation_id = observation["relative_observation_id"]
    outcome = next(
        record for record in candidate
        if record.get("relative_observation_id") == observation_id
        and record.get("horizon") == 5
    )
    candidate.extend((dict(observation), dict(outcome)))

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["metrics"]["candidate_count"] == 30
    assert report["data_quality"]["errors"] == [
        "duplicate relative candidate: %s" % observation_id,
        "duplicate relative outcome: %s/5" % observation_id,
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
        record.get("_log_timestamp", "2021-01-05T09:35:00").replace("T", " ")
        + " - INFO - "
        + json.dumps({key: value for key, value in record.items() if not key.startswith("_")}, sort_keys=True)
        for record in make_candidate_records()
    ]
    baseline_lines = [
        record.get("_log_timestamp", "2021-01-05T09:35:00").replace("T", " ")
        + " - INFO - "
        + json.dumps({key: value for key, value in record.items() if not key.startswith("_")}, sort_keys=True)
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
    next(record for record in candidate
         if record.get("event") == "relative_resonance_observation")["signal_date"] += " trailing"
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

    assert callable(getattr(analyzer, "build_relative_observation_id", None))
    assert analyzer.build_relative_observation_id(registration) == registration["relative_observation_id"]
    assert registration["supporters"] == ["BOLL", "KDJ", "RSI"]
    assert not any("impossible candidate branch" in error
                   for error in report["data_quality"]["errors"])


def test_relative_id_must_match_frozen_task4_digest_not_only_namespace_prefix():
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    outcomes = [record for record in candidate
                if record.get("event") == "observation_outcome"
                and record.get("relative_observation_id") == registration["relative_observation_id"]]
    registration["relative_observation_id"] = "RELATIVE:00"
    for outcome in outcomes:
        outcome["relative_observation_id"] = "RELATIVE:00"
        outcome["resonance_id"] = "RELATIVE:00"

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["continue_candidate"] is False
    assert any("relative observation id digest mismatch" in error
               for error in report["data_quality"]["errors"])


def test_relative_supporters_require_one_same_day_evidence_item():
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    outcomes = [record for record in candidate
                if record.get("event") == "observation_outcome"
                and record.get("relative_observation_id") == registration["relative_observation_id"]]
    old_date = (date.fromisoformat(registration["signal_date"]) - timedelta(days=1)).isoformat()
    registration["supporter_event_dates"] = {
        indicator: old_date for indicator in registration["supporters"]
    }
    _replace_relative_identity(registration, outcomes)
    for outcome in outcomes:
        outcome["supporter_event_dates"] = dict(registration["supporter_event_dates"])

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["continue_candidate"] is False
    assert any("candidate supporters lack signal-date evidence" in error
               for error in report["data_quality"]["errors"])


@pytest.mark.parametrize("closing_dates", [
    ("same",), ("reverse",),
])
def test_relative_recorded_outcomes_require_strict_horizon_closing_order(closing_dates):
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    outcomes = {record["horizon"]: record for record in candidate
                if record.get("event") == "observation_outcome"
                and record.get("relative_observation_id") == registration["relative_observation_id"]}
    if closing_dates == ("same",):
        outcomes[1]["outcome"]["closing_date"] = registration["signal_date"]
    else:
        outcomes[3]["outcome"]["closing_date"] = outcomes[1]["outcome"]["closing_date"]

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert report["continue_candidate"] is False
    assert any("relative RECORDED closing dates are not strictly ordered" in error
               for error in report["data_quality"]["errors"])


def test_outcomes_require_exact_runtime_trading_sessions():
    candidate = make_candidate_records()
    baseline = make_baseline_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    relative_horizon_three = next(
        record for record in candidate
        if record.get("event") == "observation_outcome"
        and record.get("relative_observation_id") == registration["relative_observation_id"]
        and record.get("horizon") == 3
    )
    relative_horizon_three["outcome"]["closing_date"] = "2019-01-06"
    relative_horizon_three["_log_timestamp"] = "2019-01-06T15:30:00"
    formal_horizon_five = next(record for record in baseline
                               if record.get("event") == "observation_outcome")
    formal_horizon_five["outcome"]["closing_date"] = "2021-01-06"
    formal_horizon_five["_log_timestamp"] = "2021-01-06T15:30:00"

    report = analyzer.analyze_records(candidate, baseline)

    assert report["continue_candidate"] is False
    assert any("relative outcome closing session mismatch" in error
               for error in report["data_quality"]["errors"])
    assert any("formal outcome closing session mismatch" in error
               for error in report["data_quality"]["errors"])


def test_session_calendar_rejects_shifted_outcomes_and_missing_evidence():
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    outcomes = sorted((record for record in candidate
                       if record.get("event") == "observation_outcome"
                       and record.get("relative_observation_id")
                       == registration["relative_observation_id"]),
                      key=lambda record: record["horizon"])
    for outcome, closing_date in zip(outcomes, ("2019-01-08", "2019-01-10", "2019-01-14")):
        outcome["outcome"]["closing_date"] = closing_date
        outcome["_log_timestamp"] = closing_date + "T15:30:00"

    shifted = analyzer.analyze_records(candidate, make_baseline_records())
    without_calendar = analyzer.analyze_records(
        [record for record in make_candidate_records()
         if record.get("event") != "portfolio_summary"],
        make_baseline_records(),
    )

    assert any("relative outcome closing session mismatch" in error
               for error in shifted["data_quality"]["errors"])
    assert any("missing candidate portfolio_summary session evidence" in error
               for error in without_calendar["data_quality"]["errors"])


def test_session_calendar_accepts_holiday_spanning_runtime_horizon():
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation"
                        and record.get("signal_date") == "2020-01-17")
    horizon_five = next(record for record in candidate
                       if record.get("relative_observation_id")
                       == registration["relative_observation_id"]
                       and record.get("horizon") == 5)

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert horizon_five["outcome"]["closing_date"] == "2020-02-03"
    assert not any("outcome closing session mismatch" in error
                   for error in report["data_quality"]["errors"])


def test_horizon_calendar_requires_matching_bidirectional_summary_evidence():
    complete_candidate = make_candidate_records()
    complete_baseline = make_baseline_records()
    complete = analyzer.analyze_records(complete_candidate, complete_baseline)

    missing_candidate_snapshot = [record for record in make_candidate_records()
                                  if not (record.get("event") == "portfolio_summary"
                                          and record.get("closing_date") == "2019-01-03")]
    missing_baseline_formal = [record for record in make_baseline_records()
                               if not (record.get("event") == "portfolio_summary"
                                       and record.get("closing_date") == "2021-01-12")]
    different_sets = [record for record in make_candidate_records()
                      if not (record.get("event") == "portfolio_summary"
                              and record.get("closing_date") == "2019-01-23")]
    invalid_initialization = make_baseline_records()
    invalid_initialization[0]["parameter_fingerprint"] = "bad"
    conflicting_summary = make_candidate_records()
    summary = next(record for record in conflicting_summary
                   if record.get("event") == "portfolio_summary"
                   and record.get("closing_date") == "2019-01-23")
    conflicting_summary.append(dict(summary, total_value=1.0))

    candidate_report = analyzer.analyze_records(missing_candidate_snapshot,
                                                 make_baseline_records())
    baseline_report = analyzer.analyze_records(make_candidate_records(),
                                                missing_baseline_formal)
    different_report = analyzer.analyze_records(different_sets, make_baseline_records())
    invalid_initialization_report = analyzer.analyze_records(
        make_candidate_records(), invalid_initialization,
    )
    conflicting_summary_report = analyzer.analyze_records(
        conflicting_summary, make_baseline_records(),
    )
    reversed_report = analyzer.analyze_records(
        list(reversed(complete_candidate)), list(reversed(complete_baseline)),
    )

    assert complete["continue_candidate"] is True
    assert any("candidate session evidence missing summary: 2019-01-03" in error
               for error in candidate_report["data_quality"]["errors"])
    assert any("baseline session evidence missing summary: 2021-01-12" in error
               for error in baseline_report["data_quality"]["errors"])
    for report in (candidate_report, baseline_report, different_report,
                   invalid_initialization_report, conflicting_summary_report):
        assert report["continue_candidate"] is False
        assert any("candidate/baseline session calendar dates differ" in error
                   for error in report["data_quality"]["errors"])
    assert any("baseline session calendar unavailable: invalid initialization" in error
               for error in invalid_initialization_report["data_quality"]["errors"])
    assert any("conflicting candidate session calendar evidence: 2019-01-23" in error
               for error in conflicting_summary_report["data_quality"]["errors"])
    assert reversed_report == complete


def test_session_evidence_and_namespace_contract_are_symmetric_by_role():
    candidate = [record for record in make_candidate_records()
                 if not (record.get("event") == "portfolio_summary"
                         and record.get("closing_date") == "2021-01-12")]
    baseline = [record for record in make_baseline_records()
                if not (record.get("event") == "portfolio_summary"
                        and record.get("closing_date") == "2021-01-12")]
    baseline.append({
        "event": "signal_snapshot", "build": "20260827.3",
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "code": "510300.XSHG", "decision_date": "2021-01-12",
        "signal_date": "2021-01-11", "valid": True,
        "_log_timestamp": "2021-01-12T09:35:00",
    })
    baseline.append({
        "event": "observation_outcome", "resonance_id": "RELATIVE:foreign",
        "relative_observation_id": "RELATIVE:foreign",
        "observation_kind": "RELATIVE_RESONANCE", "code": "510300.XSHG",
        "direction": "BUY_TURN", "branch": "SOFT_ALL_THREE",
        "event_date": "2021-01-05", "horizon": 5,
        "outcome": {"status": "PRICE_UNAVAILABLE", "closing_date": "2021-01-12"},
        "_log_timestamp": "2021-01-12T15:30:00",
    })

    report = analyzer.analyze_records(candidate, baseline)
    normal = analyzer.analyze_records(make_candidate_records(), make_baseline_records())

    assert report["continue_candidate"] is False
    assert any("candidate session evidence missing summary: 2021-01-12" in error
               for error in report["data_quality"]["errors"])
    assert any("baseline session evidence missing summary: 2021-01-12" in error
               for error in report["data_quality"]["errors"])
    assert any("baseline non-formal observation outcome is not allowed" in error
               for error in report["data_quality"]["errors"])
    assert normal["continue_candidate"] is True


def test_formal_horizon_five_must_close_after_its_signal_date():
    baseline = make_baseline_records()
    formal_outcome = next(record for record in baseline
                          if record.get("event") == "observation_outcome")
    formal_outcome["outcome"]["closing_date"] = formal_outcome["event_date"]

    report = analyzer.analyze_records(make_candidate_records(), baseline)

    assert report["continue_candidate"] is False
    assert any("formal horizon 5 closing must follow signal date" in error
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


def test_conflicting_relative_namespace_records_are_excluded_from_formal_statistics():
    baseline = make_baseline_records()
    for index in range(100):
        resonance_id = "RELATIVE:poison-%03d" % index
        baseline.extend((
            {"event": "resonance_decision", "accepted": True,
             "reason": "COMPLETE_RESONANCE", "resonance_id": resonance_id,
             "relative_observation_id": resonance_id,
             "observation_kind": "RELATIVE_RESONANCE", "code": "510300.XSHG",
             "direction": "BUY_TURN", "signal_date": "2021-01-05"},
            {"event": "observation_outcome", "resonance_id": resonance_id,
             "relative_observation_id": resonance_id,
             "observation_kind": "RELATIVE_RESONANCE", "code": "510300.XSHG",
             "event_date": "2021-01-05", "horizon": 5,
             "outcome": {"status": "RECORDED", "closing_date": "2021-01-05",
                         "return": -0.9}},
        ))

    report = analyzer.analyze_records(make_candidate_records(), baseline)

    assert report["metrics"]["formal_horizon_5"]["count"] == 30
    assert report["metrics"]["formal_horizon_5"]["q1"] == pytest.approx(0.01)
    assert report["data_quality"]["formal_missing_outcome_count"] == 0
    assert report["continue_candidate"] is False


def test_cli_code_container_is_quality_error_but_still_writes_report(tmp_path, capsys):
    candidate = make_candidate_records()
    next(record for record in candidate if record.get("event") == "relative_resonance_observation")["code"] = ["510300.XSHG"]
    candidate_path = tmp_path / "candidate.log"
    baseline_path = tmp_path / "baseline.log"
    output_path = tmp_path / "report.json"
    _write_log(candidate_path, candidate)
    _write_log(baseline_path, make_baseline_records())

    status = analyzer.main(["--candidate-log", str(candidate_path),
                            "--baseline-log", str(baseline_path), "--output", str(output_path)])

    assert status == 0
    assert capsys.readouterr().err == ""
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert any("candidate code" in error for error in report["data_quality"]["errors"])
    assert report["continue_candidate"] is False


def test_all_terminal_outcome_statuses_require_training_window_closing_dates():
    relative = make_relative_record(0)
    relative_outcome = make_outcome(relative, 5, 0.02)
    relative_outcome["outcome"] = {"status": "PRICE_UNAVAILABLE", "closing_date": "2022-01-03"}
    baseline = make_baseline_records()
    formal = next(record for record in baseline if record.get("event") == "observation_outcome")
    formal["outcome"] = {"status": "PRICE_UNAVAILABLE", "closing_date": "2022-01-03"}

    report = analyzer.analyze_records([make_initialized_record(), relative, relative_outcome], baseline)

    assert any("relative outcome closing outside 2019-2021: 2022-01-03" in error
               for error in report["data_quality"]["errors"])
    assert any("formal outcome closing outside 2019-2021: 2022-01-03" in error
               for error in report["data_quality"]["errors"])


def test_cli_replace_and_temporary_cleanup_failures_preserve_primary_io_error(tmp_path, capsys):
    candidate_path = tmp_path / "candidate.log"
    baseline_path = tmp_path / "baseline.log"
    output_path = tmp_path / "report.json"
    _write_log(candidate_path, make_candidate_records())
    _write_log(baseline_path, make_baseline_records())
    output_path.write_bytes(b"old output")

    with mock.patch.object(analyzer.os, "replace", side_effect=OSError("replace fault")), \
            mock.patch.object(pathlib.Path, "unlink", side_effect=OSError("cleanup fault")):
        status = analyzer.main(["--candidate-log", str(candidate_path),
                                "--baseline-log", str(baseline_path), "--output", str(output_path)])

    assert status == 2
    assert capsys.readouterr().err == "output error: replace fault\n"
    assert output_path.read_bytes() == b"old output"


def test_real_formal_decision_and_outcome_schema_do_not_enter_overlap_as_outcome_fields():
    candidate = make_candidate_records()
    candidate.extend((
        {"event": "resonance_decision", "accepted": True, "reason": "COMPLETE_RESONANCE",
         "resonance_id": "FORMAL:candidate", "code": "510300.XSHG",
             "direction": "BUY_TURN", "signal_date": "2021-01-15"},
            {"event": "observation_outcome", "resonance_id": "FORMAL:candidate",
             "code": "510300.XSHG", "event_date": "2021-01-15", "horizon": 5,
             "outcome": {"status": "RECORDED", "closing_date": "2021-01-15", "return": 0.01}},
    ))

    report = analyzer.analyze_records(candidate, make_baseline_records())

    assert not any("formal overlap direction" in error or "formal overlap signal date" in error
                   for error in report["data_quality"]["errors"])
    assert report["continue_candidate"] is True


def test_candidate_formal_terminal_outcomes_always_audit_closing_date():
    candidate = [make_initialized_record(), {
        "event": "observation_outcome", "resonance_id": "FORMAL:candidate",
        "code": "510300.XSHG", "event_date": "2021-01-05", "horizon": 5,
        "outcome": {"status": "PRICE_UNAVAILABLE", "closing_date": "2022-01-03"},
    }, {
        "event": "observation_outcome", "resonance_id": "FORMAL:recorded",
        "code": "510300.XSHG", "event_date": "2021-01-05", "horizon": 5,
        "outcome": {"status": "RECORDED", "closing_date": "2022-01-04", "return": 0.01},
    }]

    report = analyzer.analyze_records(candidate, [])

    assert any("outcome closing outside 2019-2021: 2022-01-03" in error
               for error in report["data_quality"]["errors"])
    assert any("outcome closing outside 2019-2021: 2022-01-04" in error
               for error in report["data_quality"]["errors"])


@pytest.mark.parametrize("role,payload", [
    ("candidate", 1), ("candidate", True), ("candidate", "not-a-payload"),
    ("baseline", 1), ("baseline", True), ("baseline", "not-a-payload"),
])
def test_cli_non_dict_formal_outcome_payload_is_isolated_as_data_quality(
        tmp_path, role, payload):
    candidate = make_candidate_records()
    baseline = make_baseline_records()
    target = candidate if role == "candidate" else baseline
    target.append({
        "event": "observation_outcome", "resonance_id": "FORMAL:payload-%s" % role,
        "code": "510300.XSHG", "event_date": "2021-01-05", "horizon": 5,
        "outcome": payload,
    })
    candidate_path = tmp_path / (role + "-candidate.log")
    baseline_path = tmp_path / (role + "-baseline.log")
    output_path = tmp_path / (role + "-report.json")
    _write_log(candidate_path, candidate)
    _write_log(baseline_path, baseline)

    completed = subprocess.run([
        sys.executable, str(ANALYZER_PATH), "--candidate-log", str(candidate_path),
        "--baseline-log", str(baseline_path), "--output", str(output_path),
    ], capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["continue_candidate"] is False
    assert any("invalid outcome payload" in error
               for error in report["data_quality"]["errors"])


def test_cli_surrogate_code_is_safely_isolated_as_data_quality(tmp_path):
    candidate = make_candidate_records()
    registration = next(record for record in candidate
                        if record.get("event") == "relative_resonance_observation")
    registration["code"] = "\ud800"
    candidate_path = tmp_path / "surrogate-candidate.log"
    baseline_path = tmp_path / "surrogate-baseline.log"
    output_path = tmp_path / "surrogate-report.json"
    _write_log(candidate_path, candidate)
    _write_log(baseline_path, make_baseline_records())

    completed = subprocess.run([
        sys.executable, str(ANALYZER_PATH), "--candidate-log", str(candidate_path),
        "--baseline-log", str(baseline_path), "--output", str(output_path),
    ], capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["continue_candidate"] is False
    assert any("candidate code" in error for error in report["data_quality"]["errors"])


@pytest.mark.parametrize("field,value,fragment", [
    ("direction", ["BUY_TURN"], "candidate direction"),
    ("branch", ["SOFT_ALL_THREE"], "branch"),
    ("supporters", 3, "candidate supporters"),
])
def test_cli_identity_container_pollution_isolated_without_traceback(tmp_path, field, value, fragment):
    candidate = make_candidate_records()
    registration = next(record for record in candidate if record.get("event") == "relative_resonance_observation")
    registration[field] = value
    candidate_path = tmp_path / (field + "-candidate.log")
    baseline_path = tmp_path / (field + "-baseline.log")
    output_path = tmp_path / (field + "-report.json")
    _write_log(candidate_path, candidate)
    _write_log(baseline_path, make_baseline_records())

    completed = subprocess.run([
        sys.executable, str(ANALYZER_PATH), "--candidate-log", str(candidate_path),
        "--baseline-log", str(baseline_path), "--output", str(output_path),
    ], capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert any(fragment in error for error in report["data_quality"]["errors"])
    assert report["continue_candidate"] is False
