import importlib.util
import json
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    ROOT / "resonance_reversal_strategy" / "research"
    / "analyze_relative_turn_observations.py"
)
spec = importlib.util.spec_from_file_location("relative_analyzer", ANALYZER_PATH)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


BUILD = "20260827.4"
FINGERPRINT = "f47d32b87be6d926"


def make_order_path(log_date="2021-01-05"):
    return [
        {
            "event": "order_transition", "_log_date": log_date,
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
    return {
        "event": "relative_resonance_observation",
        "relative_observation_id": "RELATIVE:%02d" % index,
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": branch,
        "code": code,
        "direction": direction,
        "signal_date": signal_date,
        "expires_date": signal_date,
        "supporters": ["BOLL", "RSI"],
        "build": BUILD,
        "relative_observation_fingerprint": FINGERPRINT,
    }


def make_outcome(record, horizon, value):
    return {
        "event": "observation_outcome",
        "relative_observation_id": record["relative_observation_id"],
        "observation_kind": "RELATIVE_RESONANCE",
        "branch": record["branch"], "direction": record["direction"],
        "code": record["code"], "event_date": record["signal_date"],
        "horizon": horizon, "build": BUILD,
        "relative_observation_fingerprint": FINGERPRINT,
        "outcome": {
            "status": "RECORDED", "closing_date": record["signal_date"],
            "return": value, "direction_adjusted_return": value,
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
        "total_value": 23856.40,
    })
    return records


def make_baseline_records():
    records = make_order_path()
    records.append({
        "event": "portfolio_summary", "closing_date": "2021-12-31",
        "total_value": 23856.40,
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
            "outcome": {"status": "RECORDED", "return": 0.01},
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


def test_analyzer_rejects_validation_period_qualified_observation():
    candidate = [make_initialized_record(), {
        "event": "relative_resonance_observation",
        "relative_observation_id": "RELATIVE:2022",
        "signal_date": "2022-01-04", "expires_date": "2022-01-04",
        "build": BUILD, "relative_observation_fingerprint": FINGERPRINT,
    }]

    with pytest.raises(ValueError, match="2022"):
        analyzer.analyze_records(candidate, [])


def test_empty_input_is_reported_as_incomplete_without_writing_state():
    report = analyzer.analyze_records([], [])

    assert report["metrics"]["candidate_count"] == 0
    assert report["data_quality"]["errors"] == [
        "missing matching strategy_initialized record",
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
    assert report["metrics"]["by_branch"]["SOFT_ALL_THREE"]["median"] == pytest.approx(0.03)
    assert report["metrics"]["by_branch"]["HARD_BOLL_SOFT_OSC"]["median"] == pytest.approx(-0.04)
    assert report["metrics"]["horizon_1"]["median"] == pytest.approx(0.0)
    assert report["metrics"]["horizon_3"]["median"] == pytest.approx(0.0)


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
    assert report["data_quality"]["errors"] == []


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
        "2021-01-05 - INFO - " + json.dumps(record, sort_keys=True)
        for record in make_candidate_records()
    ]
    baseline_lines = [
        "2021-01-05 - INFO - " + json.dumps(record, sort_keys=True)
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
