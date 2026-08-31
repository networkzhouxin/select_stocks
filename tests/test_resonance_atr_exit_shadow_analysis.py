import hashlib
import importlib.util
import json
import pathlib
import sys
from datetime import date, datetime, timedelta

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    ROOT / "resonance_reversal_strategy" / "research"
    / "analyze_atr_exit_shadows.py"
)


def load_analyzer():
    if not ANALYZER_PATH.is_file():
        pytest.fail("ATR exit shadow analyzer module is missing")
    spec = importlib.util.spec_from_file_location(
        "atr_exit_shadow_analyzer", ANALYZER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluation_sessions():
    weekdays = []
    current = date(2019, 1, 1)
    while current <= date(2021, 12, 31):
        if current.weekday() < 5:
            weekdays.append(current)
        current += timedelta(days=1)
    return tuple(weekdays[:728] + [date(2021, 12, 30), date(2021, 12, 31)])


def validated_manifest(analyzer):
    sessions = (date(2018, 1, 2),) + evaluation_sessions()
    payload = {
        "schema_version": 2,
        "market": "XSHG",
        "calendar_coverage_start": "2018-01-01",
        "calendar_coverage_end": "2021-12-31",
        "evaluation_start": "2019-01-01",
        "evaluation_end": "2021-12-31",
        "source": "JoinQuant get_all_trade_days",
        "sessions": [session.isoformat() for session in sessions],
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return analyzer.validate_session_calendar_manifest(
        raw, hashlib.sha256(raw).hexdigest(),
    )


def record(event, timestamp, **payload):
    result = {
        "event": event,
        "_log_date": timestamp.date().isoformat(),
        "_log_timestamp": timestamp.isoformat(),
    }
    result.update(payload)
    return result


def initialization(build, atr_fingerprint=None):
    payload = {
        "build": build,
        "parameter_fingerprint": "e1227fbd8b4a884e",
        "pool_fingerprint": "9123995edeb1ed84",
        "event_logic_fingerprint": "1c0b8a22f48c97c3",
        "relative_observation_fingerprint": "f47d32b87be6d926",
    }
    if atr_fingerprint is not None:
        payload["atr_shadow_fingerprint"] = atr_fingerprint
    return record(
        "strategy_initialized", datetime(2019, 1, 1), **payload,
    )


def formal_records(build, atr_fingerprint=None, order_price=10.0):
    sessions = evaluation_sessions()
    records = [initialization(build, atr_fingerprint)]
    for index in range(138):
        records.append(record(
            "order_transition", datetime(2019, 1, 2, 9, 35),
            code="510300.XSHG", side="BUY" if index % 2 == 0 else "SELL",
            outcome="FILLED", before_amount=index,
            after_amount=index + 1, requested_target=1000.0,
            pending_exit=None, trade_price=order_price,
            entrust_id="IGNORED-%s" % index,
        ))
    for index, session in enumerate(sessions):
        total_value = 23856.4 if index == len(sessions) - 1 else 20000 + index
        records.append(record(
            "portfolio_summary", datetime.combine(
                session, datetime.min.time().replace(hour=15, minute=30),
            ),
            closing_date=session.isoformat(), total_value=total_value,
            available_cash=1000.0,
            positions={"510300.XSHG": 100},
            highest_close_anchors={"510300.XSHG": 10.0},
        ))
    return records


def add_complete_shadow(records, analyzer, index, event_session_index=0):
    sessions = evaluation_sessions()
    event_date = sessions[event_session_index]
    shadow_id = "ATR_SHADOW:%020d" % index
    records.append(record(
        "atr_exit_shadow_registered", datetime.combine(
            event_date, datetime.min.time().replace(hour=9, minute=35),
        ),
        atr_shadow_id=shadow_id, observation_kind="ATR_EXIT_SHADOW",
        code="ETF%02d.XSHG" % index, event_date=event_date.isoformat(),
        reference_price=10.0, entry_price=10.2, entry_atr=1.0,
        highest_close_anchor=11.0, stop_price=9.0,
        horizons=[1, 3, 5], outcomes={}, build=analyzer.CANDIDATE_BUILD,
    ))
    for horizon in (1, 3, 5):
        closing_date = sessions[event_session_index + horizon]
        outcome = {
            "status": "RECORDED",
            "closing_date": closing_date.isoformat(),
            "closing_price": 10.2,
            "return": 0.02,
            "recovered_entry": True,
        }
        records.append(record(
            "atr_exit_shadow_outcome", datetime.combine(
                closing_date,
                datetime.min.time().replace(hour=15, minute=30),
            ),
            atr_shadow_id=shadow_id, observation_kind="ATR_EXIT_SHADOW",
            code="ETF%02d.XSHG" % index,
            event_date=event_date.isoformat(), horizon=horizon,
            outcome=outcome, build=analyzer.CANDIDATE_BUILD,
        ))


def analyze_fixture(candidate_order_price=10.0, baseline_order_price=10.0,
                    registration_date=None, outcomes=True,
                    complete_shadow_count=20):
    analyzer = load_analyzer()
    baseline = formal_records(
        analyzer.BASELINE_BUILD, order_price=baseline_order_price,
    )
    candidate = formal_records(
        analyzer.CANDIDATE_BUILD,
        atr_fingerprint=analyzer.ATR_SHADOW_FINGERPRINT,
        order_price=candidate_order_price,
    )
    if registration_date is None:
        for index in range(complete_shadow_count):
            add_complete_shadow(candidate, analyzer, index)
    else:
        event_date = date.fromisoformat(registration_date)
        shadow_id = "ATR_SHADOW:%020d" % 1
        candidate.append(record(
            "atr_exit_shadow_registered", datetime.combine(
                event_date,
                datetime.min.time().replace(hour=9, minute=35),
            ),
            atr_shadow_id=shadow_id, observation_kind="ATR_EXIT_SHADOW",
            code="510300.XSHG", event_date=event_date.isoformat(),
            reference_price=10.0, entry_price=10.2, entry_atr=1.0,
            highest_close_anchor=11.0, stop_price=9.0,
            horizons=[1, 3, 5], outcomes={},
            build=analyzer.CANDIDATE_BUILD,
        ))
        if outcomes:
            raise AssertionError("custom registration fixture only supports no outcomes")
    return analyzer.analyze_records(
        candidate, baseline, validated_manifest(analyzer),
    )


def test_shadow_analyzer_accepts_complete_positive_noninterfering_fixture():
    report = analyze_fixture()

    assert report["data_quality"]["errors"] == []
    assert report["data_quality"]["complete_count"] == 20
    assert report["gates"]["formal_order_path_exact"] is True
    assert report["gates"]["portfolio_path_exact"] is True
    assert report["continue_atr_investigation"] is True


def test_shadow_analyzer_requires_exact_order_and_portfolio_path():
    report = analyze_fixture(
        candidate_order_price=10.01, baseline_order_price=10.00,
    )

    assert report["gates"]["formal_order_path_exact"] is False
    assert report["continue_atr_investigation"] is False


def test_shadow_analyzer_right_censors_training_end_without_2022():
    report = analyze_fixture(
        registration_date="2021-12-30", outcomes=False,
        complete_shadow_count=0,
    )

    assert report["data_quality"]["right_censored_count"] == 1
    assert report["data_quality"]["errors"] == []
    assert report["continue_atr_investigation"] is False


def test_shadow_analyzer_right_censors_partial_in_window_outcome():
    analyzer = load_analyzer()
    baseline = formal_records(analyzer.BASELINE_BUILD)
    candidate = formal_records(
        analyzer.CANDIDATE_BUILD,
        atr_fingerprint=analyzer.ATR_SHADOW_FINGERPRINT,
    )
    event_date = date(2021, 12, 30)
    closing_date = date(2021, 12, 31)
    shadow_id = "ATR_SHADOW:%020d" % 1
    candidate.append(record(
        "atr_exit_shadow_registered", datetime(2021, 12, 30, 9, 35),
        atr_shadow_id=shadow_id, observation_kind="ATR_EXIT_SHADOW",
        code="513050.XSHG", event_date=event_date.isoformat(),
        reference_price=1.205, entry_price=1.272, entry_atr=0.023857,
        highest_close_anchor=1.272, stop_price=1.2084,
        horizons=[1, 3, 5], outcomes={}, build=analyzer.CANDIDATE_BUILD,
    ))
    candidate.append(record(
        "atr_exit_shadow_outcome", datetime(2021, 12, 31, 15, 30),
        atr_shadow_id=shadow_id, observation_kind="ATR_EXIT_SHADOW",
        code="513050.XSHG", event_date=event_date.isoformat(), horizon=1,
        outcome={
            "status": "RECORDED",
            "closing_date": closing_date.isoformat(),
            "closing_price": 1.264,
            "return": 0.0489626556016598,
            "recovered_entry": False,
        },
        build=analyzer.CANDIDATE_BUILD,
    ))

    report = analyzer.analyze_records(
        candidate, baseline, validated_manifest(analyzer),
    )

    assert report["data_quality"] == {
        "errors": [],
        "registration_count": 1,
        "complete_count": 0,
        "right_censored_count": 1,
    }
    assert report["metrics"]["horizons"]["1"]["count"] == 0
    assert report["continue_atr_investigation"] is False


def test_shadow_analyzer_rejects_outcome_after_training_end():
    analyzer = load_analyzer()
    baseline = formal_records(analyzer.BASELINE_BUILD)
    candidate = formal_records(
        analyzer.CANDIDATE_BUILD,
        atr_fingerprint=analyzer.ATR_SHADOW_FINGERPRINT,
    )
    add_complete_shadow(candidate, analyzer, 1)
    outcome = next(
        item for item in candidate
        if item["event"] == "atr_exit_shadow_outcome"
    )
    outcome["outcome"]["closing_date"] = "2022-01-04"

    report = analyzer.analyze_records(
        candidate, baseline, validated_manifest(analyzer),
    )

    assert any("outside 2019-2021" in error
               for error in report["data_quality"]["errors"])
    assert report["continue_atr_investigation"] is False


def test_shadow_analyzer_cli_rejects_input_output_alias(tmp_path):
    analyzer = load_analyzer()
    baseline = tmp_path / "baseline.log"
    candidate = tmp_path / "candidate.log"
    calendar = tmp_path / "calendar.json"
    for path in (baseline, candidate, calendar):
        path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be distinct"):
        analyzer.main([
            "--baseline-log", str(baseline),
            "--candidate-log", str(candidate),
            "--session-calendar", str(calendar),
            "--session-calendar-sha256", "0" * 64,
            "--output", str(candidate),
        ])


def test_shadow_analyzer_cli_writes_report_atomically(tmp_path, monkeypatch):
    analyzer = load_analyzer()
    baseline = tmp_path / "baseline.log"
    candidate = tmp_path / "candidate.log"
    calendar = tmp_path / "calendar.json"
    output = tmp_path / "report.json"
    for path in (baseline, candidate, calendar):
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        analyzer, "read_session_calendar_manifest_bytes", lambda path: b"{}",
    )
    monkeypatch.setattr(
        analyzer, "validate_session_calendar_manifest",
        lambda raw, digest: object(),
    )
    monkeypatch.setattr(
        analyzer, "load_log_records",
        lambda paths: tuple(str(path) for path in paths),
    )
    expected = {"continue_atr_investigation": False, "gates": {}}
    monkeypatch.setattr(
        analyzer, "analyze_records",
        lambda candidate_records, baseline_records, manifest: expected,
    )

    result = analyzer.main([
        "--baseline-log", str(baseline),
        "--candidate-log", str(candidate),
        "--session-calendar", str(calendar),
        "--session-calendar-sha256", "0" * 64,
        "--output", str(output),
    ])

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert list(tmp_path.glob(".report.json.*.tmp")) == []
