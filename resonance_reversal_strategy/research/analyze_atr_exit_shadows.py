"""Read-only validation and summaries for post-ATR exit shadows."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile
from datetime import date, datetime, time


BASELINE_BUILD = "20260827.4"
CANDIDATE_BUILD = "20260828.1"
PARAMETER_FINGERPRINT = "e1227fbd8b4a884e"
POOL_FINGERPRINT = "9123995edeb1ed84"
FORMAL_EVENT_FINGERPRINT = "1c0b8a22f48c97c3"
RELATIVE_OBSERVATION_FINGERPRINT = "f47d32b87be6d926"
EXPECTED_FILLED_ORDERS = 138
EXPECTED_PORTFOLIO_POINTS = 730
EXPECTED_FINAL_ASSET = 23856.40
EVALUATION_START = date(2019, 1, 1)
EVALUATION_END = date(2021, 12, 31)
TRADING_TIME = time(9, 35)
AFTER_CLOSE_TIME = time(15, 30)
HORIZONS = (1, 3, 5)


ATR_SHADOW_LOGIC_CONTRACT = {
    "namespace": "ATR_SHADOW",
    "registration": "FULLY_FILLED_ATR_EXIT_AFTER_STATE_SYNC",
    "reference_price": "EXIT_DECISION_09_35_QUOTE",
    "outcomes": [1, 3, 5],
    "outcome_time": "AFTER_CLOSE_ONLY",
    "trading_effect": "NONE",
}
ATR_SHADOW_FINGERPRINT = hashlib.sha256(json.dumps(
    ATR_SHADOW_LOGIC_CONTRACT,
    ensure_ascii=False,
    sort_keys=True,
).encode("utf-8")).hexdigest()[:16]


def _load_relative_analyzer():
    module_name = "_resonance_relative_turn_for_atr_shadow"
    module = sys.modules.get(module_name)
    if module is None:
        module_path = pathlib.Path(__file__).resolve().with_name(
            "analyze_relative_turn_observations.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError("unable to load manifest validator")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


_RELATIVE_ANALYZER = _load_relative_analyzer()
read_session_calendar_manifest_bytes = (
    _RELATIVE_ANALYZER.read_session_calendar_manifest_bytes
)
validate_session_calendar_manifest = (
    _RELATIVE_ANALYZER.validate_session_calendar_manifest
)
load_log_records = _RELATIVE_ANALYZER.load_log_records


def _calendar_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _log_timestamp(record):
    value = record.get("_log_timestamp")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _public_record(record, ignored=()):
    return {
        key: value for key, value in record.items()
        if not key.startswith("_") and key not in ignored
    }


def _canonical_record(record, ignored=()):
    return json.dumps(
        _public_record(record, ignored),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_initialization(records, role, expected_build, errors,
                             require_atr_fingerprint=False):
    initialized = [
        record for record in records
        if record.get("event") == "strategy_initialized"
    ]
    if len(initialized) != 1:
        errors.append(
            "%s must contain exactly one strategy initialization" % role
        )
        return
    record = initialized[0]
    expected = {
        "build": expected_build,
        "parameter_fingerprint": PARAMETER_FINGERPRINT,
        "pool_fingerprint": POOL_FINGERPRINT,
        "event_logic_fingerprint": FORMAL_EVENT_FINGERPRINT,
        "relative_observation_fingerprint": RELATIVE_OBSERVATION_FINGERPRINT,
    }
    if require_atr_fingerprint:
        expected["atr_shadow_fingerprint"] = ATR_SHADOW_FINGERPRINT
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            errors.append(
                "%s initialization %s mismatch" % (role, field)
            )


def _filled_order_path(records):
    path = []
    for record in records:
        if (record.get("event") == "order_transition"
                and record.get("outcome") == "FILLED"):
            path.append(_canonical_record(record, ignored=("entrust_id",)))
    return tuple(path)


def _portfolio_path(records, evaluation_sessions, role, errors):
    summaries = {}
    for record in records:
        if record.get("event") != "portfolio_summary":
            continue
        closing_date = _calendar_date(record.get("closing_date"))
        timestamp = _log_timestamp(record)
        if (closing_date is None
                or not EVALUATION_START <= closing_date <= EVALUATION_END):
            errors.append("%s portfolio date outside 2019-2021" % role)
            continue
        if (timestamp is None or timestamp.date() != closing_date
                or timestamp.time() < AFTER_CLOSE_TIME):
            errors.append("%s invalid portfolio timestamp" % role)
            continue
        canonical = _canonical_record(record)
        if closing_date in summaries:
            errors.append("duplicate %s portfolio summary" % role)
        summaries[closing_date] = canonical
    dates = tuple(sorted(summaries))
    if dates != evaluation_sessions:
        errors.append("%s portfolio sessions differ from manifest" % role)
    return tuple(summaries[item] for item in dates)


def _final_asset(records, role, errors):
    values = []
    for record in records:
        if (record.get("event") == "portfolio_summary"
                and record.get("closing_date") == EVALUATION_END.isoformat()):
            value = _finite_number(record.get("total_value"))
            if value is None:
                errors.append("invalid %s final asset" % role)
            else:
                values.append(value)
    if len(values) != 1:
        errors.append("%s must contain exactly one final asset" % role)
        return None
    return values[0]


def _summarize(values):
    values = tuple(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "hit_rate": None}
    return {
        "count": len(values),
        "mean": math.fsum(values) / len(values),
        "median": statistics.median(values),
        "hit_rate": sum(value > 0 for value in values) / len(values),
    }


def _expected_due_dates(event_date, evaluation_sessions):
    try:
        index = evaluation_sessions.index(event_date)
    except ValueError:
        return None
    due_dates = {}
    for horizon in HORIZONS:
        due_index = index + horizon
        due_dates[horizon] = (
            evaluation_sessions[due_index]
            if due_index < len(evaluation_sessions) else None
        )
    return due_dates


def _audit_shadows(candidate_records, baseline_records,
                   evaluation_sessions, errors):
    for record in baseline_records:
        if str(record.get("event", "")).startswith("atr_exit_shadow"):
            errors.append("baseline contains ATR shadow event")
    registrations = {}
    registration_order = {}
    outcomes = {}
    for ordinal, record in enumerate(candidate_records):
        event = record.get("event")
        if event == "atr_exit_shadow_registered":
            shadow_id = record.get("atr_shadow_id")
            if (not isinstance(shadow_id, str)
                    or not shadow_id.startswith("ATR_SHADOW:")):
                errors.append("invalid ATR shadow registration namespace")
                continue
            if shadow_id in registrations:
                errors.append("duplicate ATR shadow registration: %s" % shadow_id)
                continue
            if (record.get("observation_kind") != "ATR_EXIT_SHADOW"
                    or tuple(record.get("horizons") or ()) != HORIZONS
                    or record.get("build") != CANDIDATE_BUILD):
                errors.append("invalid ATR shadow registration contract: %s" % shadow_id)
            event_date = _calendar_date(record.get("event_date"))
            timestamp = _log_timestamp(record)
            if event_date not in evaluation_sessions:
                errors.append("ATR shadow registration date absent from manifest")
            if (timestamp is None or timestamp.date() != event_date
                    or timestamp.time() != TRADING_TIME):
                errors.append("invalid ATR shadow registration timestamp")
            for field in (
                    "reference_price", "entry_atr",
                    "highest_close_anchor", "stop_price"):
                if _finite_number(record.get(field)) is None:
                    errors.append("invalid ATR shadow %s" % field)
            registrations[shadow_id] = record
            registration_order[shadow_id] = ordinal
        elif event == "atr_exit_shadow_outcome":
            shadow_id = record.get("atr_shadow_id")
            horizon = record.get("horizon")
            if shadow_id not in registrations:
                errors.append("ATR shadow outcome before registration")
                continue
            if ordinal <= registration_order[shadow_id]:
                errors.append("ATR shadow outcome before registration")
            if horizon not in HORIZONS:
                errors.append("invalid ATR shadow outcome horizon")
                continue
            key = (shadow_id, horizon)
            if key in outcomes:
                errors.append("duplicate ATR shadow outcome")
                continue
            if (record.get("build") != CANDIDATE_BUILD
                    or record.get("observation_kind") != "ATR_EXIT_SHADOW"):
                errors.append("invalid ATR shadow outcome contract")
            payload = record.get("outcome")
            if not isinstance(payload, dict):
                errors.append("ATR shadow outcome must be an object")
                continue
            closing_date = _calendar_date(payload.get("closing_date"))
            if (closing_date is None
                    or not EVALUATION_START <= closing_date <= EVALUATION_END):
                errors.append("ATR shadow outcome outside 2019-2021")
            timestamp = _log_timestamp(record)
            if (timestamp is None or timestamp.date() != closing_date
                    or timestamp.time() < AFTER_CLOSE_TIME):
                errors.append("invalid ATR shadow outcome timestamp")
            outcomes[key] = payload
    complete_ids = []
    right_censored_ids = []
    returns_by_horizon = {horizon: [] for horizon in HORIZONS}
    returns_by_year = {}
    recovered_by_horizon = {horizon: [] for horizon in HORIZONS}
    for shadow_id, registration in registrations.items():
        event_date = _calendar_date(registration.get("event_date"))
        due_dates = _expected_due_dates(event_date, evaluation_sessions)
        if due_dates is None:
            continue
        if any(due_dates[horizon] is None for horizon in HORIZONS):
            right_censored_ids.append(shadow_id)
            for horizon in HORIZONS:
                payload = outcomes.get((shadow_id, horizon))
                if payload is None:
                    continue
                if (due_dates[horizon] is None
                        or _calendar_date(payload.get("closing_date"))
                        != due_dates[horizon]):
                    errors.append(
                        "right-censored ATR shadow outcome session mismatch"
                    )
                    continue
                if (payload.get("status") != "RECORDED"
                        or _finite_number(payload.get("return")) is None):
                    errors.append("invalid ATR shadow recorded outcome")
            continue
        complete = True
        for horizon in HORIZONS:
            payload = outcomes.get((shadow_id, horizon))
            if payload is None:
                errors.append("missing ATR shadow outcome")
                complete = False
                continue
            if _calendar_date(payload.get("closing_date")) != due_dates[horizon]:
                errors.append("ATR shadow outcome closing session mismatch")
                complete = False
            outcome_return = _finite_number(payload.get("return"))
            if payload.get("status") != "RECORDED" or outcome_return is None:
                errors.append("invalid ATR shadow recorded outcome")
                complete = False
                continue
            returns_by_horizon[horizon].append(outcome_return)
            recovered = payload.get("recovered_entry")
            if isinstance(recovered, bool):
                recovered_by_horizon[horizon].append(recovered)
            if horizon == 5:
                returns_by_year.setdefault(str(event_date.year), []).append(
                    outcome_return
                )
        if complete:
            complete_ids.append(shadow_id)
    return {
        "registration_count": len(registrations),
        "complete_count": len(complete_ids),
        "right_censored_count": len(right_censored_ids),
        "returns_by_horizon": returns_by_horizon,
        "returns_by_year": returns_by_year,
        "recovered_by_horizon": recovered_by_horizon,
    }


def analyze_records(candidate_records, baseline_records, session_manifest):
    session_manifest = _RELATIVE_ANALYZER._require_validated_session_manifest(
        session_manifest
    )
    evaluation_sessions = tuple(
        session for session in session_manifest.sessions
        if EVALUATION_START <= session <= EVALUATION_END
    )
    candidate_records = tuple(candidate_records)
    baseline_records = tuple(baseline_records)
    errors = []
    _validate_initialization(
        baseline_records, "baseline", BASELINE_BUILD, errors,
    )
    _validate_initialization(
        candidate_records, "candidate", CANDIDATE_BUILD, errors,
        require_atr_fingerprint=True,
    )
    baseline_orders = _filled_order_path(baseline_records)
    candidate_orders = _filled_order_path(candidate_records)
    formal_order_path_exact = (
        len(baseline_orders) == EXPECTED_FILLED_ORDERS
        and len(candidate_orders) == EXPECTED_FILLED_ORDERS
        and baseline_orders == candidate_orders
    )
    baseline_portfolio = _portfolio_path(
        baseline_records, evaluation_sessions, "baseline", errors,
    )
    candidate_portfolio = _portfolio_path(
        candidate_records, evaluation_sessions, "candidate", errors,
    )
    baseline_asset = _final_asset(baseline_records, "baseline", errors)
    candidate_asset = _final_asset(candidate_records, "candidate", errors)
    portfolio_path_exact = (
        len(baseline_portfolio) == EXPECTED_PORTFOLIO_POINTS
        and len(candidate_portfolio) == EXPECTED_PORTFOLIO_POINTS
        and baseline_portfolio == candidate_portfolio
        and baseline_asset is not None
        and candidate_asset is not None
        and math.isclose(baseline_asset, EXPECTED_FINAL_ASSET, abs_tol=0.01)
        and math.isclose(candidate_asset, EXPECTED_FINAL_ASSET, abs_tol=0.01)
    )
    shadow = _audit_shadows(
        candidate_records, baseline_records, evaluation_sessions, errors,
    )
    horizon_summaries = {
        str(horizon): _summarize(shadow["returns_by_horizon"][horizon])
        for horizon in HORIZONS
    }
    yearly = {
        year: _summarize(values)
        for year, values in sorted(shadow["returns_by_year"].items())
    }
    recovered = {}
    for horizon in HORIZONS:
        values = shadow["recovered_by_horizon"][horizon]
        recovered[str(horizon)] = (
            sum(values) / len(values) if values else None
        )
    errors = sorted(set(errors))
    horizon_5 = horizon_summaries["5"]
    gates = {
        "data_quality_complete": not errors,
        "formal_order_path_exact": formal_order_path_exact,
        "portfolio_path_exact": portfolio_path_exact,
        "complete_count_at_least_20": shadow["complete_count"] >= 20,
        "horizon_5_median_positive": (
            horizon_5["median"] is not None and horizon_5["median"] > 0
        ),
        "horizon_5_hit_rate_above_half": (
            horizon_5["hit_rate"] is not None
            and horizon_5["hit_rate"] > 0.5
        ),
    }
    return {
        "session_calendar": dict(
            session_manifest.metadata._asdict(),
            sha256=session_manifest.sha256,
        ),
        "data_quality": {
            "errors": errors,
            "registration_count": shadow["registration_count"],
            "complete_count": shadow["complete_count"],
            "right_censored_count": shadow["right_censored_count"],
        },
        "metrics": {
            "horizons": horizon_summaries,
            "yearly_horizon_5": yearly,
            "entry_recovery_rate_by_horizon": recovered,
            "filled_order_count": len(candidate_orders),
            "portfolio_point_count": len(candidate_portfolio),
            "final_asset": candidate_asset,
        },
        "gates": gates,
        "continue_atr_investigation": all(gates.values()),
    }


def _validate_distinct_paths(args):
    labelled = []
    for label, values in (
            ("baseline log", args.baseline_log),
            ("candidate log", args.candidate_log)):
        for value in values:
            labelled.append((label, value, True))
    labelled.extend((
        ("session calendar", args.session_calendar, True),
        ("output", args.output, False),
    ))
    normalized_seen = {}
    inode_seen = {}
    for label, value, must_exist in labelled:
        path = pathlib.Path(value).expanduser().resolve(strict=must_exist)
        normalized = os.path.normcase(str(path))
        inode = None
        if path.exists():
            stat_result = path.stat()
            inode = (stat_result.st_dev, stat_result.st_ino)
        if normalized in normalized_seen:
            raise ValueError(
                "%s and %s must be distinct physical files"
                % (normalized_seen[normalized], label)
            )
        if inode is not None and inode in inode_seen:
            raise ValueError(
                "%s and %s must be distinct physical files"
                % (inode_seen[inode], label)
            )
        normalized_seen[normalized] = label
        if inode is not None:
            inode_seen[inode] = label
    output = pathlib.Path(args.output).expanduser().resolve(strict=False)
    if not output.parent.is_dir():
        raise ValueError("output parent directory must exist")


def _write_json_atomically(path_value, report):
    output = pathlib.Path(path_value).expanduser().resolve(strict=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % output.name,
        suffix=".tmp",
        dir=str(output.parent),
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                report, stream, ensure_ascii=False, sort_keys=True,
                indent=2, allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(output))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", action="append", required=True)
    parser.add_argument("--candidate-log", action="append", required=True)
    parser.add_argument("--session-calendar", required=True)
    parser.add_argument("--session-calendar-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _argument_parser().parse_args(argv)
    _validate_distinct_paths(args)
    manifest = validate_session_calendar_manifest(
        read_session_calendar_manifest_bytes(args.session_calendar),
        args.session_calendar_sha256,
    )
    report = analyze_records(
        load_log_records(args.candidate_log),
        load_log_records(args.baseline_log),
        manifest,
    )
    _write_json_atomically(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
