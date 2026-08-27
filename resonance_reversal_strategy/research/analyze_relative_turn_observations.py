"""Read-only descriptive analysis for relative-resonance observation logs."""

import argparse
import html
import json
import math
import pathlib
import re
import statistics
from collections import Counter
from datetime import date


TRAIN_START = date(2019, 1, 1)
TRAIN_END = date(2021, 12, 31)
CANDIDATE_BUILD = "20260827.4"
RELATIVE_OBSERVATION_FINGERPRINT = "f47d32b87be6d926"
BASELINE_FILLED_COUNT = 138
BASELINE_FINAL_ASSET = 23856.40
PARAMETER_FINGERPRINT = "e1227fbd8b4a884e"
POOL_FINGERPRINT = "9123995edeb1ed84"
FORMAL_EVENT_FINGERPRINT = "1c0b8a22f48c97c3"
LOG_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def parse_joinquant_log_line(line, ordinal):
    """Return one structured payload from a JoinQuant log line, if present."""
    text = html.unescape(line.strip())
    payload_start = text.find("{")
    if payload_start < 0:
        return None
    try:
        payload = json.loads(text[payload_start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    match = LOG_DATE_RE.match(text)
    payload = dict(payload)
    payload["_log_date"] = match.group(1) if match else None
    payload["_ordinal"] = int(ordinal)
    return payload


def load_log_records(paths):
    """Read user-supplied logs without changing them."""
    records = []
    ordinal = 0
    for path_value in paths:
        path = pathlib.Path(path_value)
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                ordinal += 1
                record = parse_joinquant_log_line(line, ordinal)
                if record is not None:
                    records.append(record)
    return records


def _calendar_date(value):
    if value in (None, ""):
        return None
    return date.fromisoformat(str(value)[:10])


def _is_relative_record(record):
    event = record.get("event")
    return (
        event == "relative_resonance_observation"
        or (event == "observation_outcome" and (
            record.get("relative_observation_id")
            or record.get("observation_kind") == "RELATIVE_RESONANCE"
        ))
    )


def _is_matching_relative_record(record):
    return (
        _is_relative_record(record)
        and record.get("build") == CANDIDATE_BUILD
        and record.get("relative_observation_fingerprint")
        == RELATIVE_OBSERVATION_FINGERPRINT
    )


def reject_nontraining_observations(records):
    """Reject selected relative observations with a date outside 2019--2021."""
    for record in records:
        if not _is_relative_record(record):
            continue
        outcome = record.get("outcome") or {}
        observed_dates = (
            record.get("signal_date"), record.get("event_date"),
            outcome.get("closing_date") if isinstance(outcome, dict) else None,
        )
        normalized_dates = tuple(
            _calendar_date(value) for value in observed_dates
            if value not in (None, "")
        )
        if not normalized_dates:
            raise ValueError("relative observation has no training date")
        for observed_date in normalized_dates:
            if not TRAIN_START <= observed_date <= TRAIN_END:
                raise ValueError(
                    "relative observation outside 2019-2021: %s" % observed_date
                )


def lower_quartile(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * 0.25
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_returns(values):
    values = tuple(float(value) for value in values)
    if not values:
        return {
            "count": 0, "mean": None, "median": None,
            "hit_rate": None, "q1": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "hit_rate": sum(value > 0 for value in values) / len(values),
        "q1": lower_quartile(values),
    }


def extract_filled_order_path(records):
    return tuple(
        (
            record.get("_log_date"), record.get("side"),
            record.get("code"), record.get("before_amount"),
            record.get("after_amount"),
        )
        for record in records
        if (record.get("event") == "order_transition"
            and record.get("outcome") == "FILLED")
    )


def extract_final_asset(records):
    summaries = [
        record for record in records
        if record.get("event") == "portfolio_summary"
        and record.get("total_value") is not None
    ]
    return None if not summaries else _safe_number(summaries[-1]["total_value"])


def _formal_five_day_returns(records):
    directions = {
        record["resonance_id"]: record.get("direction")
        for record in records
        if record.get("event") == "resonance_decision"
        and record.get("accepted") is True
        and record.get("reason") == "COMPLETE_RESONANCE"
        and record.get("resonance_id")
    }
    values = []
    for record in records:
        if (record.get("event") != "observation_outcome"
                or record.get("horizon") != 5):
            continue
        outcome = record.get("outcome") or {}
        resonance_id = record.get("resonance_id")
        raw_return = _safe_number(outcome.get("return"))
        if (resonance_id not in directions
                or outcome.get("status") != "RECORDED"
                or raw_return is None):
            continue
        values.append(
            raw_return if directions[resonance_id] == "BUY_TURN"
            else -raw_return
        )
    return values


def _safe_number(value):
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _sort_key(record):
    return (
        str(record.get("signal_date") or record.get("event_date") or ""),
        str(record.get("relative_observation_id") or ""),
        str(record.get("horizon") or ""),
        int(record.get("_ordinal") or 0),
    )


def _filter_candidate_records(records):
    selected = []
    ignored = Counter()
    for record in records:
        event = record.get("event")
        if event == "strategy_initialized":
            if (record.get("build") == CANDIDATE_BUILD
                    and record.get("relative_observation_fingerprint")
                    == RELATIVE_OBSERVATION_FINGERPRINT):
                selected.append(record)
            else:
                ignored["initialization_mismatch"] += 1
        elif _is_relative_record(record):
            if record.get("build") != CANDIDATE_BUILD:
                ignored["relative_build_mismatch"] += 1
            elif (record.get("relative_observation_fingerprint")
                    != RELATIVE_OBSERVATION_FINGERPRINT):
                ignored["relative_fingerprint_mismatch"] += 1
            else:
                selected.append(record)
        elif event == "observation_outcome":
            ignored["formal_observation_outcome"] += 1
    return selected, dict(sorted(ignored.items()))


def _initialization_errors(records):
    expected = {
        "build": CANDIDATE_BUILD,
        "parameter_fingerprint": PARAMETER_FINGERPRINT,
        "pool_fingerprint": POOL_FINGERPRINT,
        "event_logic_fingerprint": FORMAL_EVENT_FINGERPRINT,
        "relative_observation_fingerprint": RELATIVE_OBSERVATION_FINGERPRINT,
    }
    initialized = [
        record for record in records
        if record.get("event") == "strategy_initialized"
    ]
    if not initialized:
        return ["missing matching strategy_initialized record"]
    errors = []
    for record in initialized:
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                errors.append("%s mismatch: %r" % (field, record.get(field)))
    return errors


def analyze_records(candidate_records, baseline_records):
    """Produce the frozen descriptive report without mutating either input."""
    candidate_records = list(candidate_records)
    baseline_records = list(baseline_records)
    selected, ignored_record_counts = _filter_candidate_records(candidate_records)
    relative_records = [record for record in selected if _is_relative_record(record)]
    reject_nontraining_observations(relative_records)
    errors = _initialization_errors(selected)

    candidates = []
    candidate_by_id = {}
    for record in sorted(relative_records, key=_sort_key):
        if record.get("event") != "relative_resonance_observation":
            continue
        observation_id = record.get("relative_observation_id")
        if not isinstance(observation_id, str) or not observation_id.startswith("RELATIVE:"):
            errors.append("invalid relative observation id: %r" % observation_id)
            continue
        if observation_id in candidate_by_id:
            errors.append("duplicate relative candidate: %s" % observation_id)
            continue
        if record.get("observation_kind") != "RELATIVE_RESONANCE":
            errors.append("invalid observation kind: %s" % observation_id)
        if record.get("branch") not in ("HARD_BOLL_SOFT_OSC", "SOFT_ALL_THREE"):
            errors.append("invalid branch: %s" % observation_id)
        signal_date = _calendar_date(record.get("signal_date"))
        expires_date = _calendar_date(record.get("expires_date"))
        if signal_date is None or expires_date is None:
            errors.append("candidate date missing: %s" % observation_id)
        elif expires_date < signal_date:
            errors.append("expired candidate: %s" % observation_id)
        candidate_by_id[observation_id] = record
        candidates.append(record)

    outcomes = {}
    for record in sorted(relative_records, key=_sort_key):
        if record.get("event") != "observation_outcome":
            continue
        observation_id = record.get("relative_observation_id")
        try:
            horizon = int(record.get("horizon"))
        except (TypeError, ValueError):
            errors.append("invalid relative horizon: %s" % observation_id)
            continue
        key = (observation_id, horizon)
        if key in outcomes:
            errors.append("duplicate relative outcome: %s/%s" % key)
            continue
        outcomes[key] = record

    formal_keys = {
        (record.get("code"), record.get("direction"), str(record.get("signal_date"))[:10])
        for record in candidate_records
        if record.get("event") == "resonance_decision"
        and record.get("accepted") is True
        and record.get("reason") == "COMPLETE_RESONANCE"
    }
    relative_keys = {
        (record.get("code"), record.get("direction"), str(record.get("signal_date"))[:10])
        for record in candidates
    }
    formal_overlap_count = len(formal_keys & relative_keys)

    year_counts = {"2019": 0, "2020": 0, "2021": 0}
    direction_counts = {"BUY_TURN": 0, "SELL_TURN": 0}
    etf_counts = Counter()
    returns_by_horizon = {1: [], 3: [], 5: []}
    five_day_by_branch = {"HARD_BOLL_SOFT_OSC": [], "SOFT_ALL_THREE": []}
    five_day_2021 = []
    positive_by_etf = Counter()
    missing_outcome_count = 0
    for candidate in candidates:
        observation_id = candidate["relative_observation_id"]
        signal_date = _calendar_date(candidate.get("signal_date"))
        if signal_date is not None and str(signal_date.year) in year_counts:
            year_counts[str(signal_date.year)] += 1
        direction = candidate.get("direction")
        if direction in direction_counts:
            direction_counts[direction] += 1
        else:
            errors.append("invalid candidate direction: %s" % observation_id)
        code = candidate.get("code")
        if code:
            etf_counts[code] += 1
        else:
            errors.append("candidate code missing: %s" % observation_id)
        for horizon in (1, 3, 5):
            record = outcomes.get((observation_id, horizon))
            outcome = record.get("outcome") if record is not None else None
            value = _safe_number(
                outcome.get("direction_adjusted_return")
                if isinstance(outcome, dict) else None
            )
            if (not isinstance(outcome, dict)
                    or outcome.get("status") != "RECORDED"
                    or value is None):
                missing_outcome_count += 1
                continue
            returns_by_horizon[horizon].append(value)
            if horizon == 5:
                branch = candidate.get("branch")
                if branch in five_day_by_branch:
                    five_day_by_branch[branch].append(value)
                if signal_date is not None and signal_date.year == 2021:
                    five_day_2021.append(value)
                if value > 0 and code:
                    positive_by_etf[code] += value

    total_positive = sum(positive_by_etf.values())
    max_positive_contribution = (
        max(positive_by_etf.values()) / total_positive
        if total_positive > 0 else None
    )
    formal_horizon_5 = summarize_returns(_formal_five_day_returns(baseline_records))
    horizon_5 = summarize_returns(returns_by_horizon[5])
    year_2021 = summarize_returns(five_day_2021)
    candidate_path = extract_filled_order_path(candidate_records)
    baseline_path = extract_filled_order_path(baseline_records)
    candidate_asset = extract_final_asset(candidate_records)
    baseline_asset = extract_final_asset(baseline_records)
    formal_order_path_exact = (
        len(candidate_path) == BASELINE_FILLED_COUNT
        and len(baseline_path) == BASELINE_FILLED_COUNT
        and candidate_path == baseline_path
    )
    final_asset_exact = (
        candidate_asset is not None and baseline_asset is not None
        and math.isclose(candidate_asset, BASELINE_FINAL_ASSET, abs_tol=0.01)
        and math.isclose(baseline_asset, BASELINE_FINAL_ASSET, abs_tol=0.01)
        and math.isclose(candidate_asset, baseline_asset, abs_tol=0.01)
    )
    errors = sorted(errors)
    data_quality_complete = (
        not errors and formal_overlap_count == 0 and missing_outcome_count == 0
    )
    gates = {
        "candidate_count_at_least_30": len(candidates) >= 30,
        "each_training_year_at_least_5": all(
            year_counts[str(year)] >= 5 for year in (2019, 2020, 2021)
        ),
        "horizon_5_median_positive": (
            horizon_5["median"] is not None and horizon_5["median"] > 0
        ),
        "horizon_5_hit_rate_above_half": (
            horizon_5["hit_rate"] is not None and horizon_5["hit_rate"] > 0.5
        ),
        "year_2021_median_nonnegative": (
            year_2021["median"] is not None and year_2021["median"] >= 0
        ),
        "horizon_5_q1_not_worse_than_formal": (
            horizon_5["q1"] is not None and formal_horizon_5["q1"] is not None
            and horizon_5["q1"] >= formal_horizon_5["q1"]
        ),
        "single_etf_positive_contribution_at_most_half": (
            max_positive_contribution is not None
            and max_positive_contribution <= 0.5
        ),
        "formal_order_path_exact": formal_order_path_exact,
        "final_asset_exact": final_asset_exact,
        "data_quality_complete": data_quality_complete,
    }
    return {
        "data_quality": {
            "errors": errors,
            "relative_fingerprint": RELATIVE_OBSERVATION_FINGERPRINT,
            "formal_overlap_count": formal_overlap_count,
            "missing_outcome_count": missing_outcome_count,
            "ignored_record_counts": ignored_record_counts,
        },
        "metrics": {
            "candidate_count": len(candidates),
            "year_counts": year_counts,
            "direction_counts": direction_counts,
            "etf_counts": dict(sorted(etf_counts.items())),
            "by_branch": {
                branch: summarize_returns(five_day_by_branch[branch])
                for branch in ("HARD_BOLL_SOFT_OSC", "SOFT_ALL_THREE")
            },
            "horizon_1": summarize_returns(returns_by_horizon[1]),
            "horizon_3": summarize_returns(returns_by_horizon[3]),
            "horizon_5": horizon_5,
            "year_2021_horizon_5": year_2021,
            "formal_horizon_5": formal_horizon_5,
            "max_positive_contribution_by_etf": max_positive_contribution,
            "filled_path_count": len(candidate_path),
            "final_asset": candidate_asset,
        },
        "gates": gates,
        "continue_candidate": all(gates.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-log", action="append", required=True)
    parser.add_argument("--baseline-log", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = analyze_records(
        load_log_records(args.candidate_log),
        load_log_records(args.baseline_log),
    )
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
