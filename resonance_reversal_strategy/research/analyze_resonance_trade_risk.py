"""Read-only trade-path risk analysis for resonance build 20260828.5."""

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import html
import importlib.util
import json
import math
import os
import pathlib
import re
import statistics
import tempfile


EXPECTED_BUILD = "20260828.5"
EXPECTED_ATR_POLICY = "OBSERVE_ONLY"
EXPECTED_RELATIVE_BUY_POLICY = "EMPTY_SLOT_BACKFILL"
EXPECTED_PARAMETER_FINGERPRINT = "e1227fbd8b4a884e"
EXPECTED_POOL_FINGERPRINT = "9123995edeb1ed84"
EXPECTED_EVENT_FINGERPRINT = "1c0b8a22f48c97c3"
EXPECTED_RELATIVE_FINGERPRINT = "f47d32b87be6d926"
NON_RECOVERY_COMPLETED_SESSIONS = 20
ORDINARY_COMMISSION_RATE = 0.0003
DOUBLE_FRICTION_COMMISSION_RATE = 0.0006
MINIMUM_COMMISSION = 5.0
RELATIVE_BRANCHES = frozenset((
    "HARD_BOLL_SOFT_OSC", "SOFT_ALL_THREE",
))
RELATIVE_DIRECTIONS = frozenset(("BUY_TURN", "SELL_TURN"))
RELATIVE_HORIZONS = frozenset((1, 3, 5))
RELATIVE_INDICATORS = frozenset(("BOLL", "KDJ", "RSI"))
RELATIVE_SOURCES = frozenset(("HARD", "RELATIVE"))
EVALUATION_START = date(2019, 1, 1)
EVALUATION_END = date(2021, 12, 31)

TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+-"
)
FILL_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"security=(?P<code>\S+).*action=(?P<action>open|close).*"
    r"trade price:\s*(?P<price>[0-9.]+),\s*"
    r"amount:\s*(?P<amount>\d+),\s*"
    r"commission:\s*(?P<commission>[0-9.]+)"
)
FORMAL_SORTED_BUY_RE = re.compile(
    r"^BUY_CANDIDATE_SORTED:(?P<rank>[1-9][0-9]*)$"
)
RELATIVE_SORTED_BUY_RE = re.compile(
    r"^RELATIVE_BUY_CANDIDATE_SORTED:(?P<rank>[1-9][0-9]*)$"
)


def _load_manifest_api():
    path = pathlib.Path(__file__).with_name(
        "analyze_relative_turn_observations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_resonance_risk_manifest_api", path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.read_session_calendar_manifest_bytes,
        module.validate_session_calendar_manifest,
    )


(
    read_session_calendar_manifest_bytes,
    validate_session_calendar_manifest,
) = _load_manifest_api()


@dataclass(frozen=True)
class Fill:
    timestamp: datetime
    code: str
    side: str
    price: float
    amount: int
    commission: float
    source_path: str
    ordinal: int

    @property
    def trade_date(self):
        return self.timestamp.date()


@dataclass(frozen=True)
class PortfolioPoint:
    closing_date: date
    total_value: float
    available_cash: float
    positions: tuple


@dataclass(frozen=True)
class ParsedLog:
    fills: tuple
    portfolio_points: tuple
    records: tuple


@dataclass(frozen=True)
class CompletedTrade:
    code: str
    buy: Fill
    sell: Fill
    pnl: float
    return_rate: float
    amount_ratio: float


def _reject_duplicate_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _finite_number(value, label, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be finite" % label)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % label)
    if positive and result <= 0:
        raise ValueError("%s must be positive" % label)
    if nonnegative and result < 0:
        raise ValueError("%s must be nonnegative" % label)
    return result


def _parse_fill(line, source_path, ordinal):
    match = FILL_RE.search(html.unescape(line))
    if match is None:
        return None
    timestamp = datetime.strptime(
        match.group("timestamp"), "%Y-%m-%d %H:%M:%S",
    )
    price = float(match.group("price"))
    amount = int(match.group("amount"))
    commission = float(match.group("commission"))
    if not math.isfinite(price) or price <= 0:
        raise ValueError("fill price must be finite and positive")
    if amount <= 0:
        raise ValueError("fill amount must be positive")
    if not math.isfinite(commission) or commission < 0:
        raise ValueError("fill commission must be finite and nonnegative")
    return Fill(
        timestamp=timestamp,
        code=match.group("code"),
        side="BUY" if match.group("action") == "open" else "SELL",
        price=price,
        amount=amount,
        commission=commission,
        source_path=str(source_path),
        ordinal=ordinal,
    )


def _parse_structured_record(line, source_path, ordinal):
    text = html.unescape(line.strip())
    payload_start = text.find("{")
    if payload_start < 0 or '"event"' not in text[payload_start:]:
        return None
    timestamp_match = TIMESTAMP_RE.match(text)
    if timestamp_match is None:
        raise ValueError("structured event requires emitter timestamp")
    try:
        payload = json.loads(
            text[payload_start:],
            object_pairs_hook=_reject_duplicate_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant: %s" % value)
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid structured JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("structured event must be an object")
    record = dict(payload)
    record["_timestamp"] = datetime.strptime(
        timestamp_match.group("timestamp"), "%Y-%m-%d %H:%M:%S",
    )
    record["_source_path"] = str(source_path)
    record["_ordinal"] = ordinal
    return record


def _parse_positions(value):
    if not isinstance(value, dict):
        raise ValueError("portfolio positions must be an object")
    positions = []
    for code, amount in value.items():
        if not isinstance(code, str) or not code:
            raise ValueError("portfolio position code must be nonempty")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("portfolio position amount must be positive integer")
        positions.append((code, amount))
    return tuple(sorted(positions))


def _parse_portfolio(record):
    value = record.get("closing_date")
    if not isinstance(value, str):
        raise ValueError("portfolio closing_date must be ISO date")
    try:
        closing_date = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("portfolio closing_date must be ISO date") from exc
    return PortfolioPoint(
        closing_date=closing_date,
        total_value=_finite_number(
            record.get("total_value"), "portfolio total_value", positive=True,
        ),
        available_cash=_finite_number(
            record.get("available_cash"), "portfolio available_cash",
        ),
        positions=_parse_positions(record.get("positions")),
    )


def parse_joinquant_log(paths):
    normalized = sorted(
        pathlib.Path(value).expanduser().resolve(strict=True)
        for value in paths
    )
    fills = []
    records = []
    portfolio_points = []
    ordinal = 0
    for path in normalized:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                ordinal += 1
                fill = _parse_fill(line, path, ordinal)
                if fill is not None:
                    fills.append(fill)
                record = _parse_structured_record(line, path, ordinal)
                if record is None:
                    continue
                records.append(record)
                if record.get("event") == "portfolio_summary":
                    portfolio_points.append(_parse_portfolio(record))
    for previous, current in zip(fills, fills[1:]):
        if current.timestamp < previous.timestamp:
            raise ValueError("fill timestamps must be nondecreasing")
    for previous, current in zip(portfolio_points, portfolio_points[1:]):
        if current.closing_date <= previous.closing_date:
            raise ValueError("portfolio dates must be strictly increasing")
    for previous, current in zip(records, records[1:]):
        if current["_timestamp"] < previous["_timestamp"]:
            raise ValueError("structured event timestamps must be nondecreasing")
    return ParsedLog(
        fills=tuple(fills),
        portfolio_points=tuple(portfolio_points),
        records=tuple(records),
    )


def _require_identity(parsed_log, role):
    records = [
        record for record in parsed_log.records
        if record.get("event") == "strategy_initialized"
    ]
    if len(records) != 1:
        raise ValueError("%s initialization count must be one" % role)
    record = records[0]
    expected = {
        "build": EXPECTED_BUILD,
        "atr_exit_policy": EXPECTED_ATR_POLICY,
        "relative_buy_policy": EXPECTED_RELATIVE_BUY_POLICY,
        "parameter_fingerprint": EXPECTED_PARAMETER_FINGERPRINT,
        "pool_fingerprint": EXPECTED_POOL_FINGERPRINT,
        "event_logic_fingerprint": EXPECTED_EVENT_FINGERPRINT,
        "relative_observation_fingerprint": EXPECTED_RELATIVE_FINGERPRINT,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError("%s %s mismatch" % (role, field))
    return record


def _evaluation_sessions(manifest):
    return tuple(
        session for session in manifest.sessions
        if EVALUATION_START <= session <= EVALUATION_END
    )


def _validate_training_boundary(parsed_log, manifest, role):
    evaluation_sessions = _evaluation_sessions(manifest)
    if not evaluation_sessions:
        raise ValueError("manifest has no training sessions")
    portfolio_sessions = tuple(
        point.closing_date for point in parsed_log.portfolio_points
    )
    if portfolio_sessions != evaluation_sessions:
        outside = [
            session for session in portfolio_sessions
            if session not in evaluation_sessions
        ]
        if outside:
            raise ValueError("%s portfolio outside training window" % role)
        raise ValueError("%s portfolio sessions differ from manifest" % role)
    allowed = set(evaluation_sessions)
    manifest_sessions = set(manifest.sessions)
    for fill in parsed_log.fills:
        if fill.trade_date not in allowed:
            raise ValueError("%s fill outside training window" % role)
    for record in parsed_log.records:
        event = record.get("event")
        if event == "strategy_initialized":
            timestamp = record.get("_timestamp")
            if not EVALUATION_START <= timestamp.date() <= EVALUATION_END:
                raise ValueError(
                    "%s initialization outside training window" % role
                )
            continue
        timestamp = record.get("_timestamp")
        if timestamp.date() not in allowed:
            raise ValueError("%s event outside training window" % role)
        if event == "observation_outcome":
            outcome = record.get("outcome")
            if not isinstance(outcome, dict):
                raise ValueError("%s observation outcome is invalid" % role)
            closing_value = outcome.get("closing_date")
            if outcome.get("status") == "RECORDED" and closing_value is None:
                raise ValueError(
                    "%s observation outcome closing_date is invalid" % role
                )
            if closing_value is not None:
                try:
                    closing_date = date.fromisoformat(closing_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "%s observation outcome closing_date is invalid"
                        % role
                    ) from exc
                if closing_date not in allowed:
                    raise ValueError(
                        "%s observation outcome outside training window"
                        % role
                    )
        for field in (
                "decision_date", "signal_date", "event_date",
                "expires_date", "closing_date"):
            value = record.get(field)
            if value is None:
                continue
            try:
                structured_date = date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "%s %s is invalid" % (role, field)
                ) from exc
            permitted = (
                allowed if field in ("decision_date", "closing_date")
                else manifest_sessions
            )
            if structured_date not in permitted:
                raise ValueError(
                    "%s %s outside manifest coverage" % (role, field)
                )
        supporter_dates = record.get("supporter_event_dates")
        if supporter_dates is not None:
            if not isinstance(supporter_dates, dict):
                raise ValueError("%s supporter_event_dates is invalid" % role)
            for value in supporter_dates.values():
                try:
                    supporter_date = date.fromisoformat(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "%s supporter_event_dates is invalid" % role
                    ) from exc
                if supporter_date not in manifest_sessions:
                    raise ValueError(
                        "%s supporter_event_dates outside manifest coverage"
                        % role
                    )


def _filled_order_transitions(records, fills, role):
    transitions = []
    for record in records:
        if (record.get("event") != "order_transition"
                or record.get("outcome") != "FILLED"):
            continue
        side = record.get("side")
        code = record.get("code")
        before = record.get("before_amount")
        after = record.get("after_amount")
        if (side not in ("BUY", "SELL")
                or not isinstance(code, str) or not code
                or isinstance(before, bool) or not isinstance(before, int)
                or isinstance(after, bool) or not isinstance(after, int)
                or before < 0 or after < 0):
            raise ValueError("%s filled order transition is invalid" % role)
        transitions.append(record)
    transition_path = tuple(
        (record["_timestamp"], record["side"], record["code"])
        for record in transitions
    )
    fill_path = tuple(
        (fill.timestamp, fill.side, fill.code) for fill in fills
    )
    if transition_path != fill_path:
        raise ValueError("%s fill and order paths differ" % role)
    for fill, record in zip(fills, transitions):
        changed_amount = abs(
            record["after_amount"] - record["before_amount"]
        )
        if changed_amount != fill.amount:
            raise ValueError("%s fill and order amounts differ" % role)
        if fill.side == "BUY" and not (
                record["before_amount"] == 0
                and record["after_amount"] > 0):
            raise ValueError("%s filled buy transition is invalid" % role)
        if fill.side == "SELL" and not (
                record["before_amount"] > 0
                and record["after_amount"] == 0):
            raise ValueError("%s filled sell transition is invalid" % role)
    return tuple(transitions)


def _reconcile_portfolio_ledger(records, role):
    active = {}
    for record in records:
        event = record.get("event")
        if event == "order_transition" and record.get("outcome") == "FILLED":
            code = record.get("code")
            before = record.get("before_amount")
            after = record.get("after_amount")
            if active.get(code, 0) != before:
                raise ValueError("%s order before amount mismatch" % role)
            if after == 0:
                active.pop(code, None)
            else:
                active[code] = after
            continue
        if event != "portfolio_summary":
            continue
        positions = dict(_parse_positions(record.get("positions")))
        if set(positions) != set(active):
            raise ValueError("%s portfolio position set mismatch" % role)
        active = positions


def _pair_completed_trades(fills):
    open_by_code = {}
    completed = []
    for fill in fills:
        if fill.side == "BUY":
            if fill.code in open_by_code:
                raise ValueError("duplicate open for %s" % fill.code)
            open_by_code[fill.code] = fill
            continue
        buy = open_by_code.pop(fill.code, None)
        if buy is None:
            raise ValueError("sell without open for %s" % fill.code)
        buy_cost = buy.price * buy.amount + buy.commission
        sell_proceeds = fill.price * fill.amount - fill.commission
        pnl = sell_proceeds - buy_cost
        completed.append(CompletedTrade(
            code=fill.code,
            buy=buy,
            sell=fill,
            pnl=pnl,
            return_rate=pnl / buy_cost,
            amount_ratio=float(fill.amount) / buy.amount,
        ))
    return tuple(completed), tuple(
        open_by_code[code] for code in sorted(open_by_code)
    )


def _fill_identity(fill):
    return fill.trade_date.isoformat(), fill.side, fill.code


def _reconcile_paths(ordinary, double_friction):
    ordinary_path = tuple(_fill_identity(fill) for fill in ordinary.fills)
    friction_path = tuple(
        _fill_identity(fill) for fill in double_friction.fills
    )
    if ordinary_path != friction_path:
        raise ValueError("friction fill path differs from ordinary")
    return {
        "fill_count": len(ordinary_path),
        "identity_match": True,
        "amount_difference_count": sum(
            left.amount != right.amount
            for left, right in zip(ordinary.fills, double_friction.fills)
        ),
    }


def _relative_observation_id(record):
    parts = [
        "RELATIVE", record["branch"], record["direction"], record["code"],
    ]
    sources = record["hard_or_relative_source_by_indicator"]
    dates = record["supporter_event_dates"]
    for indicator in sorted(record["supporters"]):
        parts.append("%s:%s:%s" % (
            indicator, sources[indicator], dates[indicator],
        ))
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return "RELATIVE:" + digest[:20]


def _validate_relative_registration(record, role, session_calendar):
    expected = {
        "build": EXPECTED_BUILD,
        "observation_kind": "RELATIVE_RESONANCE",
        "parameter_fingerprint": EXPECTED_PARAMETER_FINGERPRINT,
        "pool_fingerprint": EXPECTED_POOL_FINGERPRINT,
        "event_logic_fingerprint": EXPECTED_EVENT_FINGERPRINT,
        "relative_observation_fingerprint": EXPECTED_RELATIVE_FINGERPRINT,
    }
    if any(record.get(field) != value for field, value in expected.items()):
        raise ValueError("%s relative observation metadata is invalid" % role)
    observation_id = record.get("relative_observation_id")
    code = record.get("code")
    direction = record.get("direction")
    branch = record.get("branch")
    supporters = record.get("supporters")
    dates = record.get("supporter_event_dates")
    sources = record.get("hard_or_relative_source_by_indicator")
    if (not isinstance(observation_id, str)
            or not observation_id.startswith("RELATIVE:")
            or not isinstance(code, str) or not code
            or direction not in RELATIVE_DIRECTIONS
            or branch not in RELATIVE_BRANCHES
            or not isinstance(supporters, list) or not supporters
            or any(item not in RELATIVE_INDICATORS for item in supporters)
            or len(set(supporters)) != len(supporters)
            or not isinstance(dates, dict)
            or set(dates) != set(supporters)
            or not isinstance(sources, dict)
            or set(sources) != set(supporters)
            or any(value not in RELATIVE_SOURCES
                   for value in sources.values())):
        raise ValueError("%s relative observation metadata is invalid" % role)
    try:
        signal_date = date.fromisoformat(record["signal_date"])
        expires_date = date.fromisoformat(record["expires_date"])
        supporter_dates = {
            indicator: date.fromisoformat(dates[indicator])
            for indicator in supporters
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "%s relative observation metadata is invalid" % role
        ) from exc
    if (expires_date < signal_date
            or any(value > signal_date for value in supporter_dates.values())
            or signal_date not in supporter_dates.values()):
        raise ValueError("%s relative observation metadata is invalid" % role)
    timestamp = record.get("_timestamp")
    if (not isinstance(timestamp, datetime)
            or timestamp.time().hour < 9
            or (timestamp.time().hour == 9 and timestamp.time().minute < 35)):
        raise ValueError(
            "%s relative observation log timestamp is invalid" % role
        )
    if timestamp.date() not in session_calendar:
        raise ValueError(
            "%s relative observation decision date absent from manifest"
            % role
        )
    decision_index = session_calendar.index(timestamp.date())
    if (decision_index == 0
            or session_calendar[decision_index - 1] != signal_date):
        raise ValueError(
            "%s relative observation signal date is not previous manifest "
            "session" % role
        )
    valid_support_contracts = {
        "HARD_BOLL_SOFT_OSC": (
            ({"BOLL", "RSI"}, {"BOLL": "HARD", "RSI": "RELATIVE"}),
            ({"BOLL", "KDJ"}, {"BOLL": "HARD", "KDJ": "RELATIVE"}),
            ({"BOLL", "KDJ", "RSI"}, {
                "BOLL": "HARD", "KDJ": "RELATIVE", "RSI": "RELATIVE",
            }),
        ),
        "SOFT_ALL_THREE": (({"BOLL", "KDJ", "RSI"}, {
            "BOLL": "RELATIVE", "KDJ": "RELATIVE", "RSI": "RELATIVE",
        }),),
    }
    if not any(
            set(supporters) == expected_supporters
            and sources == expected_sources
            for expected_supporters, expected_sources
            in valid_support_contracts[branch]):
        raise ValueError("%s relative observation metadata is invalid" % role)
    signal_index = session_calendar.index(signal_date)
    if all(value == signal_date for value in supporter_dates.values()):
        expected_expiry = (
            session_calendar[signal_index + 1]
            if signal_index + 1 < len(session_calendar) else None
        )
    else:
        previous_signal = (
            session_calendar[signal_index - 1] if signal_index > 0 else None
        )
        expected_expiry = (
            signal_date
            if previous_signal is not None
            and set(supporter_dates.values()) == {
                previous_signal, signal_date,
            }
            else None
        )
    if expires_date != expected_expiry:
        raise ValueError("%s relative observation expiry mismatch" % role)
    _finite_number(
        record.get("event_close"), "relative observation event_close",
        positive=True,
    )
    if _relative_observation_id(record) != observation_id:
        raise ValueError("%s relative observation id digest mismatch" % role)


def _validate_relative_outcome(
        record, registration, role, session_calendar):
    observation_id = record.get("relative_observation_id")
    horizon = record.get("horizon")
    if (record.get("build") != EXPECTED_BUILD
            or record.get("observation_kind") != "RELATIVE_RESONANCE"
            or record.get("relative_observation_fingerprint")
            != EXPECTED_RELATIVE_FINGERPRINT
            or record.get("resonance_id") != observation_id
            or isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon not in RELATIVE_HORIZONS):
        raise ValueError("%s relative outcome metadata is invalid" % role)
    for field in ("code", "direction", "branch"):
        if record.get(field) != registration.get(field):
            raise ValueError("%s relative outcome identity mismatch" % role)
    if (record.get("event_date") != registration.get("signal_date")
            or record.get("supporters") != registration.get("supporters")):
        raise ValueError("%s relative outcome identity mismatch" % role)
    for field in (
            "supporter_event_dates",
            "hard_or_relative_source_by_indicator",
    ):
        if field in record and record.get(field) != registration.get(field):
            raise ValueError("%s relative outcome identity mismatch" % role)
    payload = record.get("outcome")
    if not isinstance(payload, dict):
        raise ValueError("%s relative outcome payload is invalid" % role)
    try:
        event_date = date.fromisoformat(record["event_date"])
        closing_date = date.fromisoformat(payload["closing_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "%s relative outcome closing session is invalid" % role
        ) from exc
    if event_date not in session_calendar:
        raise ValueError("%s relative outcome event date absent from manifest" % role)
    event_index = session_calendar.index(event_date)
    expected_index = event_index + horizon
    expected_closing = (
        session_calendar[expected_index]
        if expected_index < len(session_calendar) else None
    )
    if closing_date != expected_closing:
        raise ValueError("%s relative outcome closing session mismatch" % role)
    status = payload.get("status")
    if status not in (
            "RECORDED", "HORIZON_MISSED", "PRICE_UNAVAILABLE"):
        raise ValueError("%s relative outcome payload is invalid" % role)
    if status != "RECORDED":
        if (payload.get("closing_price") is not None
                or payload.get("return") is not None
                or payload.get("direction_adjusted_return") is not None):
            raise ValueError(
                "%s relative outcome terminal payload is invalid" % role
            )
        return
    timestamp = record.get("_timestamp")
    if (not isinstance(timestamp, datetime)
            or timestamp.date() != closing_date):
        raise ValueError("%s relative outcome log date mismatch" % role)
    if (timestamp.time().hour < 15
            or (timestamp.time().hour == 15
                and timestamp.time().minute < 30)):
        raise ValueError("%s relative outcome log timestamp is invalid" % role)
    closing_price = _finite_number(
        payload.get("closing_price"), "relative outcome closing_price",
        positive=True,
    )
    raw_return = _finite_number(
        payload.get("return"), "relative outcome return",
    )
    adjusted_return = _finite_number(
        payload.get("direction_adjusted_return"),
        "relative outcome direction_adjusted_return",
    )
    event_close = _finite_number(
        registration.get("event_close"), "relative observation event_close",
        positive=True,
    )
    expected_raw = closing_price / event_close - 1.0
    expected_adjusted = (
        expected_raw
        if registration.get("direction") == "BUY_TURN"
        else -expected_raw
    )
    if (not math.isclose(raw_return, expected_raw, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(
                adjusted_return, expected_adjusted,
                rel_tol=0.0, abs_tol=1e-12,
            )):
        raise ValueError("%s relative outcome return mismatch" % role)


def _relative_records(records, role, session_calendar):
    registrations = {}
    outcomes = {}
    for record in records:
        event = record.get("event")
        if event == "relative_resonance_observation":
            observation_id = record.get("relative_observation_id")
            if observation_id in registrations:
                raise ValueError("duplicate relative observation id")
            _validate_relative_registration(record, role, session_calendar)
            registrations[observation_id] = record
            continue
        if event != "observation_outcome":
            continue
        observation_id = record.get("relative_observation_id")
        resonance_id = record.get("resonance_id")
        if not (
                isinstance(observation_id, str)
                and observation_id.startswith("RELATIVE:")):
            if isinstance(resonance_id, str) and resonance_id.startswith(
                    "RELATIVE:"):
                raise ValueError("%s relative outcome metadata is invalid" % role)
            continue
        horizon = record.get("horizon")
        key = (observation_id, horizon)
        if key in outcomes:
            raise ValueError("duplicate relative observation outcome")
        outcomes[key] = record
    for (observation_id, _), record in outcomes.items():
        registration = registrations.get(observation_id)
        if registration is None:
            raise ValueError("%s orphan relative outcome" % role)
        _validate_relative_outcome(
            record, registration, role, session_calendar,
        )
    return registrations, outcomes


def _entry_identities(records, relative_metadata):
    candidates = {}
    for record in records:
        event = record.get("event")
        if event == "relative_resonance_observation":
            continue
        if event != "resonance_decision":
            continue
        reason = record.get("reason") or ""
        formal_match = FORMAL_SORTED_BUY_RE.fullmatch(reason)
        relative_match = RELATIVE_SORTED_BUY_RE.fullmatch(reason)
        if formal_match is not None:
            source = "FORMAL"
            rank_match = formal_match
        elif relative_match is not None:
            source = "RELATIVE"
            rank_match = relative_match
        elif (reason.startswith("BUY_CANDIDATE_SORTED:")
              or reason.startswith("RELATIVE_BUY_CANDIDATE_SORTED:")):
            raise ValueError("sorted buy rank is invalid")
        else:
            continue
        entry_rank = int(rank_match.group("rank"))
        if (record.get("accepted") is not True
                or record.get("direction") != "BUY_TURN"
                or entry_rank <= 0
                or not isinstance(record.get("code"), str)
                or not record.get("code")
                or not isinstance(record.get("resonance_id"), str)
                or not isinstance(record.get("supporters"), list)
                or not record.get("supporters")):
            raise ValueError("sorted buy decision is invalid")
        is_relative_id = record["resonance_id"].startswith("RELATIVE:")
        if (source == "RELATIVE") != is_relative_id:
            raise ValueError("sorted buy decision namespace mismatch")
        key = (record["_timestamp"].date(), record.get("code"))
        if key in candidates:
            raise ValueError("ambiguous buy source for %s/%s" % key)
        candidates[key] = (source, entry_rank, record)
    return candidates


def _trade_sources(parsed_log, relative_metadata):
    candidates = _entry_identities(parsed_log.records, relative_metadata)
    sources = {}
    for fill in parsed_log.fills:
        if fill.side != "BUY":
            continue
        key = (fill.trade_date, fill.code)
        candidate = candidates.get(key)
        if candidate is None:
            raise ValueError(
                "filled buy has no unique sorted candidate: %s/%s" % key
            )
        source, entry_rank, record = candidate
        timestamp = record.get("_timestamp")
        if (not isinstance(timestamp, datetime)
                or timestamp != fill.timestamp
                or (timestamp.hour, timestamp.minute, timestamp.second)
                != (9, 35, 0)
                or record.get("_source_path") != fill.source_path):
            raise ValueError("sorted buy decision time is invalid")
        if record.get("_ordinal") >= fill.ordinal:
            raise ValueError("sorted buy decision must precede fill")
        resonance_id = record.get("resonance_id")
        branch = None
        if source == "RELATIVE":
            observation = relative_metadata.get(resonance_id)
            if observation is None:
                raise ValueError("relative filled buy lacks observation metadata")
            if (observation.get("code") != fill.code
                    or observation.get("direction") != "BUY_TURN"
                    or observation.get("signal_date")
                    != record.get("signal_date")
                    or set(observation.get("supporters") or ())
                    != set(record.get("supporters") or ())):
                raise ValueError("relative buy metadata mismatch")
            branch = observation.get("branch")
        sources[key] = {
            "entry_source": source,
            "entry_branch": branch,
            "entry_rank": entry_rank,
            "resonance_id": resonance_id,
            "signal_date": record.get("signal_date"),
            "supporters": tuple(record.get("supporters") or ()),
        }
    return sources


def _position_amounts_by_date(points):
    return {
        point.closing_date: dict(point.positions)
        for point in points
    }


def _atr_observations_by_code(records):
    result = {}
    for record in records:
        if record.get("event") != "atr_check":
            continue
        timestamp = record.get("_timestamp")
        if (not isinstance(timestamp, datetime)
                or (timestamp.hour, timestamp.minute, timestamp.second)
                != (9, 35, 0)
                or record.get("execution_policy") != EXPECTED_ATR_POLICY
                or record.get("order_submitted") is not False):
            raise ValueError("atr_check identity is invalid")
        code = record.get("code")
        price = _finite_number(
            record.get("current_price"), "atr_check current_price",
            positive=True,
        )
        key = (record["_timestamp"].date(), code)
        if key in result:
            raise ValueError("duplicate atr_check mark")
        anchor = record.get("highest_close_anchor")
        if anchor is not None:
            anchor = _finite_number(
                anchor, "atr_check highest_close_anchor", positive=True,
            )
        result[key] = {
            "current_price": price,
            "highest_close_anchor": anchor,
        }
    return result


def _longest_negative_run(values):
    longest = 0
    current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _trade_report(
        trade, source, marks, amounts, sell_registrations,
        evaluation_sessions):
    buy_cost = trade.buy.price * trade.buy.amount + trade.buy.commission
    path_by_date = {trade.buy.trade_date: 0.0}
    held_sessions = [
        session for session in evaluation_sessions
        if trade.buy.trade_date < session < trade.sell.trade_date
    ]
    for mark_date in held_sessions:
        price = marks.get((mark_date, trade.code))
        if price is None:
            raise ValueError(
                "missing atr_check mark: %s/%s"
                % (mark_date, trade.code)
            )
        amount = amounts.get(mark_date, {}).get(trade.code)
        if amount is None:
            raise ValueError(
                "atr_check mark lacks position amount: %s/%s"
                % (mark_date, trade.code)
            )
        path_by_date[mark_date] = (price * amount - buy_cost) / buy_cost
    path_by_date[trade.sell.trade_date] = trade.return_rate
    path_returns = [
        path_by_date[session] for session in sorted(path_by_date)
    ]
    mfe = max(path_returns)
    mae = min(path_returns)
    sell_observations = [
        record for record in sell_registrations.values()
        if record.get("code") == trade.code
        and trade.buy.trade_date <= record["_timestamp"].date()
        <= trade.sell.trade_date
    ]
    return {
        "code": trade.code,
        "entry_date": trade.buy.trade_date.isoformat(),
        "exit_date": trade.sell.trade_date.isoformat(),
        "entry_source": source["entry_source"],
        "entry_branch": source["entry_branch"],
        "supporters": list(source["supporters"]),
        "pnl": trade.pnl,
        "return_rate": trade.return_rate,
        "amount_ratio": trade.amount_ratio,
        "mfe": mfe,
        "mae": mae,
        "max_profit_giveback": max(0.0, mfe - trade.return_rate),
        "longest_underwater_sessions": _longest_negative_run(path_returns),
        "relative_sell_observation_count": len(sell_observations),
        "first_relative_sell_date": (
            min(record["_timestamp"].date() for record in sell_observations)
            .isoformat() if sell_observations else None
        ),
    }


def _safe_rate(numerator, denominator):
    return float(numerator) / denominator if denominator else None


def _group_summary(trades, field):
    grouped = {}
    for trade in trades:
        key = trade.get(field) or "NONE"
        grouped.setdefault(key, []).append(trade)
    return {
        key: {
            "count": len(values),
            "wins": sum(value["pnl"] > 0 for value in values),
            "win_rate": _safe_rate(
                sum(value["pnl"] > 0 for value in values), len(values),
            ),
            "pnl": sum(value["pnl"] for value in values),
            "median_mae": statistics.median(
                value["mae"] for value in values
            ),
            "median_mfe": statistics.median(
                value["mfe"] for value in values
            ),
            "median_profit_giveback": statistics.median(
                value["max_profit_giveback"] for value in values
            ),
        }
        for key, values in sorted(grouped.items())
    }


ENTRY_QUALITY_CATEGORICAL_FIELDS = (
    "entry_source", "entry_branch", "supporters", "entry_rank",
    "entry_market_state", "code",
)
ENTRY_QUALITY_CONTINUOUS_FIELDS = (
    "rsi14", "adx14", "atr_to_close", "boll_width", "volume_ratio",
    "normalized_boll_mid_slope",
)
VOLUME_RATIO_DELTA_ZERO_ABS_TOLERANCE = 1e-12


def _entry_snapshot_features(record, buy_date, code, session_calendar, role):
    timestamp = record.get("_timestamp")
    if (not isinstance(timestamp, datetime)
            or (timestamp.hour, timestamp.minute, timestamp.second)
            != (9, 35, 0)
            or timestamp.date() != buy_date
            or record.get("decision_date") != buy_date.isoformat()
            or record.get("code") != code
            or record.get("valid") is not True):
        raise ValueError("%s entry signal snapshot identity is invalid" % role)
    try:
        decision_index = session_calendar.index(buy_date)
    except ValueError as exc:
        raise ValueError("%s buy date absent from manifest" % role) from exc
    if (decision_index == 0
            or record.get("signal_date")
            != session_calendar[decision_index - 1].isoformat()):
        raise ValueError("%s signal snapshot is not T-1" % role)
    expected_identity = {
        "build": EXPECTED_BUILD,
        "parameter_fingerprint": EXPECTED_PARAMETER_FINGERPRINT,
        "pool_fingerprint": EXPECTED_POOL_FINGERPRINT,
        "event_logic_fingerprint": EXPECTED_EVENT_FINGERPRINT,
        "relative_observation_fingerprint": EXPECTED_RELATIVE_FINGERPRINT,
    }
    if any(record.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("%s entry signal snapshot metadata is invalid" % role)
    trade_values = record.get("trade_values")
    observation_values = record.get("observation_values")
    trace = record.get("event_detection_trace")
    if (not isinstance(trade_values, dict)
            or not isinstance(observation_values, dict)
            or not isinstance(trace, dict)):
        raise ValueError("%s entry signal snapshot values are invalid" % role)
    boll_trace = trace.get("boll")
    current_boll = boll_trace.get("current") if isinstance(boll_trace, dict) else None
    if not isinstance(current_boll, dict):
        raise ValueError("%s entry signal snapshot close is invalid" % role)
    close = _finite_number(
        current_boll.get("close"), "entry signal close", positive=True,
    )
    atr14 = _finite_number(
        trade_values.get("atr14"), "entry signal atr14", positive=True,
    )
    boll_mid_slope = _finite_number(
        observation_values.get("boll_mid_slope"),
        "entry signal boll_mid_slope",
    )
    return {
        "rsi14": _finite_number(
            trade_values.get("rsi14"), "entry signal rsi14",
        ),
        "adx14": _finite_number(
            observation_values.get("adx14"), "entry signal adx14",
            nonnegative=True,
        ),
        "atr_to_close": atr14 / close,
        "boll_width": _finite_number(
            observation_values.get("boll_width"),
            "entry signal boll_width", nonnegative=True,
        ),
        "volume_ratio": _finite_number(
            observation_values.get("volume_ratio"),
            "entry signal volume_ratio", nonnegative=True,
        ),
        "normalized_boll_mid_slope": boll_mid_slope / close,
    }


def _entry_quality_rows(
        records, completed, sources, session_calendar, role):
    identities = {
        (trade.buy.trade_date, trade.code) for trade in completed
    }
    snapshots = {}
    for record in records:
        if record.get("event") != "signal_snapshot":
            continue
        timestamp = record.get("_timestamp")
        code = record.get("code")
        if not isinstance(timestamp, datetime) or not isinstance(code, str):
            continue
        key = (timestamp.date(), code)
        if key not in identities:
            continue
        if key in snapshots:
            raise ValueError("%s duplicate signal snapshot" % role)
        snapshots[key] = record
    rows = []
    for trade in completed:
        identity = (trade.buy.trade_date, trade.code)
        snapshot = snapshots.get(identity)
        if snapshot is None:
            raise ValueError(
                "%s filled buy lacks unique signal snapshot" % role
            )
        source = sources[identity]
        features = _entry_snapshot_features(
            snapshot, trade.buy.trade_date, trade.code,
            session_calendar, role,
        )
        if source["signal_date"] != snapshot.get("signal_date"):
            raise ValueError(
                "%s sorted buy and signal snapshot dates differ" % role
            )
        rows.append({
            "code": trade.code,
            "entry_date": trade.buy.trade_date.isoformat(),
            "entry_source": source["entry_source"],
            "entry_branch": source["entry_branch"] or "NONE",
            "supporters": "+".join(sorted(source["supporters"])),
            "entry_rank": source["entry_rank"],
            "pnl": trade.pnl,
            "return_rate": trade.return_rate,
            "features": features,
        })
    return rows


def _quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values):
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
    }


def _entry_market_state(features):
    slope = _finite_number(
        features.get("normalized_boll_mid_slope"),
        "entry market state normalized_boll_mid_slope",
    )
    volume_ratio = _finite_number(
        features.get("volume_ratio"),
        "entry market state volume_ratio", nonnegative=True,
    )
    slope_state = (
        "SLOPE_POSITIVE" if slope > 0 else "SLOPE_NONPOSITIVE"
    )
    volume_state = (
        "VOLUME_ABOVE_ONE"
        if volume_ratio > 1.0 else "VOLUME_AT_OR_BELOW_ONE"
    )
    return "%s|%s" % (slope_state, volume_state)


def _profit_concentration(rows):
    gross_profits = sorted(
        (row["pnl"] for row in rows if row["pnl"] > 0), reverse=True,
    )
    total = sum(gross_profits)
    return {
        "gross_profit": total,
        "top_1_gross_profit_share": (
            gross_profits[0] / total if gross_profits else None
        ),
        "top_3_gross_profit_share": (
            sum(gross_profits[:3]) / total if gross_profits else None
        ),
    }


def _year_summary(rows):
    grouped = {}
    for row in rows:
        year = row["entry_date"][:4]
        grouped.setdefault(year, []).append(row)
    return {
        year: {
            "count": len(values),
            "wins": sum(row["pnl"] > 0 for row in values),
            "win_rate": _safe_rate(
                sum(row["pnl"] > 0 for row in values), len(values),
            ),
            "pnl": sum(row["pnl"] for row in values),
        }
        for year, values in sorted(grouped.items())
    }


def _entry_quality_cohort(rows, overall_win_rate, overall_by_year):
    wins = sum(row["pnl"] > 0 for row in rows)
    losses = sum(row["pnl"] < 0 for row in rows)
    breakeven = sum(row["pnl"] == 0 for row in rows)
    pnl = sum(row["pnl"] for row in rows)
    median_return = statistics.median(
        row["return_rate"] for row in rows
    )
    concentration = _profit_concentration(rows)
    by_year = _year_summary(rows)
    stable_years = sum(
        summary["count"] >= 2
        and summary["win_rate"]
        < overall_by_year[year]["win_rate"]
        for year, summary in by_year.items()
    )
    gate = {
        "at_least_eight_trades": len(rows) >= 8,
        "at_least_four_losses": losses >= 4,
        "win_rate_lags_overall_by_ten_points": (
            _safe_rate(wins, len(rows)) <= overall_win_rate - 0.10
        ),
        "median_return_not_positive": median_return <= 0,
        "pnl_or_concentration_condition": (
            pnl <= 0
            or (
                concentration["top_1_gross_profit_share"] is not None
                and concentration["top_1_gross_profit_share"] >= 0.50
                and median_return <= 0
            )
        ),
        "stable_in_at_least_two_years": stable_years >= 2,
    }
    gate["passed"] = all(gate.values())
    return {
        "count": len(rows),
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": _safe_rate(wins, len(rows)),
        "pnl": pnl,
        "median_return": median_return,
        "profit_concentration": concentration,
        "by_entry_year": by_year,
        "candidate_gate": gate,
    }


def _entry_quality_path(rows):
    wins = sum(row["pnl"] > 0 for row in rows)
    losses = sum(row["pnl"] < 0 for row in rows)
    breakeven = sum(row["pnl"] == 0 for row in rows)
    overall_win_rate = _safe_rate(wins, len(rows))
    overall_by_year = _year_summary(rows)
    categorical = {}
    for field in ENTRY_QUALITY_CATEGORICAL_FIELDS:
        grouped = {}
        for row in rows:
            value = (
                _entry_market_state(row["features"])
                if field == "entry_market_state" else row[field]
            )
            grouped.setdefault(str(value), []).append(row)
        categorical[field] = {
            value: _entry_quality_cohort(
                values, overall_win_rate, overall_by_year,
            )
            for value, values in sorted(grouped.items())
        }
    continuous = {}
    for field in ENTRY_QUALITY_CONTINUOUS_FIELDS:
        continuous[field] = {
            "winner": _distribution([
                row["features"][field] for row in rows if row["pnl"] > 0
            ]),
            "loser": _distribution([
                row["features"][field] for row in rows if row["pnl"] < 0
            ]),
            "breakeven": _distribution([
                row["features"][field] for row in rows if row["pnl"] == 0
            ]),
        }
    return {
        "closed_count": len(rows),
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": overall_win_rate,
        "pnl": sum(row["pnl"] for row in rows),
        "profit_concentration": _profit_concentration(rows),
        "by_entry_year": overall_by_year,
        "categorical": categorical,
        "continuous": continuous,
    }


def _volume_ratio_marginal_path(rows):
    wins = sum(row["pnl"] > 0 for row in rows)
    overall_win_rate = _safe_rate(wins, len(rows))
    overall_by_year = _year_summary(rows)
    grouped = {}
    for row in rows:
        volume_ratio = _finite_number(
            row["features"].get("volume_ratio"),
            "entry volume ratio marginal value", nonnegative=True,
        )
        group = (
            "VOLUME_ABOVE_ONE"
            if volume_ratio > 1.0 else "VOLUME_AT_OR_BELOW_ONE"
        )
        grouped.setdefault(group, []).append(row)
    groups = {}
    for group, values in sorted(grouped.items()):
        summary = _entry_quality_cohort(
            values, overall_win_rate, overall_by_year,
        )
        summary.pop("candidate_gate")
        groups[group] = summary
    result = {
        "boundary": {
            "above_one": "> 1.0",
            "at_or_below_one": "<= 1.0",
            "threshold_search_performed": False,
            "delta_zero_absolute_tolerance": (
                VOLUME_RATIO_DELTA_ZERO_ABS_TOLERANCE
            ),
        },
        "groups": groups,
        "comparison_available": False,
        "overall_delta_at_or_below_minus_above": None,
        "by_entry_year_delta_at_or_below_minus_above": {},
        "cross_year_direction_stability": {
            "win_rate": None,
            "pnl": None,
        },
    }
    below = groups.get("VOLUME_AT_OR_BELOW_ONE")
    above = groups.get("VOLUME_ABOVE_ONE")
    if below is None or above is None:
        return result
    result["comparison_available"] = True
    result["overall_delta_at_or_below_minus_above"] = {
        "win_rate": below["win_rate"] - above["win_rate"],
        "pnl": below["pnl"] - above["pnl"],
        "median_return": below["median_return"] - above["median_return"],
    }
    common_years = sorted(
        set(below["by_entry_year"]) & set(above["by_entry_year"])
    )
    by_year_delta = {
        year: {
            "win_rate": (
                below["by_entry_year"][year]["win_rate"]
                - above["by_entry_year"][year]["win_rate"]
            ),
            "pnl": (
                below["by_entry_year"][year]["pnl"]
                - above["by_entry_year"][year]["pnl"]
            ),
        }
        for year in common_years
    }
    result["by_entry_year_delta_at_or_below_minus_above"] = by_year_delta
    result["cross_year_direction_stability"] = {
        metric: _year_delta_directions_stable(by_year_delta, metric)
        for metric in ("win_rate", "pnl")
    }
    return result


def _delta_direction(value):
    if value is None:
        return "UNAVAILABLE"
    if abs(value) <= VOLUME_RATIO_DELTA_ZERO_ABS_TOLERANCE:
        return "ZERO"
    if value > 0:
        return "POSITIVE"
    if value < 0:
        return "NEGATIVE"
    return "ZERO"


def _same_available_delta_direction(left, right):
    left_direction = _delta_direction(left)
    right_direction = _delta_direction(right)
    if "UNAVAILABLE" in (left_direction, right_direction):
        return None
    if "ZERO" in (left_direction, right_direction):
        return False
    return left_direction == right_direction


def _year_delta_directions_stable(by_year_delta, metric):
    if len(by_year_delta) < 2:
        return None
    directions = {
        _delta_direction(values.get(metric))
        for values in by_year_delta.values()
    }
    if "UNAVAILABLE" in directions:
        return None
    if "ZERO" in directions:
        return False
    return len(directions) == 1


def _volume_ratio_marginal_attribution(ordinary_rows, double_rows):
    ordinary = _volume_ratio_marginal_path(ordinary_rows)
    double_friction = _volume_ratio_marginal_path(double_rows)
    metrics = ("win_rate", "pnl", "median_return")
    ordinary_overall = (
        ordinary["overall_delta_at_or_below_minus_above"] or {}
    )
    double_overall = (
        double_friction["overall_delta_at_or_below_minus_above"] or {}
    )
    overall_matches = {
        metric: _same_available_delta_direction(
            ordinary_overall.get(metric), double_overall.get(metric),
        )
        for metric in metrics
    }
    ordinary_years = ordinary[
        "by_entry_year_delta_at_or_below_minus_above"
    ]
    double_years = double_friction[
        "by_entry_year_delta_at_or_below_minus_above"
    ]
    common_years = sorted(set(ordinary_years) & set(double_years))
    yearly_matches = {
        year: {
            metric: _same_available_delta_direction(
                ordinary_years[year].get(metric),
                double_years[year].get(metric),
            )
            for metric in ("win_rate", "pnl")
        }
        for year in common_years
    }
    direction_matches = (
        list(overall_matches.values())
        + [
            value
            for matches in yearly_matches.values()
            for value in matches.values()
        ]
    )
    complete_direction_evidence = (
        ordinary["comparison_available"]
        and double_friction["comparison_available"]
        and len(yearly_matches) >= 2
        and all(value is not None for value in direction_matches)
    )
    all_directions_match = (
        all(direction_matches) if complete_direction_evidence else None
    )
    return {
        "scope": {
            "processing_stage": "POST_BACKTEST_READ_ONLY_ATTRIBUTION",
            "path_assumption": "ORIGINAL_TRADE_PATH_FIXED",
            "strategy_behavior_changed": False,
            "rule_candidate_created": False,
        },
        "ordinary": ordinary,
        "double_friction": double_friction,
        "cross_friction_stability": {
            "overall_delta_direction_matches": overall_matches,
            "by_entry_year_delta_direction_matches": yearly_matches,
            "all_reported_directions_match": all_directions_match,
        },
    }


def _entry_quality_candidate_decision(ordinary, double_friction):
    eligible = []
    for field in ENTRY_QUALITY_CATEGORICAL_FIELDS:
        if field == "code":
            continue
        ordinary_groups = ordinary["categorical"][field]
        double_groups = double_friction["categorical"][field]
        for value in sorted(set(ordinary_groups) & set(double_groups)):
            if (ordinary_groups[value]["candidate_gate"]["passed"]
                    and double_groups[value]["candidate_gate"]["passed"]):
                eligible.append({"field": field, "value": value})
    return {
        "eligible_groups": eligible,
        "proceed_to_counterfactual_design": bool(eligible),
    }


def _relative_sell_diagnostics(records, registrations, outcomes):
    held_codes = set()
    held_ids = set()
    for record in records:
        event = record.get("event")
        if event == "order_transition" and record.get("outcome") == "FILLED":
            code = record.get("code")
            if (record.get("side") == "BUY"
                    and record.get("before_amount") == 0
                    and isinstance(record.get("after_amount"), int)
                    and record.get("after_amount") > 0):
                held_codes.add(code)
            elif (record.get("side") == "SELL"
                    and record.get("after_amount") == 0):
                held_codes.discard(code)
            continue
        if (event == "relative_resonance_observation"
                and record.get("direction") == "SELL_TURN"
                and record.get("code") in held_codes):
            held_ids.add(record.get("relative_observation_id"))
    horizon_five = []
    for observation_id in held_ids:
        record = outcomes.get((observation_id, 5))
        if record is None:
            continue
        outcome = record.get("outcome") or {}
        if outcome.get("status") != "RECORDED":
            continue
        value = _finite_number(
            outcome.get("return"), "relative sell outcome return",
        )
        horizon_five.append(value)
    return {
        "held_observation_count": len(held_ids),
        "horizon_5_count": len(horizon_five),
        "horizon_5_sell_hit_rate": _safe_rate(
            sum(value < 0 for value in horizon_five), len(horizon_five),
        ),
        "horizon_5_mean_forward_return": (
            statistics.fmean(horizon_five) if horizon_five else None
        ),
    }


def _non_recovery_trigger(
        buy, actual_exit_date, source, atr_observations, amounts,
        evaluation_sessions, commission_rate):
    try:
        buy_index = evaluation_sessions.index(buy.trade_date)
    except ValueError as exc:
        raise ValueError("buy date absent from evaluation sessions") from exc
    decision_index = buy_index + NON_RECOVERY_COMPLETED_SESSIONS
    if decision_index >= len(evaluation_sessions):
        return None
    decision_date = evaluation_sessions[decision_index]
    if actual_exit_date is not None and decision_date > actual_exit_date:
        return None
    observation = atr_observations.get((decision_date, buy.code))
    if observation is None:
        return None
    anchor = observation.get("highest_close_anchor")
    if anchor is None or anchor > buy.price:
        return None
    prior_session = evaluation_sessions[decision_index - 1]
    execution_amount = amounts.get(prior_session, {}).get(buy.code)
    if execution_amount is None:
        raise ValueError(
            "non-recovery trigger lacks prior-session position amount: %s/%s"
            % (decision_date, buy.code)
        )
    execution_price = observation["current_price"]
    commission = max(
        MINIMUM_COMMISSION,
        execution_price * execution_amount * commission_rate,
    )
    buy_cost = buy.price * buy.amount + buy.commission
    counterfactual_pnl = (
        execution_price * execution_amount - commission - buy_cost
    )
    return {
        "code": buy.code,
        "entry_date": buy.trade_date.isoformat(),
        "decision_date": decision_date.isoformat(),
        "actual_exit_date": (
            actual_exit_date.isoformat() if actual_exit_date is not None
            else None
        ),
        "entry_source": source["entry_source"],
        "entry_branch": source["entry_branch"],
        "entry_price": buy.price,
        "prior_highest_close_anchor": anchor,
        "execution_price": execution_price,
        "execution_amount": execution_amount,
        "execution_commission": commission,
        "counterfactual_pnl": counterfactual_pnl,
    }


def _non_recovery_counterfactual(
        completed, open_positions, sources, atr_observations, amounts,
        evaluation_sessions, commission_rate):
    rows = []
    counterfactual_by_identity = {}
    for trade in completed:
        identity = (trade.buy.trade_date, trade.code)
        row = _non_recovery_trigger(
            trade.buy, trade.sell.trade_date, sources[identity],
            atr_observations, amounts, evaluation_sessions,
            commission_rate,
        )
        if row is None:
            continue
        row.update({
            "actual_pnl": trade.pnl,
            "pnl_delta": row["counterfactual_pnl"] - trade.pnl,
            "actual_winner": trade.pnl > 0,
            "counterfactual_winner": row["counterfactual_pnl"] > 0,
        })
        rows.append(row)
        counterfactual_by_identity[identity] = row["counterfactual_pnl"]
    open_rows = []
    for buy in open_positions:
        identity = (buy.trade_date, buy.code)
        row = _non_recovery_trigger(
            buy, None, sources[identity], atr_observations, amounts,
            evaluation_sessions, commission_rate,
        )
        if row is not None:
            open_rows.append(row)
    actual_pnls = [trade.pnl for trade in completed]
    counterfactual_pnls = [
        counterfactual_by_identity.get(
            (trade.buy.trade_date, trade.code), trade.pnl,
        )
        for trade in completed
    ]
    actual_wins = sum(value > 0 for value in actual_pnls)
    counterfactual_wins = sum(value > 0 for value in counterfactual_pnls)
    actual_total = sum(actual_pnls)
    counterfactual_total = sum(counterfactual_pnls)
    actual_worst = min(actual_pnls) if actual_pnls else None
    counterfactual_worst = (
        min(counterfactual_pnls) if counterfactual_pnls else None
    )
    improved_count = sum(row["pnl_delta"] > 0 for row in rows)
    harmed_count = sum(row["pnl_delta"] < 0 for row in rows)
    winners_turned_to_losses = sum(
        row["actual_winner"] and not row["counterfactual_winner"]
        for row in rows
    )
    gate = {
        "win_rate_not_lower": counterfactual_wins >= actual_wins,
        "closed_pnl_improved": counterfactual_total > actual_total,
        "worst_trade_loss_reduced": (
            actual_worst is not None
            and actual_worst < 0
            and counterfactual_worst > actual_worst
        ),
        "at_least_three_improved_trades": improved_count >= 3,
    }
    gate["passed"] = all(gate.values())
    return {
        "commission_rate": commission_rate,
        "closed_trade_count": len(completed),
        "open_position_count": len(open_positions),
        "triggered_closed_count": len(rows),
        "triggered_open_count": len(open_rows),
        "improved_trade_count": improved_count,
        "harmed_trade_count": harmed_count,
        "actual_winners_turned_to_losses": winners_turned_to_losses,
        "actual_wins": actual_wins,
        "counterfactual_wins": counterfactual_wins,
        "actual_win_rate": _safe_rate(actual_wins, len(completed)),
        "counterfactual_win_rate": _safe_rate(
            counterfactual_wins, len(completed),
        ),
        "actual_closed_pnl": actual_total,
        "counterfactual_closed_pnl": counterfactual_total,
        "pnl_delta": counterfactual_total - actual_total,
        "actual_worst_trade_pnl": actual_worst,
        "counterfactual_worst_trade_pnl": counterfactual_worst,
        "gate": gate,
        "rows": sorted(
            rows, key=lambda row: (row["decision_date"], row["code"]),
        ),
        "open_rows": sorted(
            open_rows,
            key=lambda row: (row["decision_date"], row["code"]),
        ),
    }


def _max_drawdown_episode(points, sources, completed, open_positions):
    peak = points[0]
    best = (0.0, peak, peak)
    for point in points:
        if point.total_value > peak.total_value:
            peak = point
        drawdown = 1.0 - point.total_value / peak.total_value
        if drawdown > best[0]:
            best = (drawdown, peak, point)
    _, peak_point, trough_point = best
    intervals = []
    for trade in completed:
        intervals.append((
            trade.code, trade.buy.trade_date, trade.sell.trade_date,
            sources[(trade.buy.trade_date, trade.code)],
        ))
    for fill in open_positions:
        intervals.append((
            fill.code, fill.trade_date, None,
            sources[(fill.trade_date, fill.code)],
        ))

    def holdings(point):
        result = []
        for code, amount in point.positions:
            matching = [
                (entry_date, source)
                for entry_code, entry_date, exit_date, source in intervals
                if (entry_code == code
                    and entry_date <= point.closing_date
                    and (exit_date is None
                         or point.closing_date < exit_date))
            ]
            if len(matching) != 1:
                raise ValueError(
                    "portfolio holding lacks one active trade interval: %s/%s"
                    % (point.closing_date, code)
                )
            entry_date, source = max(matching, key=lambda item: item[0])
            result.append({
                "code": code,
                "amount": amount,
                "entry_date": entry_date.isoformat(),
                "entry_source": source["entry_source"],
                "entry_branch": source["entry_branch"],
            })
        return result

    return {
        "max_drawdown": best[0],
        "peak_date": peak_point.closing_date.isoformat(),
        "trough_date": trough_point.closing_date.isoformat(),
        "peak_total_value": peak_point.total_value,
        "trough_total_value": trough_point.total_value,
        "peak_holdings": holdings(peak_point),
        "trough_holdings": holdings(trough_point),
    }


def _manifest_report(manifest):
    sessions = _evaluation_sessions(manifest)
    return {
        "sha256": manifest.sha256,
        "session_count": len(sessions),
        "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
    }


def _source_file_report(paths):
    result = []
    for path in sorted(
            pathlib.Path(value).expanduser().resolve(strict=True)
            for value in paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result.append({
            "path": str(path),
            "sha256": digest.hexdigest(),
        })
    return result


def analyze_paths(ordinary_paths, double_friction_paths, manifest):
    ordinary = parse_joinquant_log(ordinary_paths)
    double_friction = parse_joinquant_log(double_friction_paths)
    ordinary_init = _require_identity(ordinary, "ordinary")
    double_init = _require_identity(double_friction, "double friction")
    _validate_training_boundary(ordinary, manifest, "ordinary")
    _validate_training_boundary(double_friction, manifest, "double friction")
    _filled_order_transitions(ordinary.records, ordinary.fills, "ordinary")
    _filled_order_transitions(
        double_friction.records, double_friction.fills, "double friction",
    )
    _reconcile_portfolio_ledger(ordinary.records, "ordinary")
    _reconcile_portfolio_ledger(
        double_friction.records, "double friction",
    )
    ordinary_relative, ordinary_outcomes = _relative_records(
        ordinary.records, "ordinary", manifest.sessions,
    )
    double_relative, _ = _relative_records(
        double_friction.records, "double friction", manifest.sessions,
    )
    path_reconciliation = _reconcile_paths(ordinary, double_friction)
    completed, open_positions = _pair_completed_trades(ordinary.fills)
    double_completed, double_open_positions = _pair_completed_trades(
        double_friction.fills
    )
    sources = _trade_sources(ordinary, ordinary_relative)
    double_sources = _trade_sources(double_friction, double_relative)
    atr_observations = _atr_observations_by_code(ordinary.records)
    double_atr_observations = _atr_observations_by_code(
        double_friction.records
    )
    marks = {
        key: observation["current_price"]
        for key, observation in atr_observations.items()
    }
    amounts = _position_amounts_by_date(ordinary.portfolio_points)
    double_amounts = _position_amounts_by_date(
        double_friction.portfolio_points
    )
    evaluation_sessions = _evaluation_sessions(manifest)
    sell_registrations = {
        observation_id: record
        for observation_id, record in ordinary_relative.items()
        if record.get("direction") == "SELL_TURN"
    }
    sell_outcomes = {
        key: record for key, record in ordinary_outcomes.items()
        if key[0] in sell_registrations
    }
    trade_rows = [
        _trade_report(
            trade,
            sources[(trade.buy.trade_date, trade.code)],
            marks,
            amounts,
            sell_registrations,
            evaluation_sessions,
        )
        for trade in completed
    ]
    ordinary_counterfactual = _non_recovery_counterfactual(
        completed, open_positions, sources, atr_observations, amounts,
        evaluation_sessions, ORDINARY_COMMISSION_RATE,
    )
    double_counterfactual = _non_recovery_counterfactual(
        double_completed, double_open_positions, double_sources,
        double_atr_observations, double_amounts, evaluation_sessions,
        DOUBLE_FRICTION_COMMISSION_RATE,
    )
    ordinary_entry_rows = _entry_quality_rows(
        ordinary.records, completed, sources, manifest.sessions, "ordinary",
    )
    double_entry_rows = _entry_quality_rows(
        double_friction.records, double_completed, double_sources,
        manifest.sessions, "double friction",
    )
    ordinary_entry_quality = _entry_quality_path(ordinary_entry_rows)
    double_entry_quality = _entry_quality_path(double_entry_rows)
    return {
        "source_files": {
            "ordinary": _source_file_report(ordinary_paths),
            "double_friction": _source_file_report(double_friction_paths),
        },
        "session_calendar": _manifest_report(manifest),
        "identity": {
            "ordinary_build": ordinary_init["build"],
            "double_friction_build": double_init["build"],
            "atr_exit_policy": ordinary_init["atr_exit_policy"],
            "relative_buy_policy": ordinary_init["relative_buy_policy"],
        },
        "path_reconciliation": path_reconciliation,
        "non_recovery_counterfactual": {
            "rule": {
                "completed_holding_sessions": (
                    NON_RECOVERY_COMPLETED_SESSIONS
                ),
                "qualification": (
                    "PRIOR_HIGHEST_CLOSE_NOT_ABOVE_ENTRY_PRICE"
                ),
                "execution": "DECISION_SESSION_0935_ATR_CHECK_PRICE",
                "path_assumption": "ORIGINAL_TRADE_PATH_FIXED",
            },
            "ordinary": ordinary_counterfactual,
            "double_friction": double_counterfactual,
            "decision": {
                "ordinary_passed": ordinary_counterfactual["gate"][
                    "passed"
                ],
                "double_friction_passed": double_counterfactual["gate"][
                    "passed"
                ],
                "proceed_to_strategy_candidate": (
                    ordinary_counterfactual["gate"]["passed"]
                    and double_counterfactual["gate"]["passed"]
                ),
            },
        },
        "entry_quality_attribution": {
            "scope": {
                "processing_stage": "POST_BACKTEST_READ_ONLY_ATTRIBUTION",
                "path_assumption": "ORIGINAL_TRADE_PATH_FIXED",
                "strategy_behavior_changed": False,
                "open_positions_excluded": True,
            },
            "feature_contract": {
                "categorical": list(ENTRY_QUALITY_CATEGORICAL_FIELDS),
                "continuous": list(ENTRY_QUALITY_CONTINUOUS_FIELDS),
                "forbidden_post_entry_predictors": [
                    "mfe", "mae", "max_profit_giveback",
                    "longest_underwater_sessions",
                ],
            },
            "semantic_boundaries": {
                "normalized_boll_mid_slope": {
                    "positive": "> 0",
                    "nonpositive": "<= 0",
                },
                "volume_ratio": {
                    "above_one": "> 1.0",
                    "at_or_below_one": "<= 1.0",
                },
                "threshold_search_performed": False,
            },
            "candidate_gate_contract": {
                "minimum_trades": 8,
                "minimum_losses": 4,
                "minimum_win_rate_lag": 0.10,
                "maximum_median_return": 0.0,
                "minimum_consistent_years": 2,
                "minimum_trades_per_consistent_year": 2,
                "alternative_profit_concentration": 0.50,
                "code_is_diagnostic_only": True,
            },
            "ordinary": ordinary_entry_quality,
            "double_friction": double_entry_quality,
            "volume_ratio_marginal": _volume_ratio_marginal_attribution(
                ordinary_entry_rows, double_entry_rows,
            ),
            "decision": _entry_quality_candidate_decision(
                ordinary_entry_quality, double_entry_quality,
            ),
        },
        "trade_summary": {
            "closed_count": len(completed),
            "open_count": len(open_positions),
            "wins": sum(trade.pnl > 0 for trade in completed),
            "win_rate": _safe_rate(
                sum(trade.pnl > 0 for trade in completed), len(completed),
            ),
            "pnl": sum(trade.pnl for trade in completed),
            "by_entry_source": _group_summary(
                trade_rows, "entry_source",
            ),
            "by_entry_branch": _group_summary(
                trade_rows, "entry_branch",
            ),
        },
        "relative_sell_diagnostics": _relative_sell_diagnostics(
            ordinary.records, sell_registrations, sell_outcomes,
        ),
        "drawdown_episode": _max_drawdown_episode(
            ordinary.portfolio_points, sources, completed, open_positions,
        ),
        "trades": trade_rows,
    }


def _paths_alias(left, right):
    left = pathlib.Path(left).expanduser().resolve(strict=False)
    right = pathlib.Path(right).expanduser().resolve(strict=False)
    if left == right:
        return True
    if left.exists() and right.exists():
        return os.path.samefile(left, right)
    return False


def _write_json_atomically(path_value, payload):
    path = pathlib.Path(path_value).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent), text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload, stream, ensure_ascii=False, indent=2,
                sort_keys=True, allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-log", action="append", required=True)
    parser.add_argument(
        "--double-friction-log", action="append", required=True,
    )
    parser.add_argument("--session-calendar", required=True)
    parser.add_argument("--session-calendar-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _argument_parser().parse_args(argv)
    inputs = (
        list(args.ordinary_log)
        + list(args.double_friction_log)
        + [args.session_calendar]
    )
    if any(_paths_alias(args.output, value) for value in inputs):
        raise ValueError("output aliases an input")
    for index, left in enumerate(inputs):
        for right in inputs[index + 1:]:
            if _paths_alias(left, right):
                raise ValueError("analysis inputs must be distinct")
    raw_manifest = read_session_calendar_manifest_bytes(
        args.session_calendar
    )
    manifest = validate_session_calendar_manifest(
        raw_manifest, args.session_calendar_sha256,
    )
    report = analyze_paths(
        args.ordinary_log, args.double_friction_log, manifest,
    )
    _write_json_atomically(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
