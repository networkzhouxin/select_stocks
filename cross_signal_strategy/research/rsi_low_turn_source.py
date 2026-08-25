"""Fail-closed, point-in-time input loader for the RSI low-turn shadow."""

from dataclasses import dataclass
from datetime import date, datetime, time
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cross_signal_strategy.research.rsi_low_turn_outcomes import FutureSnapshot
from cross_signal_strategy.research.rsi_low_turn_shadow import RsiTurnInput, calculate_rsi6


MIN_COLLECTION_START = date(2026, 8, 26)
PURPOSE = "rsi_low_turn_prospective_shadow"
SHANGHAI = ZoneInfo("Asia/Shanghai")
_MANIFEST_KEYS = frozenset({
    "purpose", "version", "collection_start", "timezone", "append_only",
    "daily_subdir", "minute_subdir",
})
_DAILY_COLUMNS = [
    "code", "date", "open", "high", "low", "close", "volume", "available_at", "source",
]
_MINUTE_COLUMNS = [
    "code", "timestamp", "open", "close", "volume", "num_trades", "available_at", "source",
]


class SourceContractError(ValueError):
    """Raised when prospective-shadow input lacks a point-in-time proof."""


@dataclass(frozen=True)
class SourceManifest:
    root: Path
    purpose: str
    version: str
    collection_start: date
    timezone: str
    append_only: bool
    daily_subdir: str
    minute_subdir: str


class ApprovedFuturePriceSource:
    """Read prospective future labels from the same approved, append-only source."""

    def __init__(
        self,
        data_root: Path,
        approved_root: Path,
        *,
        source_contents: Mapping[str, bytes] | None = None,
        collected_at: datetime | None = None,
    ):
        self.manifest = load_manifest(data_root, approved_root)
        self.source_contents = None if source_contents is None else dict(source_contents)
        if collected_at is not None:
            _require_future_as_of(collected_at)
        self.collected_at = collected_at

    def snapshot_for(
        self, event: Mapping[str, object], horizon: int, as_of: datetime,
    ) -> FutureSnapshot:
        code = _event_code(event)
        _require_code(code)
        _require_future_as_of(as_of)
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer")

        arrival_date = _event_arrival_date(event)
        daily_path = _source_file(self.manifest.root / self.manifest.daily_subdir, code, "daily")
        minute_path = _source_file(self.manifest.root / self.manifest.minute_subdir, code, "minute")
        daily_relative_path = f"{self.manifest.daily_subdir}/{code}.csv"
        minute_relative_path = f"{self.manifest.minute_subdir}/{code}.csv"
        daily_content = _snapshot_content(self.source_contents, daily_relative_path, daily_path)
        minute_content = _snapshot_content(self.source_contents, minute_relative_path, minute_path)
        daily = _future_daily_frame_from_content(daily_content, code)
        minute = _future_minute_frame_from_content(minute_content, code)
        collected_at = self.collected_at or as_of
        if collected_at > as_of:
            raise SourceContractError("future source collected_at cannot follow as_of")
        consumed_daily = daily[
            (daily["date"] > arrival_date)
            & (daily["_available_at"] <= pd.Timestamp(as_of))
        ]
        _require_consumed_sources(consumed_daily, "daily")
        future_sessions = sorted(consumed_daily["date"])
        if len(future_sessions) < horizon:
            return FutureSnapshot(horizon, "pending_horizon_not_arrived", None, None, None, None)
        target_date = future_sessions[horizon - 1]
        if target_date > as_of.date():
            return FutureSnapshot(horizon, "pending_horizon_not_arrived", None, None, None, None)

        target_timestamp = pd.Timestamp(datetime.combine(target_date, time(9, 35), SHANGHAI))
        exact_timestamp = minute[minute["_timestamp"] == target_timestamp]
        if len(exact_timestamp) != 1:
            return FutureSnapshot(
                horizon, "pending_missing_executable_price", None, None, None, None,
                target_timestamp.to_pydatetime(), collected_at, minute_relative_path,
                hashlib.sha256(minute_content).hexdigest(), len(minute_content),
            )
        timely = exact_timestamp[
            (exact_timestamp["_available_at"] <= pd.Timestamp(as_of))
            & (exact_timestamp["_available_at"] <= target_timestamp)
        ]
        if len(timely) != 1:
            return FutureSnapshot(
                horizon, "pending_missing_executable_price", None, None, None, None,
                target_timestamp.to_pydatetime(), collected_at, minute_relative_path,
                hashlib.sha256(minute_content).hexdigest(), len(minute_content),
            )
        quote = timely.iloc[0]
        _require_source_value(quote["source"], "minute")
        exit_open, volume, num_trades = (
            _as_finite(quote[column]) for column in ("open", "volume", "num_trades")
        )
        if not (exit_open is not None and exit_open > 0 and volume is not None and volume > 0
                and num_trades is not None and num_trades > 0):
            return FutureSnapshot(
                horizon, "pending_missing_executable_price", None, None, None, None,
                target_timestamp.to_pydatetime(), collected_at, minute_relative_path,
                hashlib.sha256(minute_content).hexdigest(), len(minute_content),
            )

        return FutureSnapshot(
            horizon,
            "matured",
            exit_open,
            None,
            None,
            quote["_available_at"].to_pydatetime(),
            target_timestamp.to_pydatetime(),
            collected_at,
            minute_relative_path,
            hashlib.sha256(minute_content).hexdigest(),
            len(minute_content),
        )


def validate_root(data_root: Path, approved_root: Path) -> Path:
    """Require the caller to provide the separately approved, exact source root."""
    try:
        data = Path(data_root).resolve(strict=True)
        approved = Path(approved_root).resolve(strict=True)
    except FileNotFoundError as exc:
        raise SourceContractError("data root or approved root does not exist") from exc
    if data != approved:
        raise SourceContractError("data root does not equal approved root")
    forbidden = (
        "cross_signal_train_2019_2021", "cross_signal_warmup_2018", "按年份合并", "merged", "validation",
    )
    rendered = str(data).casefold()
    if any(token.casefold() in rendered for token in forbidden):
        raise SourceContractError("forbidden research or validation root")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(data_root: Path, approved_root: Path) -> SourceManifest:
    root = validate_root(data_root, approved_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SourceContractError("manifest.json is required")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceContractError("manifest.json is unreadable") from exc
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
        raise SourceContractError("manifest must contain exactly the source-contract fields")
    try:
        collection_start = date.fromisoformat(raw["collection_start"])
    except (TypeError, ValueError) as exc:
        raise SourceContractError("manifest collection_start must be an ISO date") from exc
    if raw["purpose"] != PURPOSE:
        raise SourceContractError("manifest purpose is not the RSI shadow purpose")
    if raw["version"] != "rsi-low-turn-shadow-v0.1":
        raise SourceContractError("manifest version is not the frozen shadow version")
    if collection_start < MIN_COLLECTION_START:
        raise SourceContractError("collection_start cannot precede 2026-08-26")
    if raw["timezone"] != "Asia/Shanghai":
        raise SourceContractError("manifest timezone must be Asia/Shanghai")
    if raw["append_only"] is not True:
        raise SourceContractError("manifest append_only must be true")
    if raw["daily_subdir"] != "daily" or raw["minute_subdir"] != "minute_0935":
        raise SourceContractError("manifest subdirectories do not match the source contract")
    return SourceManifest(root, raw["purpose"], raw["version"], collection_start,
                          raw["timezone"], raw["append_only"], raw["daily_subdir"],
                          raw["minute_subdir"])


def _require_arrival(arrival_dt: datetime) -> None:
    timezone_key = getattr(arrival_dt.tzinfo, "key", None)
    if timezone_key != "Asia/Shanghai":
        raise SourceContractError("arrival datetime must be aware Asia/Shanghai")
    if arrival_dt.timetz().replace(tzinfo=None) != time(9, 35):
        raise SourceContractError("arrival datetime must be exactly 09:35")


def _read_csv(path: Path, expected_columns: list[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise SourceContractError("%s source file is required" % label)
    try:
        content = path.read_bytes()
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise SourceContractError("%s source file is unreadable" % label) from exc
    return _read_csv_bytes(content, expected_columns, label)


def _read_csv_bytes(content: bytes, expected_columns: list[str], label: str) -> pd.DataFrame:
    if not isinstance(content, bytes):
        raise TypeError("source snapshot content must be bytes")
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise SourceContractError("%s source file is unreadable" % label) from exc
    if frame.columns.tolist() != expected_columns:
        raise SourceContractError("%s columns do not match the source contract" % label)
    return frame


def _require_code(code: str) -> None:
    if not isinstance(code, str) or re.fullmatch(r"\d{6}", code) is None:
        raise SourceContractError("code must be a supported six-digit ETF code")


def _source_file(subdir: Path, code: str, label: str) -> Path:
    try:
        approved_subdir = subdir.resolve(strict=True)
        candidate = (subdir / (code + ".csv")).resolve(strict=False)
    except OSError as exc:
        raise SourceContractError("%s source subdirectory is unavailable" % label) from exc
    if not approved_subdir.is_dir() or not candidate.is_relative_to(approved_subdir):
        raise SourceContractError("%s source path escapes approved subdirectory" % label)
    return candidate


def _aware_timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SourceContractError("%s must be an aware timestamp" % label) from exc
    if timestamp.tzinfo is None or getattr(timestamp.tzinfo, "key", None) not in ("Asia/Shanghai", None):
        raise SourceContractError("%s must be Asia/Shanghai-aware" % label)
    if timestamp.utcoffset() != SHANGHAI.utcoffset(timestamp.to_pydatetime()):
        raise SourceContractError("%s must use the Shanghai offset" % label)
    return timestamp.tz_convert(SHANGHAI)


def _as_finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _require_source_value(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceContractError(f"consumed {label} row source is unknown")
    return value.strip()


def _require_consumed_sources(frame: pd.DataFrame, label: str) -> None:
    for value in frame["source"]:
        _require_source_value(value, label)


def _require_future_as_of(as_of: datetime) -> None:
    if not isinstance(as_of, datetime) or getattr(as_of.tzinfo, "key", None) != "Asia/Shanghai":
        raise SourceContractError("future as_of must be Asia/Shanghai-aware")


def _event_code(event: Mapping[str, object]) -> str:
    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    code = event.get("code")
    if not isinstance(code, str):
        raise ValueError("event code must be a six-digit ETF code")
    return code


def _event_arrival_date(event: Mapping[str, object]) -> date:
    value = event.get("arrival_date")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("event arrival_date must be an ISO date") from exc
    raise ValueError("event arrival_date must be a date")


def _future_daily_frame(path: Path, code: str) -> pd.DataFrame:
    return _future_daily_frame_from_content(path.read_bytes(), code)


def _future_daily_frame_from_content(content: bytes, code: str) -> pd.DataFrame:
    daily = _read_csv_bytes(content, _DAILY_COLUMNS, "daily").copy()
    if not daily["code"].astype(str).eq(code).all():
        raise SourceContractError("source file code does not match requested code")
    try:
        daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.date
        for column in ("open", "high", "low", "close", "volume"):
            daily[column] = pd.to_numeric(daily[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise SourceContractError("daily source has invalid values") from exc
    daily["_available_at"] = daily["available_at"].map(
        lambda value: _aware_timestamp(value, "daily available_at")
    )
    if daily["date"].duplicated().any():
        raise SourceContractError("daily source has duplicate sessions")
    return daily


def prior_source_session_from_snapshot(
    content: bytes, code: str, arrival_dt: datetime,
) -> date:
    """Return the exact prior session proved by one immutable daily snapshot."""
    _require_code(code)
    _require_arrival(arrival_dt)
    daily = _read_csv_bytes(content, _DAILY_COLUMNS, "daily").copy()
    if not daily["code"].astype(str).eq(code).all():
        raise SourceContractError("source file code does not match requested code")
    try:
        daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.date
    except (TypeError, ValueError) as exc:
        raise SourceContractError("daily source has invalid values") from exc
    if daily["date"].duplicated().any():
        raise SourceContractError("daily source has duplicate sessions")
    candidates = daily.loc[daily["date"] < arrival_dt.date(), "date"]
    if candidates.empty:
        raise SourceContractError("no prior source session is available")
    return max(candidates)


def _future_minute_frame(path: Path, code: str) -> pd.DataFrame:
    return _future_minute_frame_from_content(path.read_bytes(), code)


def _future_minute_frame_from_content(content: bytes, code: str) -> pd.DataFrame:
    minute = _read_csv_bytes(content, _MINUTE_COLUMNS, "minute").copy()
    if not minute["code"].astype(str).eq(code).all():
        raise SourceContractError("source file code does not match requested code")
    minute["_timestamp"] = minute["timestamp"].map(
        lambda value: _aware_timestamp(value, "minute timestamp")
    )
    minute["_available_at"] = minute["available_at"].map(
        lambda value: _aware_timestamp(value, "minute available_at")
    )
    return minute


def _background(frame: pd.DataFrame) -> dict[str, float]:
    close, high, low = frame["close"], frame["high"], frame["low"]
    rsi12 = _rsi(close, 12)
    rsi24 = _rsi(close, 24)
    k, d, j = _kdj(high, low, close, 9, 3, 3)
    dif, dea, hist = _macd(close, 12, 26, 9)
    upper, mid, lower = _bollinger(close, 20, 2)
    atr = _atr(high, low, close, 14)
    return {
        "rsi12": rsi12.iloc[-1], "rsi24": rsi24.iloc[-1],
        "kdj_k": k.iloc[-1], "kdj_d": d.iloc[-1], "kdj_j": j.iloc[-1],
        "macd_dif": dif.iloc[-1], "macd_dea": dea.iloc[-1], "macd_hist": hist.iloc[-1],
        "boll_upper": upper.iloc[-1], "boll_mid": mid.iloc[-1], "boll_lower": lower.iloc[-1],
        "atr14": atr.iloc[-1],
    }


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    result = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    result[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    result[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return result


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int, m1: int, m2: int):
    lowest, highest = low.rolling(n).min(), high.rolling(n).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, 2 * (dif - dea)


def _bollinger(close: pd.Series, period: int, std_mult: float):
    mid, std = close.rolling(period).mean(), close.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def load_arrival_input(
    data_root: Path,
    approved_root: Path,
    code: str,
    arrival_dt: datetime,
    *,
    expected_prior_session: date | None = None,
    source_contents: Mapping[str, bytes] | None = None,
) -> RsiTurnInput:
    """Load only data whose session and publication time precede the 09:35 arrival."""
    _require_code(code)
    _require_arrival(arrival_dt)
    manifest = load_manifest(data_root, approved_root)
    if arrival_dt.date() < manifest.collection_start:
        raise SourceContractError("arrival precedes manifest collection_start")
    daily_path = _source_file(manifest.root / manifest.daily_subdir, code, "daily")
    minute_path = _source_file(manifest.root / manifest.minute_subdir, code, "minute")
    manifest_content = _snapshot_content(source_contents, "manifest.json", manifest.root / "manifest.json")
    daily_content = _snapshot_content(source_contents, f"daily/{code}.csv", daily_path)
    minute_content = _snapshot_content(source_contents, f"minute_0935/{code}.csv", minute_path)
    daily = _read_csv_bytes(daily_content, _DAILY_COLUMNS, "daily")
    minute = _read_csv_bytes(minute_content, _MINUTE_COLUMNS, "minute")
    if not daily["code"].astype(str).eq(code).all() or not minute["code"].astype(str).eq(code).all():
        raise SourceContractError("source file code does not match requested code")

    daily = daily.copy()
    try:
        daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.date
        for column in ("open", "high", "low", "close", "volume"):
            daily[column] = pd.to_numeric(daily[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise SourceContractError("daily source has invalid values") from exc
    daily["_available_at"] = daily["available_at"].map(lambda value: _aware_timestamp(value, "daily available_at"))
    if daily["date"].duplicated().any():
        raise SourceContractError("daily source has duplicate sessions")
    prior_session = prior_source_session_from_snapshot(daily_content, code, arrival_dt)
    if expected_prior_session is not None and prior_session != expected_prior_session:
        raise SourceContractError("ETF files do not share one consistent prior source session")
    exact_prior = daily[daily["date"] == prior_session]
    if len(exact_prior) != 1 or exact_prior.iloc[0]["_available_at"] > pd.Timestamp(arrival_dt):
        raise SourceContractError("exact prior source session is not available by arrival")
    causal = daily[
        (daily["date"] <= prior_session)
        & (daily["_available_at"] <= pd.Timestamp(arrival_dt))
    ].copy()
    causal = causal.sort_values("date", kind="stable")
    if causal.empty:
        raise SourceContractError("no causal daily rows are available")
    _require_consumed_sources(causal, "daily")

    minute = minute.copy()
    minute["_timestamp"] = minute["timestamp"].map(lambda value: _aware_timestamp(value, "minute timestamp"))
    minute["_available_at"] = minute["available_at"].map(lambda value: _aware_timestamp(value, "minute available_at"))
    arrival = pd.Timestamp(arrival_dt)
    timely = minute[(minute["_timestamp"] == arrival) & (minute["_available_at"] <= arrival)]
    if len(timely) != 1:
        raise SourceContractError("exact timely 09:35 minute proof is required")
    quote = timely.iloc[0]
    _require_source_value(quote["source"], "minute")
    entry_open, volume, num_trades = (_as_finite(quote[column]) for column in ("open", "volume", "num_trades"))
    price_proved = bool(entry_open is not None and entry_open > 0 and volume is not None and volume > 0 and num_trades is not None and num_trades > 0)

    rsi6 = calculate_rsi6(causal["close"])
    background = _background(causal)
    if len(rsi6) < 3 or len(causal) < 2:
        raise SourceContractError("insufficient indicator history for frozen inputs")
    r2, r1, r0 = (float(value) for value in rsi6.iloc[-3:])
    c1, c0 = (float(value) for value in causal["close"].iloc[-2:])
    if not all(
        math.isfinite(float(value))
        for value in (r2, r1, r0, c1, c0, *background.values())
    ):
        raise SourceContractError("insufficient indicator history or non-finite frozen input")
    hashes = tuple(
        hashlib.sha256(content).hexdigest()
        for content in (manifest_content, daily_content, minute_content)
    )
    return RsiTurnInput(code=code, arrival_dt=arrival_dt, signal_date=causal["date"].iloc[-1],
                        r2=r2, r1=r1, r0=r0, c1=c1, c0=c0, entry_open=entry_open,
                        price_proved=price_proved, price_reason=None if price_proved else "price_unproved",
                        background=background, source_hashes=hashes)


def _snapshot_content(
    source_contents: Mapping[str, bytes] | None, relative_path: str, path: Path,
) -> bytes:
    if source_contents is not None:
        content = source_contents.get(relative_path)
        if not isinstance(content, bytes):
            raise SourceContractError(f"source snapshot is missing: {relative_path}")
        return content
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SourceContractError(f"source snapshot is unreadable: {path.name}") from exc
