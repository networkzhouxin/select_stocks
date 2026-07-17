# -*- coding: utf-8 -*-
"""Point-in-time underlying-index data contract for training-only research.

The module is deliberately separate from every signal and execution adapter.
It may label an already completed trade for an observation-only diagnostic, but
it cannot alter scores, orders, positions, or risk controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Union

import pandas as pd


APPROVED_UNDERLYING_ROOT = Path(
    r"G:\financial\history_data\cross_signal_underlying_train_2018_2021"
)
SOURCE_START = pd.Timestamp("2018-01-01")
TRAINING_START = pd.Timestamp("2019-01-01")
TRAINING_END = pd.Timestamp("2021-12-31")
REQUIRED_COLUMNS = (
    "etf_code",
    "source_id",
    "session_date",
    "available_at",
    "close",
    "is_final",
)


@dataclass(frozen=True)
class UnderlyingSpec:
    etf_code: str
    source_id: str
    index_name: str
    publisher: str


UNDERLYING_SPECS: Mapping[str, UnderlyingSpec] = {
    "513100": UnderlyingSpec(
        "513100", "NDX", "Nasdaq-100 Index", "Nasdaq"
    ),
    "513500": UnderlyingSpec(
        "513500", "SPX", "S&P 500 Index", "S&P Dow Jones Indices"
    ),
    "513050": UnderlyingSpec(
        "513050", "H30533", "CSI Overseas China Internet 50 Index", "CSI"
    ),
    "513880": UnderlyingSpec(
        "513880", "N225", "Nikkei Stock Average", "Nikkei"
    ),
}


@dataclass(frozen=True)
class UnderlyingDirection:
    code: str
    source_id: str
    decision_at: str
    previous_session_date: str
    latest_session_date: str
    latest_available_at: str
    one_session_return: float
    confirmed: bool


PathLike = Union[str, Path]


def _resolve(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _code_text(value: object) -> str:
    return str(value).split(".")[0]


def _parse_aware_timestamps(values) -> pd.Series:
    parsed = []
    for value in values:
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("underlying frame contains invalid available_at") from exc
        if stamp.tzinfo is None:
            raise ValueError("underlying frame requires timezone-aware available_at")
        parsed.append(stamp.tz_convert("UTC"))
    return pd.Series(parsed, dtype="datetime64[ns, UTC]")


def validate_underlying_frame(
    frame: pd.DataFrame,
    expected_code: str,
) -> pd.DataFrame:
    """Validate one official-index history and normalize timestamps to UTC."""
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError("underlying frame missing columns: %s" % ", ".join(missing))

    code = _code_text(expected_code)
    spec = UNDERLYING_SPECS.get(code)
    if spec is None:
        raise ValueError("unsupported formal QDII code: %s" % code)

    normalized = frame.loc[:, REQUIRED_COLUMNS].copy()
    normalized["etf_code"] = normalized["etf_code"].map(_code_text)
    if not normalized["etf_code"].eq(code).all():
        raise ValueError("underlying frame ETF code mismatch for %s" % code)
    if not normalized["source_id"].astype(str).eq(spec.source_id).all():
        raise ValueError("underlying frame source mismatch for %s" % code)

    sessions = pd.to_datetime(normalized["session_date"], errors="coerce")
    if sessions.isna().any():
        raise ValueError("underlying frame contains invalid session_date")
    if (sessions < SOURCE_START).any() or (sessions > TRAINING_END).any():
        raise ValueError("underlying frame contains dates outside approved source dates")
    normalized["session_date"] = sessions.dt.normalize()
    if normalized["session_date"].duplicated().any():
        raise ValueError("underlying frame contains duplicate session_date")

    normalized["available_at"] = _parse_aware_timestamps(
        normalized["available_at"].tolist()
    ).array
    closes = pd.to_numeric(normalized["close"], errors="coerce")
    if closes.isna().any() or (closes <= 0).any():
        raise ValueError("underlying frame requires positive close")
    normalized["close"] = closes.astype(float)

    if not normalized["is_final"].map(lambda value: value is True).all():
        raise ValueError("underlying frame must contain final observations only")
    normalized["is_final"] = True
    return normalized.sort_values("session_date").reset_index(drop=True)


def _decision_timestamp(decision_at: object) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(decision_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("decision_at must be a timezone-aware timestamp") from exc
    if stamp.tzinfo is None:
        raise ValueError("decision_at must be a timezone-aware timestamp")
    china = stamp.tz_convert("Asia/Shanghai")
    if (china.hour, china.minute, china.second) != (9, 35, 0):
        raise ValueError("decision_at must be exactly 09:35 Asia/Shanghai")
    if china.tz_localize(None).normalize() < TRAINING_START or china.tz_localize(None).normalize() > TRAINING_END:
        raise ValueError("decision_at must stay inside the 2019-2021 training window")
    return china


def _visible_at(frame: pd.DataFrame, decision: pd.Timestamp) -> pd.DataFrame:
    decision_date = decision.tz_localize(None).normalize()
    return frame.loc[
        (frame["available_at"] <= decision.tz_convert("UTC"))
        & (frame["session_date"] <= decision_date)
    ]


def select_underlying_direction(
    frame: pd.DataFrame,
    code: str,
    decision_at: object,
) -> UnderlyingDirection | None:
    """Return the latest completed one-session sign visible at China 09:35."""
    code_text = _code_text(code)
    decision = _decision_timestamp(decision_at)
    normalized = validate_underlying_frame(frame, expected_code=code_text)
    visible = _visible_at(normalized, decision).sort_values("session_date")
    if len(visible) < 2:
        return None

    previous = visible.iloc[-2]
    latest = visible.iloc[-1]
    one_session_return = float(latest["close"]) / float(previous["close"]) - 1.0
    return UnderlyingDirection(
        code=code_text,
        source_id=UNDERLYING_SPECS[code_text].source_id,
        decision_at=decision.isoformat(),
        previous_session_date=pd.Timestamp(previous["session_date"]).strftime("%Y-%m-%d"),
        latest_session_date=pd.Timestamp(latest["session_date"]).strftime("%Y-%m-%d"),
        latest_available_at=pd.Timestamp(latest["available_at"]).isoformat(),
        one_session_return=one_session_return,
        confirmed=one_session_return > 0.0,
    )


@dataclass(frozen=True)
class UnderlyingMarketDataLoader:
    """Exact-root, read-only loader for isolated official-index histories."""

    root: PathLike = APPROVED_UNDERLYING_ROOT

    def __post_init__(self) -> None:
        resolved = _resolve(self.root)
        if resolved != _resolve(APPROVED_UNDERLYING_ROOT):
            raise ValueError(
                "Use approved underlying-index data root only: %s"
                % APPROVED_UNDERLYING_ROOT
            )
        object.__setattr__(self, "root", resolved)
        object.__setattr__(self, "_cache", {})

    def load_history(self, code: str, decision_at: object) -> pd.DataFrame:
        code_text = _code_text(code)
        decision = _decision_timestamp(decision_at)
        if code_text not in UNDERLYING_SPECS:
            raise ValueError("unsupported formal QDII code: %s" % code_text)

        paths = [self.root / "warmup" / "2018" / (code_text + ".csv")]
        paths.extend(
            self.root / "training" / str(year) / (code_text + ".csv")
            for year in range(2019, decision.year + 1)
        )
        frames = []
        for path in paths:
            if not path.exists():
                continue
            cache = getattr(self, "_cache")
            key = str(path)
            if key not in cache:
                cache[key] = validate_underlying_frame(
                    pd.read_csv(path, dtype={"etf_code": str, "source_id": str}),
                    expected_code=code_text,
                )
            frames.append(cache[key])
        if not frames:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        combined = validate_underlying_frame(
            pd.concat(frames, ignore_index=True),
            expected_code=code_text,
        )
        visible = _visible_at(combined, decision)
        return visible.reset_index(drop=True).copy()
