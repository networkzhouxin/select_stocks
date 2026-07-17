# -*- coding: utf-8 -*-
"""Audited acquisition for the pre-registered QDII underlying indices.

Raw historical values and historical point-in-time availability are separate
facts.  This module may download and stage the former, but it only creates the
formal six-column contract when a source-specific publication rule has been
approved.  A download timestamp is never treated as ``available_at``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from cross_signal_strategy.research.underlying_market_data import (
    APPROVED_UNDERLYING_ROOT,
    UNDERLYING_SPECS,
    validate_underlying_frame,
)


REQUEST_START = "2018-01-01"
REQUEST_END = "2021-12-31"
SOURCE_START = pd.Timestamp(REQUEST_START)
SOURCE_END = pd.Timestamp(REQUEST_END)
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class AvailabilityEvidenceMissing(ValueError):
    """Raised when a final value lacks an approved historical availability rule."""


@dataclass(frozen=True)
class SourcePlan:
    etf_code: str
    source_id: str
    provider: str
    locator: str
    value_source_url: str
    request_start: str = REQUEST_START
    request_end: str = REQUEST_END


@dataclass(frozen=True)
class AvailabilityPolicy:
    policy_id: str
    timezone: str
    hour: int
    minute: int
    evidence_url: str
    evidence_note: str


SOURCE_PLANS: Mapping[str, SourcePlan] = {
    "513100": SourcePlan(
        "513100",
        "NDX",
        "FRED",
        "NASDAQ100",
        "https://fred.stlouisfed.org/series/NASDAQ100",
    ),
    "513500": SourcePlan(
        "513500",
        "SPX",
        "FRED",
        "SP500",
        "https://fred.stlouisfed.org/series/SP500",
    ),
    "513050": SourcePlan(
        "513050",
        "H30533",
        "CSI",
        "H30533",
        "https://www.csindex.com.cn/csindex-home/perf/index-perf",
    ),
    "513880": SourcePlan(
        "513880",
        "N225",
        "FRED",
        "NIKKEI225",
        "https://fred.stlouisfed.org/series/NIKKEI225",
    ),
}


# Only policies backed by an explicit publisher schedule belong here.  SPX and
# H30533 intentionally remain absent until equivalent evidence is archived.
APPROVED_AVAILABILITY_POLICIES: Mapping[str, AvailabilityPolicy] = {
    "513100": AvailabilityPolicy(
        policy_id="nasdaq_final_correction_cutoff_1715_et",
        timezone="America/New_York",
        hour=17,
        minute=15,
        evidence_url=(
            "https://indexes.nasdaq.com/docs/"
            "Nasdaq_Index_Methodology_Guide.pdf"
        ),
        evidence_note=(
            "Use the end-of-day correction cutoff, not the ordinary market "
            "close, as the conservative final-value availability boundary."
        ),
    ),
    "513880": AvailabilityPolicy(
        policy_id="nikkei_daily_value_after_close_1600_jst",
        timezone="Asia/Tokyo",
        hour=16,
        minute=0,
        evidence_url=(
            "https://indexes.nikkei.co.jp/nkave/archives/file/"
            "update_schedule_en.pdf"
        ),
        evidence_note=(
            "Official schedule states daily Nikkei 225 values are updated "
            "after the TSE close, at approximately 16:00 JST."
        ),
    ),
}


def _code_text(value: object) -> str:
    return str(value).split(".")[0]


def normalize_raw_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an official close series without filling missing sessions."""
    required = {"session_date", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("raw source history missing columns: %s" % ", ".join(missing))

    normalized = frame.loc[:, ["session_date", "close"]].copy()
    normalized["session_date"] = pd.to_datetime(
        normalized["session_date"], errors="coerce"
    ).dt.normalize()
    if normalized["session_date"].isna().any():
        raise ValueError("raw source history contains invalid session_date")
    if (
        (normalized["session_date"] < SOURCE_START).any()
        or (normalized["session_date"] > SOURCE_END).any()
    ):
        raise ValueError("raw source history violates the 2018-2021 source boundary")

    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    if normalized["close"].isna().any() or (normalized["close"] <= 0).any():
        raise ValueError("raw source history requires positive close values")
    if normalized["session_date"].duplicated().any():
        raise ValueError("raw source history contains duplicate session_date")
    return normalized.sort_values("session_date").reset_index(drop=True)


def parse_fred_csv(text: str, series_id: str) -> pd.DataFrame:
    """Parse one FRED archival CSV and drop explicit holiday blanks."""
    source = pd.read_csv(StringIO(str(text)))
    if source.shape[1] < 2:
        raise ValueError("FRED CSV must contain a date and value column")
    date_column = "observation_date" if "observation_date" in source else source.columns[0]
    if series_id not in source.columns:
        raise ValueError("FRED CSV missing series column: %s" % series_id)
    raw = pd.DataFrame({
        "session_date": source[date_column],
        "close": pd.to_numeric(source[series_id], errors="coerce"),
    }).dropna(subset=["close"])
    return normalize_raw_history(raw)


def normalize_csindex_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract the official CSI session date and final close columns."""
    if "日期" not in frame.columns or "收盘" not in frame.columns:
        raise ValueError("CSI history must contain 日期 and 收盘 columns")
    return normalize_raw_history(pd.DataFrame({
        "session_date": frame["日期"],
        "close": frame["收盘"],
    }))


def fetch_fred_history(
    code: str,
    http_get: Callable | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch one locked FRED series using only the approved source dates."""
    code_text = _code_text(code)
    plan = SOURCE_PLANS.get(code_text)
    if plan is None or plan.provider != "FRED":
        raise ValueError("code is not configured for FRED: %s" % code_text)
    if http_get is None:
        import requests

        http_get = requests.get
    response = http_get(
        FRED_CSV_URL,
        params={
            "id": plan.locator,
            "cosd": plan.request_start,
            "coed": plan.request_end,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_fred_csv(response.text, series_id=plan.locator)


def fetch_csindex_history(fetcher: Callable | None = None) -> pd.DataFrame:
    """Fetch H30533 from the CSI endpoint through AKShare's narrow adapter."""
    plan = SOURCE_PLANS["513050"]
    if fetcher is None:
        import akshare as ak

        fetcher = ak.stock_zh_index_hist_csindex
    source = fetcher(
        symbol=plan.locator,
        start_date=plan.request_start.replace("-", ""),
        end_date=plan.request_end.replace("-", ""),
    )
    return normalize_csindex_history(source)


def collect_raw_sources(
    http_get: Callable | None = None,
    csindex_fetcher: Callable | None = None,
) -> dict[str, pd.DataFrame]:
    """Download all four raw histories without creating availability times."""
    return {
        "513100": fetch_fred_history("513100", http_get=http_get),
        "513500": fetch_fred_history("513500", http_get=http_get),
        "513050": fetch_csindex_history(fetcher=csindex_fetcher),
        "513880": fetch_fred_history("513880", http_get=http_get),
    }


def apply_approved_availability_policy(
    code: str,
    raw_history: pd.DataFrame,
) -> pd.DataFrame:
    """Create contract rows only when a publication rule is approved."""
    code_text = _code_text(code)
    plan = SOURCE_PLANS.get(code_text)
    if plan is None:
        raise ValueError("unsupported underlying source code: %s" % code_text)
    policy = APPROVED_AVAILABILITY_POLICIES.get(code_text)
    if policy is None:
        raise AvailabilityEvidenceMissing(
            "%s has no approved historical final-value availability policy" % code_text
        )

    raw = normalize_raw_history(raw_history)
    available_at = []
    for session_date in raw["session_date"]:
        local = pd.Timestamp(session_date).tz_localize(policy.timezone)
        local += pd.Timedelta(hours=policy.hour, minutes=policy.minute)
        available_at.append(local.tz_convert("UTC"))
    contract = pd.DataFrame({
        "etf_code": code_text,
        "source_id": plan.source_id,
        "session_date": raw["session_date"],
        "available_at": available_at,
        "close": raw["close"],
        "is_final": True,
    })
    return validate_underlying_frame(contract, expected_code=code_text)


def build_contract_bundle(
    raw_histories: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build all four contract frames atomically or report every blocker."""
    expected = set(SOURCE_PLANS)
    supplied = {_code_text(code) for code in raw_histories}
    if supplied != expected:
        raise ValueError(
            "raw source bundle must contain exactly: %s" % ",".join(sorted(expected))
        )
    blocked = sorted(expected.difference(APPROVED_AVAILABILITY_POLICIES))
    if blocked:
        raise AvailabilityEvidenceMissing(
            "formal bundle blocked by missing availability evidence: %s"
            % ",".join(blocked)
        )
    return {
        code: apply_approved_availability_policy(code, raw_histories[code])
        for code in sorted(expected)
    }


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def write_raw_staging_bundle(
    raw_histories: Mapping[str, pd.DataFrame],
    staging_root: str | Path,
    acquired_at: str,
) -> Path:
    """Write auditable raw files outside the immutable approved data root."""
    root = Path(staging_root).expanduser().resolve()
    approved = Path(APPROVED_UNDERLYING_ROOT).expanduser().resolve()
    if _is_inside(root, approved):
        raise ValueError("raw staging cannot target the approved immutable root")

    acquired = pd.Timestamp(acquired_at)
    if acquired.tzinfo is None:
        raise ValueError("acquired_at must be timezone-aware metadata")
    expected = set(SOURCE_PLANS)
    supplied = {_code_text(code) for code in raw_histories}
    if supplied != expected:
        raise ValueError(
            "raw source bundle must contain exactly: %s" % ",".join(sorted(expected))
        )

    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_manifest = {}
    for code in sorted(expected):
        normalized = normalize_raw_history(raw_histories[code]).copy()
        normalized["session_date"] = normalized["session_date"].dt.strftime("%Y-%m-%d")
        payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
        relative = Path("raw") / (code + ".csv")
        path = root / relative
        path.write_bytes(payload)
        file_manifest[code] = {
            "path": relative.as_posix(),
            "rows": len(normalized),
            "first_session": normalized["session_date"].iloc[0],
            "last_session": normalized["session_date"].iloc[-1],
            "sha256": sha256(payload).hexdigest(),
            "source": asdict(SOURCE_PLANS[code]),
            "availability_policy": (
                asdict(APPROVED_AVAILABILITY_POLICIES[code])
                if code in APPROVED_AVAILABILITY_POLICIES
                else None
            ),
        }

    blocked = sorted(expected.difference(APPROVED_AVAILABILITY_POLICIES))
    manifest = {
        "acquired_at": acquired.isoformat(),
        "acquired_at_is_not_available_at": True,
        "data_scope": "2018_warmup_plus_2019_2021_training_only",
        "formal_publishable": not blocked,
        "blocked_codes": blocked,
        "files": file_manifest,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def run_source_acquisition(
    staging_root: str | Path,
    acquired_at: str,
    http_get: Callable | None = None,
    csindex_fetcher: Callable | None = None,
) -> Path:
    """Collect the locked sources and write a non-publishable raw staging bundle."""
    frames = collect_raw_sources(
        http_get=http_get,
        csindex_fetcher=csindex_fetcher,
    )
    return write_raw_staging_bundle(
        frames,
        staging_root=staging_root,
        acquired_at=acquired_at,
    )
