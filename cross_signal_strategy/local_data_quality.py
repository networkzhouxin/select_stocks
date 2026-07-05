# -*- coding: utf-8 -*-
"""JoinQuant log versus local-data diagnostics for cross_signal_strategy."""

from __future__ import annotations

import re
from typing import Callable, Iterable, List, Mapping, Sequence


RICH_INDICATOR_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) 09:35:00 .*? -\s+([0-9]{6})\.(?:XSHG|XSHE) "
    r"buy=(-?\d+) rev=(-?\d+) loc=(-?\d+) trend=(-?\d+) vol=(-?\d+) sell=(-?\d+) "
    r"close=([0-9.]+)"
)

CROSS_FLAG_NAMES = [
    "rsi6_cross_rsi12_up",
    "rsi6_cross_rsi24_up",
    "macd_cross_up",
    "kdj_k_cross_up",
    "kdj_j_cross_up",
    "rsi6_cross_rsi12_down",
    "rsi6_cross_rsi24_down",
    "macd_cross_down",
    "kdj_k_cross_down",
    "kdj_j_cross_down",
]

CROSS_FLAG_LABELS = {
    "RSI12_UP": "rsi6_cross_rsi12_up",
    "RSI24_UP": "rsi6_cross_rsi24_up",
    "MACD_UP": "macd_cross_up",
    "KDJ_K_UP": "kdj_k_cross_up",
    "KDJ_J_UP": "kdj_j_cross_up",
    "RSI12_DOWN": "rsi6_cross_rsi12_down",
    "RSI24_DOWN": "rsi6_cross_rsi24_down",
    "MACD_DOWN": "macd_cross_down",
    "KDJ_K_DOWN": "kdj_k_cross_down",
    "KDJ_J_DOWN": "kdj_j_cross_down",
}

CROSS_FLAG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) 09:35:00 .*? -\s+([0-9]{6})\.(?:XSHG|XSHE) "
    r"rev=(-?\d+) buy=(-?\d+) sell=(-?\d+) "
)


def parse_joinquant_rich_indicator_rows(text: str) -> List[dict]:
    rows = []
    seen = set()
    for line in str(text).splitlines():
        match = RICH_INDICATOR_RE.search(line)
        if not match:
            continue
        date, code, buy, rev, _loc, _trend, _vol, sell, close = match.groups()
        key = (date, code)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "date": date,
            "code": code,
            "buy": int(buy),
            "rev": int(rev),
            "sell": int(sell),
            "close": float(close),
        })
    return rows


def parse_joinquant_cross_flag_rows(text: str) -> List[dict]:
    rows = []
    seen = set()
    for line in str(text).splitlines():
        match = CROSS_FLAG_RE.search(line)
        if not match:
            continue
        date, code, rev, buy, sell = match.groups()
        flags = {}
        for label, field in CROSS_FLAG_LABELS.items():
            flag_match = re.search(r"\b%s=(True|False)\b" % re.escape(label), line)
            if flag_match is None:
                flags = {}
                break
            flags[field] = flag_match.group(1) == "True"
        if not flags:
            continue
        key = (date, code)
        if key in seen:
            continue
        seen.add(key)
        row = {
            "date": date,
            "code": code,
            "rev": int(rev),
            "buy": int(buy),
            "sell": int(sell),
        }
        row.update(flags)
        rows.append(row)
    return rows


def summarize_cross_flag_alignment(
    rows: Iterable[Mapping[str, object]],
    score_provider: Callable[[str, str, int], tuple[Mapping[str, object] | None, str | None]],
    windows: Sequence[int],
    flag_names: Sequence[str] = CROSS_FLAG_NAMES,
) -> dict:
    summary = {
        int(window): {
            "rows": 0,
            "scored_rows": 0,
            "mismatched_rows": 0,
            "flag_mismatches": 0,
            "by_flag": {name: 0 for name in flag_names},
        }
        for window in windows
    }
    for row in rows:
        date = str(row["date"])
        code = str(row["code"])
        for window in windows:
            bucket = summary[int(window)]
            bucket["rows"] += 1
            score, reason = score_provider(code, date, int(window))
            if score is None:
                continue
            bucket["scored_rows"] += 1
            row_mismatched = False
            for name in flag_names:
                if name not in row:
                    continue
                if bool(row[name]) == bool(score.get(name)):
                    continue
                row_mismatched = True
                bucket["flag_mismatches"] += 1
                bucket["by_flag"][name] += 1
            if row_mismatched:
                bucket["mismatched_rows"] += 1
    return summary


def find_close_mismatches(rows: Iterable[Mapping[str, object]], adapter, tolerance: float = 0.002) -> List[dict]:
    mismatches = []
    for row in rows:
        date = str(row["date"])
        code = str(row["code"])
        score, reason = adapter.score(code, date, return_reason=True)
        if score is None:
            continue
        jq_close = float(row["close"])
        local_close = float(score["close"])
        diff = abs(jq_close - local_close)
        if diff > tolerance:
            mismatches.append({
                "date": date,
                "code": code,
                "jq_close": jq_close,
                "local_close": local_close,
                "diff": round(diff, 6),
            })
    return mismatches
